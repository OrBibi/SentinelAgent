"""Fast in-process anomaly filter: deterministic signatures plus an Isolation Forest.

Everything here is pure CPU work with no I/O, so it is safe to run on the request path.
The fitted forest is compiled into flat Python lists so scoring a single request avoids
scikit-learn's per-call overhead (milliseconds), and scores are memoized on quantized
feature vectors so repeated traffic shapes cost a dictionary lookup.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import unquote

import numpy as np
from sklearn.ensemble import IsolationForest

from app.models import FastVerdict, RequestFeatures, Tier

EULER_GAMMA = 0.5772156649015329
SPECIAL_CHARS = "<>'\"%;(){}|\\$`*!^~[]"
_STRIP_SPECIAL = str.maketrans("", "", SPECIAL_CHARS)
METHOD_CODES = {"GET": 0, "HEAD": 0, "OPTIONS": 0, "POST": 1, "PUT": 2, "PATCH": 2, "DELETE": 3}
# Body prefix scanned by signatures; larger bodies are still forwarded and length-scored.
INSPECT_BODY_BYTES = 16_384

@dataclass(frozen=True, slots=True)
class Signature:
    """A regex guarded by literal triggers; the regex only runs if a trigger substring is present."""

    triggers: tuple[str, ...]
    pattern: re.Pattern[str]

    def count(self, text: str) -> int:
        if not any(t in text for t in self.triggers):
            return 0
        return len(self.pattern.findall(text))


def _sig(triggers: tuple[str, ...], pattern: str) -> Signature:
    return Signature(triggers, re.compile(pattern, re.MULTILINE))


# Path traversal signatures, matched against the lowercased raw and iteratively-decoded text.
TRAVERSAL_SIGNATURES: tuple[Signature, ...] = (
    _sig(("..",), r"(?:^|[/\\=])\.\.(?:[/\\]|$)"),  # ../ or ..\ as a path segment
    _sig(("%2e%2e",), r"%2e%2e(?:%2f|%5c|/|\\)"),  # URL-encoded dot-dot followed by a separator
    _sig(("..",), r"\.\.(?:%2f|%5c)"),  # dot-dot with encoded separator
    _sig(("%c0", "%c1", "%e0"), r"%c0%ae|%c0%af|%c1%9c|%e0%80%ae"),  # overlong UTF-8 encodings
    _sig(("%252e",), r"%252e%252e"),  # double-encoded dot-dot
    _sig(
        ("etc/", "proc/", "windows/"),
        r"(?:^|[/\\=])(?:etc/(?:passwd|shadow|hosts)|proc/self/|windows/(?:win\.ini|system32))",
    ),
    _sig(
        ("boot.ini", "win.ini", ".ssh/", ".env", ".git/"),
        r"(?:^|[/\\=])(?:boot\.ini|win\.ini|\.ssh/|\.env(?:$|[/?&])|\.git/(?:config|head))",
    ),
    _sig(("%00", "\x00"), r"%00|\x00"),  # null byte truncation
    _sig(("://",), r"(?:^|=)(?:file|php|zip|expect)://"),  # stream-wrapper based file access
)

INJECTION_SIGNATURES: tuple[Signature, ...] = (
    _sig(("union",), r"\bunion\b[\s/*+]+(?:all[\s/*+]+)?select\b"),
    _sig(("'",), r"'\s*(?:or|and)\s*'?\d+'?\s*=\s*'?\d+"),
    _sig(("'",), r"'\s*(?:--|#|;)"),
    _sig((";", "||", "&&"), r"(?:;|\|\||&&)\s*(?:cat|ls|id|whoami|curl|wget|nc|bash|sh)\b"),
    _sig(("<", "javascript:", "onerror", "onload"), r"<\s*script\b|javascript:|onerror\s*=|onload\s*="),
    _sig(("${",), r"\$\{\s*jndi\s*:"),
    _sig(("sleep", "benchmark"), r"\b(?:sleep|benchmark|pg_sleep)\s*\(\s*\d"),
    _sig(("$",), r"\{\s*\"?\$(?:where|ne|gt|regex)\"?\s*:"),
)

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_NUMERIC_RE = re.compile(r"^\d{1,20}$")
_HEX_ID_RE = re.compile(r"^[0-9a-f]{24}$", re.I)


def is_object_id(segment: str) -> bool:
    return bool(_NUMERIC_RE.match(segment) or _UUID_RE.match(segment) or _HEX_ID_RE.match(segment))


def heuristic_template(path: str) -> tuple[str, str | None]:
    """Replace object-id-like segments with `{id}`; return the template and the last object id."""
    segments = path.split("/")
    last_id: str | None = None
    for i, seg in enumerate(segments):
        if seg and is_object_id(seg):
            last_id = seg
            segments[i] = "{id}"
    return "/".join(segments) or "/", last_id


def decode_variants(text: str, rounds: int = 3) -> str:
    """Return the raw text plus up to `rounds` URL-decoded variants, lowercased, joined by newlines."""
    variants = [text.lower()]
    if "%" not in text:
        return variants[0]
    current = text
    for _ in range(rounds):
        decoded = unquote(current)
        if decoded == current:
            break
        variants.append(decoded.lower())
        current = decoded
    return "\n".join(variants)


def _average_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER_GAMMA) - 2.0 * (n - 1.0) / n


@dataclass(slots=True)
class _CompiledTree:
    left: list[int]
    right: list[int]
    feature: list[int]
    threshold: list[float]
    leaf_depth_bonus: list[float]


class CompiledIsolationForest:
    """Pure-Python evaluator equivalent to `IsolationForest.score_samples` for a single row."""

    def __init__(self, forest: IsolationForest) -> None:
        self._trees: list[_CompiledTree] = []
        for est, feats in zip(forest.estimators_, forest.estimators_features_):
            tree = est.tree_
            feat_map = [int(f) for f in feats]
            self._trees.append(
                _CompiledTree(
                    left=tree.children_left.tolist(),
                    right=tree.children_right.tolist(),
                    feature=[feat_map[f] if f >= 0 else -1 for f in tree.feature.tolist()],
                    threshold=tree.threshold.tolist(),
                    leaf_depth_bonus=[_average_path_length(int(n)) for n in tree.n_node_samples.tolist()],
                )
            )
        self._norm = len(self._trees) * _average_path_length(int(forest.max_samples_))

    def anomaly_score(self, row: list[float] | tuple[float, ...]) -> float:
        """Return s in (0, 1); values above 0.5 indicate anomalies (original paper's definition)."""
        total = 0.0
        for t in self._trees:
            node = 0
            depth = 0
            left, right, feature, threshold = t.left, t.right, t.feature, t.threshold
            while left[node] != -1:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
                depth += 1
            total += depth + t.leaf_depth_bonus[node]
        return 2.0 ** (-total / self._norm)


def _benign_training_set(n: int = 4000, seed: int = 1337) -> np.ndarray:
    """Synthetic profile of ordinary API traffic used to fit the forest at startup."""
    rng = np.random.default_rng(seed)
    methods = rng.choice([0, 1, 2, 3], size=n, p=[0.65, 0.2, 0.1, 0.05])
    has_body = methods > 0
    body_len = np.where(has_body, rng.lognormal(mean=5.0, sigma=1.0, size=n), 0.0)
    query_len = np.where(rng.random(n) < 0.5, rng.integers(5, 120, size=n), 0)
    special_ratio = np.clip(rng.normal(0.02, 0.015, size=n), 0.0, 0.08)
    window_rate = rng.integers(1, 30, size=n)
    depth = rng.integers(1, 6, size=n)
    id_segments = rng.choice([0, 1, 2], size=n, p=[0.5, 0.4, 0.1])
    zeros = np.zeros(n)
    return np.column_stack(
        [
            np.log1p(body_len),
            np.log1p(query_len),
            special_ratio,
            zeros,  # traversal hits
            zeros,  # injection hits
            np.log1p(window_rate),
            methods,
            depth,
            id_segments,
        ]
    )


class FastFilter:
    """Synchronous request scorer combining signatures, schema checks, and Isolation Forest."""

    def __init__(
        self,
        *,
        suspicious_threshold: float,
        hard_block_threshold: float,
        bola_distinct_threshold: int,
        n_estimators: int = 64,
        seed: int = 1337,
        cache_size: int = 8192,
    ) -> None:
        self.suspicious_threshold = suspicious_threshold
        self.hard_block_threshold = hard_block_threshold
        self.bola_distinct_threshold = bola_distinct_threshold
        self.forest = IsolationForest(n_estimators=n_estimators, max_samples=256, random_state=seed)
        self.forest.fit(_benign_training_set(seed=seed))
        self._compiled = CompiledIsolationForest(self.forest)
        # Feature vectors are quantized, so repeated traffic shapes hit this cache.
        self._cached_score = lru_cache(maxsize=cache_size)(self._compiled.anomaly_score)

    # ------------------------------------------------------------------ features

    @staticmethod
    def extract_features(
        method: str,
        path: str,
        query: str,
        body: bytes,
        *,
        window_rate: int = 0,
        route_known: bool | None = None,
        matched_route: str | None = None,
        undeclared_fields: list[str] | None = None,
        privileged_fields: list[str] | None = None,
    ) -> RequestFeatures:
        body_text = body[:INSPECT_BODY_BYTES].decode("utf-8", errors="replace") if body else ""
        target = f"{path}?{query}" if query else path
        haystack = decode_variants(target)
        if body_text:
            haystack += "\n" + decode_variants(body_text, rounds=1)

        traversal_hits = sum(sig.count(haystack) for sig in TRAVERSAL_SIGNATURES)
        injection_hits = sum(sig.count(haystack) for sig in INJECTION_SIGNATURES)
        sample = target + body_text[:2048]
        special = len(sample) - len(sample.translate(_STRIP_SPECIAL))
        id_segments = sum(1 for seg in path.split("/") if seg and is_object_id(seg))

        return RequestFeatures(
            method=method.upper(),
            path=path,
            query=query,
            body_length=len(body),
            query_length=len(query),
            special_char_ratio=special / max(len(sample), 1),
            traversal_hits=traversal_hits,
            injection_hits=injection_hits,
            numeric_id_segments=id_segments,
            window_rate=window_rate,
            route_known=route_known,
            matched_route=matched_route,
            undeclared_fields=undeclared_fields or [],
            privileged_fields=privileged_fields or [],
        )

    @staticmethod
    def to_vector(f: RequestFeatures) -> tuple[float, ...]:
        return (
            round(math.log1p(f.body_length), 1),
            round(math.log1p(f.query_length), 1),
            round(f.special_char_ratio, 2),
            float(f.traversal_hits),
            float(f.injection_hits),
            round(math.log1p(f.window_rate), 1),
            float(METHOD_CODES.get(f.method, 1)),
            float(f.path.count("/")),
            float(f.numeric_id_segments),
        )

    # ------------------------------------------------------------------ scoring

    def forest_risk(self, f: RequestFeatures) -> float:
        """Map the isolation score to 0..1 risk; 0.5 (inlier boundary) maps to 0, 0.75+ to 1."""
        s = self._cached_score(self.to_vector(f))
        if not math.isfinite(s):
            return 0.0
        return min(max((s - 0.5) / 0.25, 0.0), 1.0)

    def evaluate(self, f: RequestFeatures, *, distinct_objects: int = 0) -> FastVerdict:
        scores: list[tuple[float, str, str | None]] = []

        if f.traversal_hits:
            scores.append((0.95, f"path traversal signature x{f.traversal_hits}", "path_traversal"))
        if f.injection_hits:
            # Stay under the hard-block threshold so the request is still forwarded and the agent decides.
            score = min(0.6 + 0.15 * f.injection_hits, 0.85)
            scores.append((score, f"injection signature x{f.injection_hits}", "injection"))
        if f.route_known is False:
            scores.append((0.55, "method/path not declared in OpenAPI schema", "schema_violation"))
        if f.privileged_fields:
            scores.append((0.7, f"undeclared privileged body fields: {', '.join(f.privileged_fields)}", "mass_assignment"))
        elif f.undeclared_fields:
            scores.append((0.5, f"undeclared body fields: {', '.join(f.undeclared_fields[:5])}", "mass_assignment"))
        if distinct_objects >= self.bola_distinct_threshold:
            scores.append((0.7, f"{distinct_objects} distinct object ids on one route", "bola"))
        if f.special_char_ratio > 0.25 and f.query_length + f.body_length > 24:
            scores.append((0.5, f"special char ratio {f.special_char_ratio:.2f}", "obfuscation"))

        risk = self.forest_risk(f)
        if risk > 0:
            scores.append((0.6 * risk, f"isolation forest risk {risk:.2f}", "anomaly"))

        if not scores:
            return FastVerdict(tier=Tier.CLEAN, score=0.0)

        scores.sort(key=lambda s: s[0], reverse=True)
        top_score, _, top_hint = scores[0]
        reasons = [reason for score, reason, _ in scores if score >= 0.1]
        if top_score >= self.hard_block_threshold:
            tier = Tier.HARD_BLOCK
        elif top_score >= self.suspicious_threshold:
            tier = Tier.SUSPICIOUS
        else:
            tier = Tier.CLEAN
        return FastVerdict(tier=tier, score=round(top_score, 4), reasons=reasons, threat_hint=top_hint)
