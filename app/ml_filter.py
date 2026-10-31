"""Fast in-process anomaly filter: deterministic signatures plus an Isolation Forest.

Scoring one request is pure CPU work with no I/O, so it is safe to run on the request path.
The forest is loaded once from ``ml_forest.json``. It was fit on benign steps from the train
holdout servers, plus signature-clean suspicious-benign steps and a small synthetic rate
supplement. ``scripts/fit_isolation_forest.py`` is what rebuilds that file.
The compiled trees are flat Python lists so scoring avoids scikit-learn's per-call overhead,
and scores are memoized on quantized feature vectors.
"""

from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from sklearn.ensemble import IsolationForest

# Compiled trees loaded by FastFilter. Rebuilt only by scripts/fit_isolation_forest.py.
FOREST_PATH = Path(__file__).with_name("ml_forest.json")

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
    _sig(("'",), r"'[\s+]*(?:or|and)[\s+]*'?\d+'?[\s+]*=[\s+]*'?\d+"),
    # Quoted tautologies such as ' OR 'a'='a' and " OR "ab"="ab", including a closing paren.
    _sig(
        ("' or ", "'or", "' and ", "'and", '" or ', '"or', '" and ', '"and'),
        r"""['\"][\s+]*\)?[\s+]*(?:or|and)[\s+]*\(?[\s+]*['\"][^'\"]{0,40}['\"][\s+]*=[\s+]*['\"][^'\"]{0,40}['\"]""",
    ),
    _sig(("'",), r"'[\s+]*(?:--|#|;)"),
    _sig(
        (" or ", "+or+", "+or ", " and ", "+and+", "+and "),
        r"(?:^|[\s+'\"=+])(?:or|and)[\s+]+\d+[\s+]*=[\s+]*\d+",
    ),
    _sig((";", "||", "&&", "|", "cmd", "powershell"), r"(?:;|\|\||&&|\|)\s*(?:cat|ls|id|whoami|curl|wget|nc|bash|sh|cmd|powershell|calc)(?:\.exe)?\b|\b(?:cmd|powershell)(?:\.exe)?\s+(?:/c|-enc|-command)\b"),
    _sig(
        ("<", "javascript:", "onerror", "onload", "onmouseover", "onfocus"),
        r"<\s*script\b|javascript:|on(?:error|load|mouseover|focus|toggle)\s*=|<\s*(?:svg|img|iframe|body)\b",
    ),
    _sig(("${",), r"\$\{\s*jndi\s*:"),
    _sig(("${",), r"\$\{[^{}]{0,20}\*[^{}]{0,20}\}"),  # template math such as ${7*7}
    _sig(("{{",), r"\{\{.{1,60}\}\}"),
    _sig(("#{",), r"#\{.{1,40}\}"),
    _sig(("sleep", "benchmark"), r"\b(?:sleep|benchmark|pg_sleep)\s*\(\s*\d"),
    _sig(("$",), r"\{\s*\"?\$(?:where|ne|gt|regex)\"?\s*:"),
    _sig(("o:",), r"\bo:\d+:"),  # PHP serialized object, including a quoted class name
    _sig(("gasv", "aced0005", "ro0ab"), r"gasv[a-z0-9+/=]{4,}|aced0005|ro0ab"),
    _sig(
        ("http://", "https://", "169.254"),
        r"https?://(?:127\.0\.0\.1|localhost|0\.0\.0\.0|\[::1\]|169\.254\.169\.254|metadata\.google(?:\.internal)?"
        r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})",
    ),
)

# Statement-level SQL words. Everyday words and normal list parameters
# (and, or, limit, offset, order, group, user, key) are omitted: they show up
# in ordinary requests. A hit only raises a low suspicious score.
_SQL_KEYWORD = re.compile(
    r"\b(?:alter|ascii|benchmark|cast|char|concat|convert|create|cursor|declare|delay|delete|drop"
    r"|dumpfile|exec|execute|from|grant|group_concat|having|information_schema|insert|into|join"
    r"|load_file|merge|nchar|nvarchar|opendatasource|openrowset|outfile|pg_sleep|procedure|revoke"
    r"|row_number|rownum|schema|select|shutdown|sleep|sp_executesql|substring|syscolumns|sysobjects"
    r"|table|truncate|union|update|values|varchar|waitfor|where|xp_cmdshell)\b"
)

# A line break is log forging only when the next line is a log stamp: a level word,
# a bracketed level, or a timestamp followed by a level. A bare newline is not.
_LOG_LEVEL = r"(?:debug|info|warn(?:ing)?|error|fatal|critical|trace|notice|audit)"
# \\n and \\r match the escaped newline inside a JSON body. \n and \r match a raw break.
_LOG_STAMP_AFTER_BREAK = re.compile(
    rf"(?:%0d%0a|%0a%0d|%0d|%0a|\\r\\n|\\n|\\r|\r\n|\n|\r)\s*(?:"
    rf"\[\s*{_LOG_LEVEL}\s*\]"
    rf"|{_LOG_LEVEL}\b"
    rf"|(?:\d{{4}}[-/]\d{{1,2}}[-/]\d{{1,2}}|\d{{1,2}}:\d{{2}}:\d{{2}}).{{0,48}}?{_LOG_LEVEL}\b)",
    re.I,
)
_IDENTITY_CRLF = re.compile(
    rf'"(?:user(?:name)?|email|login|name)"\s*:\s*"[^"\\]{{0,80}}(?:\\n|\\r|\r|\n)\s*'
    rf"(?:\[\s*{_LOG_LEVEL}\s*\]|{_LOG_LEVEL}\b)",
    re.I,
)
_LOWER_LOOKUP = re.compile(r"\$\{(?:lower|upper)::?-?([a-z0-9]{1,8})\}")
_EMPTY_LOOKUP = re.compile(r"\$\{::-?([a-z0-9]{1,8})\}")

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
    if "%" not in text and "%" not in text.lower():
        return variants[0]
    current = text
    for _ in range(rounds):
        decoded = unquote(current)
        if decoded == current:
            break
        variants.append(decoded.lower())
        current = decoded
    return "\n".join(variants)


def unfold_lookups(text: str, rounds: int = 4) -> str:
    """Expand `${lower:j}` style wrappers so an obfuscated lookup becomes readable text.

    This only rewrites the string. It does not resolve the lookup or contact any host.
    """
    current = text.lower()
    for _ in range(rounds):
        nxt = _LOWER_LOOKUP.sub(lambda match: match.group(1), current)
        nxt = _EMPTY_LOOKUP.sub(lambda match: match.group(1), nxt)
        if nxt == current:
            break
        current = nxt
    return current


def signature_text(text: str) -> str:
    """URL-decoded text plus a lookup-unfolded copy when the unfold changed anything."""
    decoded = decode_variants(text)
    unfolded = unfold_lookups(decoded)
    if unfolded != decoded:
        return decoded + "\n" + unfolded
    return decoded


def describe_decodings(value: str) -> str:
    """Return URL, base64, hex, and lookup-wrapper decodings. Never executes the value."""
    text = str(value)[:500]
    lines: list[str] = []
    current = text
    for i in range(3):
        decoded = unquote(current)
        if decoded == current:
            break
        lines.append(f"url-{i + 1}: {decoded[:300]}")
        current = decoded
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[A-Za-z0-9+/]{8,}={0,2}", compact) and len(compact) % 4 == 0:
        try:
            shown = base64.b64decode(compact, validate=True).decode("utf-8", errors="replace")[:300]
            lines.append(f"base64: {shown}")
        except ValueError:
            pass
    hexbody = compact[2:] if compact.lower().startswith("0x") else compact
    if re.fullmatch(r"[0-9a-fA-F]{8,}", hexbody) and len(hexbody) % 2 == 0:
        try:
            shown = bytes.fromhex(hexbody).decode("utf-8", errors="replace")[:300]
            lines.append(f"hex: {shown}")
        except ValueError:
            pass
    unfolded = unfold_lookups(text)
    if unfolded != text.lower():
        lines.append(f"lookup: {unfolded[:300]}")
    return "\n".join(lines) if lines else "no decoding applied"


def log_forging_hits(query: str, body_text: str, headers: dict[str, str] | None) -> int:
    """Count line breaks whose next line is a log stamp.

    A newline that only continues a value, and a raw newline inside a comment or
    description, are ignored. The stamp is a level word, a bracketed level, or a
    timestamp followed by a level.
    """
    hits = 0
    fields = [query]
    if headers:
        fields.extend(headers.values())
    for text in fields:
        if _LOG_STAMP_AFTER_BREAK.search(text):
            hits += 1
    if body_text and (_LOG_STAMP_AFTER_BREAK.search(body_text) or _IDENTITY_CRLF.search(body_text)):
        hits += 1
    return hits


def sql_keyword_hits(text: str) -> int:
    """Count whole-word SQL statement keywords. Substrings inside ordinary words do not count."""
    return len(_SQL_KEYWORD.findall(text))


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

    @classmethod
    def load(cls, path: Path) -> CompiledIsolationForest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        compiled = cls.__new__(cls)
        compiled._trees = [
            _CompiledTree(
                left=tree["left"],
                right=tree["right"],
                feature=tree["feature"],
                threshold=tree["threshold"],
                leaf_depth_bonus=tree["leaf_depth_bonus"],
            )
            for tree in payload["trees"]
        ]
        compiled._norm = float(payload["norm"])
        return compiled

    def dump(self, path: Path) -> None:
        payload = {
            "norm": self._norm,
            "trees": [
                {
                    "left": tree.left,
                    "right": tree.right,
                    "feature": tree.feature,
                    "threshold": tree.threshold,
                    "leaf_depth_bonus": tree.leaf_depth_bonus,
                }
                for tree in self._trees
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

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


def _benign_training_set(n: int = 400, seed: int = 1337) -> np.ndarray:
    """Small synthetic supplement so window rate still varies around the real rows."""
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


def fit_compiled_forest(
    rows: np.ndarray, *, n_estimators: int = 64, max_samples: int = 2048, seed: int = 1337
) -> CompiledIsolationForest:
    """Fit an isolation forest and return its compiled trees. This is not called on startup."""
    samples = min(max_samples, len(rows))
    forest = IsolationForest(n_estimators=n_estimators, max_samples=samples, random_state=seed)
    forest.fit(rows)
    return CompiledIsolationForest(forest)


class FastFilter:
    """Synchronous request scorer combining signatures, schema checks, and a loaded Isolation Forest.

    The trees are read from ``app/ml_forest.json``. They were fit on benign steps from the
    train holdout servers, signature-clean suspicious-benign steps, and a small synthetic
    rate supplement. Rebuild that file with ``scripts/fit_isolation_forest.py``.
    """

    def __init__(
        self,
        *,
        suspicious_threshold: float,
        hard_block_threshold: float,
        bola_distinct_threshold: int,
        cache_size: int = 8192,
        forest_path: Path | None = None,
    ) -> None:
        self.suspicious_threshold = suspicious_threshold
        self.hard_block_threshold = hard_block_threshold
        self.bola_distinct_threshold = bola_distinct_threshold
        path = forest_path or FOREST_PATH
        self._compiled = CompiledIsolationForest.load(path)
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
        headers: dict[str, str] | None = None,
    ) -> RequestFeatures:
        body_text = body[:INSPECT_BODY_BYTES].decode("utf-8", errors="replace") if body else ""
        target = f"{path}?{query}" if query else path
        haystack = signature_text(target)
        if body_text:
            haystack += "\n" + signature_text(body_text)
        if headers:
            cookie = next((v for k, v in headers.items() if k.lower() == "cookie"), "")
            if cookie:
                haystack += "\n" + signature_text(cookie)

        traversal_hits = sum(sig.count(haystack) for sig in TRAVERSAL_SIGNATURES)
        injection_hits = sum(sig.count(haystack) for sig in INJECTION_SIGNATURES)
        unfolded = unfold_lookups(haystack)
        if unfolded != haystack and "jndi:" in unfolded:
            injection_hits += 1
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
            log_forging_hits=log_forging_hits(query, body_text, headers),
            sql_keyword_hits=sql_keyword_hits(haystack),
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

    def evaluate(
        self,
        f: RequestFeatures,
        *,
        distinct_objects: int = 0,
        identity_mismatch: bool = False,
        enum_mismatch: bool = False,
    ) -> FastVerdict:
        scores: list[tuple[float, str, str | None]] = []

        if identity_mismatch:
            scores.append((0.7, "caller principal does not match a user-scoped identifier", "bola"))
        if enum_mismatch:
            scores.append((0.7, "query parameter value is outside the declared enum", "schema_violation"))
        if f.traversal_hits:
            scores.append((0.95, f"path traversal signature x{f.traversal_hits}", "path_traversal"))
        if f.injection_hits:
            # Stay under the hard-block threshold so the request is still forwarded and the agent decides.
            score = min(0.6 + 0.15 * f.injection_hits, 0.85)
            scores.append((score, f"injection signature x{f.injection_hits}", "injection"))
        elif f.sql_keyword_hits:
            # A reserved word alone is a weak signal. Send it to the agent; do not hard-block.
            scores.append((0.5, f"sql keyword x{f.sql_keyword_hits}", "injection"))
        if f.log_forging_hits:
            score = min(0.6 + 0.15 * f.log_forging_hits, 0.85)
            scores.append((score, f"log forging signature x{f.log_forging_hits}", "log_forging"))
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


WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def needs_inline_hold(features: RequestFeatures, *, method: str = "", watched: bool = False) -> bool:
    """True when forwarding now could change the store before the model can stop it.

    Comments, searches, and public files are reviewed in the background instead.
    Identity and enum mismatches are denied on the fast path and do not need a hold.
    """
    if features.injection_hits or features.log_forging_hits:
        return True
    if features.privileged_fields:
        return True
    return watched and method.upper() in WRITE_METHODS
