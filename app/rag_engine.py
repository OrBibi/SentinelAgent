"""OpenAPI ingestion, schema validation, and retrieval over threat patterns and endpoint specs.

Two layers:
- `RouteIndex`: in-memory compiled routes used on the hot path (regex match only, no I/O).
- `VectorStore`: Chroma (or an in-memory fallback) queried only by the async agent worker.

Embeddings come from a local hashing vectorizer, so no model download or network call is needed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlparse

import numpy as np
import yaml
from sklearn.feature_extraction.text import HashingVectorizer

from app.ml_filter import heuristic_template
from app.models import SuspiciousRequest

logger = logging.getLogger("sentinel.rag")

HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
EMBED_DIM = 1024


class OpenAPIError(ValueError):
    """Raised when an uploaded document is not a usable OpenAPI/Swagger schema."""


# ---------------------------------------------------------------------- embeddings


class HashingEmbedder:
    """Deterministic, dependency-free text embeddings (L2-normalized hashed word n-grams)."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self._vec = HashingVectorizer(
            n_features=dim,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            token_pattern=r"(?u)[a-z0-9_{}%.\-]{2,}",
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).toarray().astype(np.float32).tolist()


# ---------------------------------------------------------------------- vector stores


@dataclass(slots=True)
class Hit:
    id: str
    document: str
    metadata: dict[str, Any]
    score: float


class VectorStore(Protocol):
    def upsert(self, ids: list[str], docs: list[str], metas: list[dict[str, Any]]) -> None: ...
    def delete_where(self, where: dict[str, Any]) -> None: ...
    def query(self, text: str, k: int, where: dict[str, Any] | None = None) -> list[Hit]: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    """Brute-force cosine search; used when Chroma is disabled or unavailable."""

    def __init__(self, embedder: HashingEmbedder) -> None:
        self._embedder = embedder
        self._items: dict[str, tuple[str, dict[str, Any], np.ndarray]] = {}
        self._lock = threading.Lock()

    def upsert(self, ids: list[str], docs: list[str], metas: list[dict[str, Any]]) -> None:
        vectors = self._embedder.embed(docs)
        with self._lock:
            for i, d, m, v in zip(ids, docs, metas, vectors):
                self._items[i] = (d, m, np.asarray(v, dtype=np.float32))

    def delete_where(self, where: dict[str, Any]) -> None:
        with self._lock:
            for key in [k for k, (_, m, _) in self._items.items() if _meta_matches(m, where)]:
                del self._items[key]

    def query(self, text: str, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        q = np.asarray(self._embedder.embed([text])[0], dtype=np.float32)
        with self._lock:
            candidates = [(i, d, m, v) for i, (d, m, v) in self._items.items() if _meta_matches(m, where)]
        scored = [Hit(i, d, m, float(np.dot(q, v))) for i, d, m, v in candidates]
        return sorted(scored, key=lambda h: h.score, reverse=True)[:k]

    def count(self) -> int:
        return len(self._items)


class ChromaVectorStore:
    """Persistent Chroma collection fed with precomputed embeddings.

    Chroma's client is not safe for concurrent use, and ingestion (request thread pool)
    overlaps agent retrieval, so every call takes `_lock`.
    """

    def __init__(self, embedder: HashingEmbedder, path: str, collection: str = "sentinel_knowledge") -> None:
        import chromadb  # imported lazily so the proxy still runs without Chroma installed

        self._embedder = embedder
        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            name=collection, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, ids: list[str], docs: list[str], metas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        embeddings = self._embedder.embed(docs)
        with self._lock:
            self._col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    def delete_where(self, where: dict[str, Any]) -> None:
        with self._lock:
            try:
                self._col.delete(where=_chroma_where(where))
            except Exception as exc:  # noqa: BLE001 - an empty match is not fatal to ingestion
                logger.warning("chroma delete skipped: %s", exc)

    def query(self, text: str, k: int, where: dict[str, Any] | None = None) -> list[Hit]:
        embedding = self._embedder.embed([text])
        with self._lock:
            total = self._col.count()
            if total == 0:
                return []
            res = self._col.query(
                query_embeddings=embedding,
                n_results=min(k, total),
                where=_chroma_where(where) if where else None,
            )
        hits: list[Hit] = []
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append(Hit(i, d or "", dict(m or {}), 1.0 - float(dist)))
        return hits

    def count(self) -> int:
        with self._lock:
            return self._col.count()


def _meta_matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    return not where or all(meta.get(k) == v for k, v in where.items())


def _chroma_where(where: dict[str, Any]) -> dict[str, Any]:
    if len(where) <= 1:
        return where
    return {"$and": [{k: v} for k, v in where.items()]}


# ---------------------------------------------------------------------- route index


@dataclass(slots=True)
class ParamSpec:
    name: str
    location: str
    required: bool = False
    type: str | None = None
    enum: list[Any] | None = None


@dataclass(slots=True)
class RouteSpec:
    method: str
    template: str
    source: str
    regex: re.Pattern[str]
    path_param_names: list[str]
    params: list[ParamSpec] = field(default_factory=list)
    body_fields: list[str] = field(default_factory=list)
    secured: bool = False
    summary: str = ""
    operation_id: str = ""

    @property
    def static_weight(self) -> tuple[int, int]:
        """Sort key preferring templates with fewer variables and longer literal parts."""
        return (len(self.path_param_names), -len(self.template))


@dataclass(slots=True)
class RouteMatch:
    template: str
    method_allowed: bool
    path_params: dict[str, str]
    spec: RouteSpec | None
    allowed_methods: list[str]


_PARAM_RE = re.compile(r"\{([^}/]+)\}")


def _compile_template(template: str) -> tuple[re.Pattern[str], list[str]]:
    names: list[str] = []
    parts: list[str] = []
    last = 0
    for m in _PARAM_RE.finditer(template):
        parts.append(re.escape(template[last : m.start()]))
        parts.append(f"(?P<p{len(names)}>[^/]+)")
        names.append(m.group(1))
        last = m.end()
    parts.append(re.escape(template[last:]))
    return re.compile("^" + "".join(parts) + "/?$"), names


class RouteIndex:
    """Thread-safe set of compiled routes grouped by schema source."""

    def __init__(self) -> None:
        self._by_source: dict[str, list[RouteSpec]] = {}
        self._routes: list[RouteSpec] = []
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return bool(self._routes)

    @property
    def sources(self) -> dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._by_source.items()}

    def replace_source(self, source: str, routes: list[RouteSpec]) -> None:
        with self._lock:
            self._by_source[source] = routes
            merged = [r for rs in self._by_source.values() for r in rs]
            self._routes = sorted(merged, key=lambda r: r.static_weight)

    def remove_source(self, source: str) -> bool:
        with self._lock:
            existed = self._by_source.pop(source, None) is not None
            self._routes = sorted((r for rs in self._by_source.values() for r in rs), key=lambda r: r.static_weight)
            return existed

    def match(self, method: str, path: str) -> RouteMatch | None:
        """Return the best route for `path`; `method_allowed` is False if only other methods match.

        The first matching template wins (static routes sort ahead of parameterized ones).
        Every method declared on that template is reported, including when the request method matches.
        """
        method = method.upper()
        hits: list[tuple[RouteSpec, dict[str, str]]] = []
        template: str | None = None
        for r in self._routes:
            found = r.regex.match(path)
            if not found:
                continue
            if template is None:
                template = r.template
            elif r.template != template:
                continue
            params = {name: found.group(f"p{i}") for i, name in enumerate(r.path_param_names)}
            hits.append((r, params))
        if not hits or template is None:
            return None
        allowed = [r.method for r, _ in hits]
        for r, params in hits:
            if r.method == method:
                return RouteMatch(template, True, params, r, allowed)
        return RouteMatch(template, False, hits[0][1], None, allowed)

    def classify(self, method: str, path: str) -> RouteClass:
        """Hot-path helper; `known` is None when no schema is loaded, so schema checks are skipped."""
        if not self.loaded:
            template, object_id = heuristic_template(path)
            return RouteClass(None, template, object_id, None)
        match = self.match(method, path)
        if match is None:
            template, object_id = heuristic_template(path)
            return RouteClass(False, template, object_id, None)
        object_id = next(reversed(match.path_params.values()), None) if match.path_params else None
        return RouteClass(match.method_allowed, match.template, object_id, match.spec)


@dataclass(slots=True)
class RouteClass:
    known: bool | None
    template: str
    object_id: str | None
    spec: RouteSpec | None


PRIVILEGED_FIELDS = frozenset(
    {"role", "roles", "is_admin", "admin", "isadmin", "permissions", "scopes", "balance", "owner_id",
     "user_id", "verified", "is_verified", "is_staff", "is_superuser", "tenant_id", "account_type"}
)


def undeclared_body_fields(
    spec: RouteSpec | None, body: bytes, content_type: str, max_bytes: int = 65_536
) -> tuple[list[str], list[str]]:
    """Return (undeclared, privileged) top-level JSON body keys versus the declared request schema."""
    if spec is None or not spec.body_fields or not body or len(body) > max_bytes or "json" not in content_type:
        return [], []
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return [], []
    if not isinstance(data, dict):
        return [], []
    undeclared = sorted(str(k) for k in data if k not in spec.body_fields)[:20]
    return undeclared, [k for k in undeclared if k.lower() in PRIVILEGED_FIELDS]


# ---------------------------------------------------------------------- OpenAPI parsing


def load_schema_document(raw: bytes | str) -> dict[str, Any]:
    text = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise OpenAPIError(f"document is neither valid JSON nor YAML: {exc}") from exc
    if not isinstance(doc, dict):
        raise OpenAPIError("schema root must be an object")
    if "openapi" not in doc and "swagger" not in doc:
        raise OpenAPIError("missing 'openapi' or 'swagger' version field")
    if not isinstance(doc.get("paths"), dict) or not doc["paths"]:
        raise OpenAPIError("schema has no 'paths'")
    return doc


def _resolve(doc: dict[str, Any], node: Any, depth: int = 0) -> Any:
    """Resolve a local JSON pointer `$ref` (one chain, bounded depth)."""
    while isinstance(node, dict) and "$ref" in node and depth < 8:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            return {}
        target: Any = doc
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node = target
        depth += 1
    return node


def _base_paths(doc: dict[str, Any]) -> list[str]:
    if "swagger" in doc:
        base = str(doc.get("basePath") or "")
        return [base.rstrip("/")]
    bases: list[str] = []
    for server in doc.get("servers") or []:
        url = str(server.get("url", "")) if isinstance(server, dict) else ""
        path = urlparse(url).path if "://" in url else url
        path = _PARAM_RE.sub("", path).rstrip("/") if "{" in path else path.rstrip("/")
        if path not in bases:
            bases.append(path)
    return bases or [""]


def _param_spec(doc: dict[str, Any], raw: Any) -> ParamSpec | None:
    p = _resolve(doc, raw)
    if not isinstance(p, dict) or "name" not in p:
        return None
    schema = _resolve(doc, p.get("schema") or {})
    ptype = schema.get("type") if isinstance(schema, dict) else None
    enum = schema.get("enum") if isinstance(schema, dict) else None
    return ParamSpec(
        name=str(p["name"]),
        location=str(p.get("in", "query")),
        required=bool(p.get("required", False)),
        type=ptype or p.get("type"),
        enum=enum or p.get("enum"),
    )


def _body_fields(doc: dict[str, Any], op: dict[str, Any], params: list[Any]) -> list[str]:
    schema: Any = None
    body = _resolve(doc, op.get("requestBody") or {})
    if isinstance(body, dict) and body.get("content"):
        content = body["content"]
        media = content.get("application/json") or next(iter(content.values()), {})
        schema = media.get("schema") if isinstance(media, dict) else None
    else:
        for raw in params:
            p = _resolve(doc, raw)
            if isinstance(p, dict) and p.get("in") == "body":
                schema = p.get("schema")
    schema = _resolve(doc, schema or {})
    fields: list[str] = []
    if isinstance(schema, dict):
        for part in [schema, *[_resolve(doc, s) for s in schema.get("allOf", [])]]:
            if isinstance(part, dict):
                fields.extend(str(k) for k in (part.get("properties") or {}))
    return sorted(set(fields))


def parse_routes(doc: dict[str, Any], source: str) -> list[RouteSpec]:
    global_security = bool(doc.get("security"))
    routes: list[RouteSpec] = []
    for base in _base_paths(doc):
        for raw_path, item in doc["paths"].items():
            item = _resolve(doc, item)
            if not isinstance(item, dict):
                continue
            shared_params = item.get("parameters") or []
            for method in HTTP_METHODS:
                op = item.get(method)
                if not isinstance(op, dict):
                    continue
                all_params = [*shared_params, *(op.get("parameters") or [])]
                specs = [s for s in (_param_spec(doc, p) for p in all_params) if s and s.location != "body"]
                template = f"{base}{raw_path}" if raw_path.startswith("/") else f"{base}/{raw_path}"
                regex, names = _compile_template(template)
                secured = bool(op["security"]) if "security" in op else global_security
                routes.append(
                    RouteSpec(
                        method=method.upper(),
                        template=template,
                        source=source,
                        regex=regex,
                        path_param_names=names,
                        params=specs,
                        body_fields=_body_fields(doc, op, all_params),
                        secured=secured,
                        summary=str(op.get("summary") or op.get("description") or "")[:300],
                        operation_id=str(op.get("operationId") or ""),
                    )
                )
    return routes


def route_document(r: RouteSpec) -> str:
    params = ", ".join(
        f"{p.name} ({p.location}, {p.type or 'any'}{', required' if p.required else ''})" for p in r.params
    )
    return "\n".join(
        [
            f"Endpoint: {r.method} {r.template}",
            f"Operation: {r.operation_id or 'n/a'} - {r.summary or 'no summary'}",
            f"Parameters: {params or 'none'}",
            f"Request body fields: {', '.join(r.body_fields) or 'none'}",
            f"Authentication required: {'yes' if r.secured else 'no'}",
            f"Review classes: {', '.join(sorted(review_classes(r))) or 'none'}",
        ]
    )


def review_classes(spec: RouteSpec | None) -> frozenset[str]:
    """Classify a declared operation once, from its shape, not from any one product's names.

    object_access: the path identifies one object.
    free_text: the body carries a field that is not only a credential or an id.
    url_or_file: a parameter name looks like a URL or a file.
    """
    if spec is None:
        return frozenset()
    classes: set[str] = set()
    if spec.path_param_names:
        classes.add("object_access")
    if any(not _is_credential(name) and not _is_identifier(name) for name in spec.body_fields):
        classes.add("free_text")
    names = [*spec.body_fields, *(param.name for param in spec.params)]
    if any(_is_url_or_file(name) for name in names):
        classes.add("url_or_file")
    return frozenset(classes)


def needs_agent_review(spec: RouteSpec | None, distinct_objects: int) -> bool:
    """True when the operation should be shown to the model even if the fast score is low.

    A single object id is ordinary. The model is asked from the second distinct id on that route.
    Free text and URL or file fields are reviewed on every request.
    """
    classes = review_classes(spec)
    if "free_text" in classes or "url_or_file" in classes:
        return True
    return "object_access" in classes and distinct_objects >= 2


def _is_credential(name: str) -> bool:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return key in {"username", "user", "password", "passwd", "email", "passcode", "token", "secret"}


def _is_identifier(name: str) -> bool:
    key = name.lower().replace("-", "").replace("_", "")
    return key == "id" or key.endswith("id")


def _is_url_or_file(name: str) -> bool:
    key = name.lower()
    return any(part in key for part in ("url", "uri", "href", "file", "path", "avatar", "image", "attachment", "document"))
    params = ", ".join(
        f"{p.name} ({p.location}, {p.type or 'any'}{', required' if p.required else ''})" for p in r.params
    )
    return "\n".join(
        [
            f"Endpoint: {r.method} {r.template}",
            f"Operation: {r.operation_id or 'n/a'} - {r.summary or 'no summary'}",
            f"Parameters: {params or 'none'}",
            f"Request body fields: {', '.join(r.body_fields) or 'none'}",
            f"Authentication required: {'yes' if r.secured else 'no'}",
        ]
    )


# ---------------------------------------------------------------------- schema validation

_TYPE_CHECKS: dict[str, re.Pattern[str]] = {
    "integer": re.compile(r"^-?\d+$"),
    "number": re.compile(r"^-?\d+(?:\.\d+)?$"),
    "boolean": re.compile(r"^(?:true|false|0|1)$", re.I),
}


def validate_against_spec(match: RouteMatch | None, query: str, body_preview: str, schema_loaded: bool) -> list[str]:
    """Deterministic findings comparing a request with its declared operation."""
    if not schema_loaded:
        return ["no OpenAPI schema loaded; schema validation skipped"]
    if match is None:
        return ["path is not declared in any loaded OpenAPI schema (possible shadow/undocumented endpoint)"]
    if not match.method_allowed or match.spec is None:
        return [f"method not declared for {match.template}; allowed: {', '.join(sorted(set(match.allowed_methods)))}"]

    spec = match.spec
    findings: list[str] = [f"matches declared operation {spec.method} {spec.template}"]
    by_name = {(p.location, p.name): p for p in spec.params}

    for name, value in match.path_params.items():
        p = by_name.get(("path", name))
        check = _TYPE_CHECKS.get(p.type or "") if p else None
        if check and not check.match(value):
            findings.append(f"path parameter '{name}' expects {p.type} but received {value[:60]!r}")

    declared_query = {p.name for p in spec.params if p.location == "query"}
    received = [k for k, _ in parse_qsl(query, keep_blank_values=True)]
    unexpected = sorted({k for k in received if k not in declared_query})
    if unexpected:
        findings.append(f"undeclared query parameters: {', '.join(unexpected[:10])}")
    for k, v in parse_qsl(query, keep_blank_values=True):
        p = by_name.get(("query", k))
        if p and p.enum and v not in [str(e) for e in p.enum]:
            findings.append(f"query parameter '{k}' value {v[:40]!r} is outside the declared enum")

    if spec.body_fields and body_preview.lstrip().startswith("{"):
        try:
            body = json.loads(body_preview)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict):
            extra = sorted(set(body) - set(spec.body_fields))
            if extra:
                findings.append(f"undeclared body fields (possible mass assignment): {', '.join(extra[:10])}")
    if spec.secured:
        findings.append("operation requires authentication; object-level authorization applies")
    return findings


# ---------------------------------------------------------------------- threat knowledge base

THREAT_PATTERNS: list[dict[str, str]] = [
    {
        "id": "threat:path-traversal-dot-segments",
        "threat_type": "path_traversal",
        "title": "Path traversal via dot-dot segments (CWE-22, OWASP A01:2021)",
        "body": "Technique: parent-directory segments move the resolved location out of the directory the application "
        "is allowed to read. One form starts in a legitimate folder such as uploads, images, or documents and then "
        "climbs with ../ until it reaches operating-system files, secrets, configuration, or another user's data. "
        "Another form is the climb alone, or a sensitive location with no friendly prefix. Missing a legitimate "
        "start does not make the value safe. One relative step that stays inside public application content, and "
        "does not continue upward, is not this technique. Path, query, and free text are all sinks. Response: block.",
    },
    {
        "id": "threat:path-traversal-encoded",
        "threat_type": "path_traversal",
        "title": "Encoded and double-encoded traversal (CWE-22, CWE-174)",
        "body": "Technique: the same parent-directory climb is hidden so a literal filter misses it. Layers include "
        "percent-encoding, encoding applied twice, overlong UTF-8, mixed slash directions, and a truncated name that "
        "hides the real extension. Decoding the value back to a directory climb is enough. The encoding itself is "
        "intent to evade, including when it appears inside free text. Response: block; high confidence.",
    },
    {
        "id": "threat:sensitive-file-targets",
        "threat_type": "path_traversal",
        "title": "Sensitive file targets",
        "body": "Technique: after leaving the application content root, the path names a location that holds "
        "credentials, account databases, process environment, boot configuration, or private keys. Also a shell "
        "command whose argument is such a location. Clients of a store API do not need those locations. A public "
        "asset that remains inside the application tree is not this class. Response: block and blocklist the source.",
    },
    {
        "id": "threat:path-traversal-free-text",
        "threat_type": "path_traversal",
        "title": "Path traversal inside free-text fields (CWE-22)",
        "body": "Free-text fields are sinks. A comment, search query, ticket, review title, or note is an attack when "
        "it climbs out of an allowed directory or names a sensitive operating-system, secret, or configuration file. "
        "The climb can start in a legitimate folder or have no such prefix. A sentence that only points at one public "
        "asset that stays inside application content, and does not keep climbing, is documentation. Response: block "
        "when the resolved location leaves that content.",
    },
    {
        "id": "threat:obfuscated-payloads",
        "threat_type": "injection",
        "title": "Obfuscated payloads: encoding layers and XOR (CWE-116)",
        "body": "Technique: the attacker hides a traversal, SQL statement, or command behind a reversible transform "
        "so the raw characters do not match a signature. Common transforms are percent-encoding stacked more than "
        "once, base64, unicode escapes, and XOR or a single-byte key applied to the payload, sometimes with the key "
        "stated nearby. Recover the underlying string. If the decoded form is a directory climb, executable SQL, or "
        "a shell command, treat it as that attack even though the wire value looks harmless. A normal word that "
        "happens to contain punctuation is not an encoded payload. Response: block when a decode yields an attack.",
    },
    {
        "id": "threat:null-byte-wrappers",
        "threat_type": "path_traversal",
        "title": "Null byte truncation and stream wrappers (CWE-158)",
        "body": "Two techniques that do not need a parent-directory climb. Truncation: a file name is cut before a "
        "check sees its real suffix, classically with a null character, so the application opens a different file "
        "than the one it validated. Command read: a shell invocation whose target is a credential, account, or "
        "environment file. Naming that file inside an ordinary sentence, with no command and no climb, is discussion. "
        "Performing the truncation or the command is an attack in any field. Response: block the client.",
    },
    {
        "id": "threat:bola",
        "threat_type": "bola",
        "title": "Broken Object Level Authorization (OWASP API1:2023)",
        "body": "Indicators: one client requesting many distinct object IDs on the same route template in a short window "
        "(sequential or random enumeration), IDs that do not belong to the caller, swapping IDs in path or body. "
        "Schema context: routes with path parameters such as /users/{userId} or /orders/{orderId} that require auth. "
        "Response: the first object id a client uses on a route is normal. From the second distinct id, "
        "judge the hop. An id whose text looks secret is not evidence. Enumeration across many ids justifies "
        "blocklisting; one extra id can be ordinary work and should be allowed when the history looks like use, "
        "or alerted when it is unclear.",
    },
    {
        "id": "threat:bopla-mass-assignment",
        "threat_type": "mass_assignment",
        "title": "Broken Object Property Level Authorization / mass assignment (OWASP API3:2023)",
        "body": "Indicators: request bodies containing fields not declared in the OpenAPI schema, especially role, "
        "is_admin, permissions, balance, owner_id, verified. Response: alert; block when privileged fields are present.",
    },
    {
        "id": "threat:bfla-undeclared",
        "threat_type": "api_intrusion",
        "title": "Broken Function Level Authorization and undeclared operations (OWASP API5:2023, API9:2023)",
        "body": "An undeclared write or delete on a route that only offers a read is a probe on the first request, "
        "not one harmless miss. A path is also a probe on the first request when its segments name a control plane "
        "rather than a product feature: administration, internal tooling, debugging, private export, staff operations, "
        "dumps, or impersonation. A health path is the declared health operation, or an undeclared path that is "
        "exactly /health, /ready, /live, or /status. Documentation, help, and legal text are missing features. "
        "An undeclared path that adds a prefix, such as /system/status, is a probe. Response: block a "
        "control-plane path or an undeclared method. Alert only for one ordinary missing route.",
    },
    {
        "id": "threat:resource-consumption",
        "threat_type": "rate_abuse",
        "title": "Unrestricted resource consumption (OWASP API4:2023)",
        "body": "Indicators: request rate far above normal for one client, very large payloads, huge page sizes or limits. "
        "Response: rate limit; blocklist persistent abusers.",
    },
    {
        "id": "threat:injection",
        "threat_type": "injection",
        "title": "Injection: SQL, NoSQL, command, template (OWASP A03:2021)",
        "body": "Indicators: quote-balanced boolean tautologies, UNION SELECT, time-based functions, $where/$ne operators, "
        "shell metacharacters followed by commands, ${jndi: lookups, script tags. SQL syntax is specified separately "
        "and is malicious in any field. Response: block when the value contains the payload syntax.",
    },
    {
        "id": "threat:sql-injection",
        "threat_type": "injection",
        "title": "SQL injection in any field (OWASP A03:2021, CWE-89)",
        "body": "Technique: untrusted text changes the structure of a query instead of staying one value. The shape is "
        "any of: a string delimiter followed by a condition that is always true, a second statement after the first, "
        "a combined result set, or a comment marker that discards the rest of the original query. Free-text fields "
        "count, because the server may concatenate them into a statement. A sentence that only names the technique "
        "and does not form that shape is discussion. Response: block when the value itself has the shape.",
    },
    {
        "id": "threat:xss",
        "threat_type": "xss",
        "title": "Cross-site scripting in any field (CWE-79)",
        "body": "Technique: a value is browser code rather than data. The shape is a script element, an event handler "
        "attribute, a javascript: URL, or a tag such as svg, img, iframe, or body that carries one. Any field is a "
        "sink, including free text the server may reflect or store. A sentence that names the tag and does not form "
        "that shape is discussion. Response: block when the value itself is the code.",
    },
    {
        "id": "threat:command-execution",
        "threat_type": "injection",
        "title": "Command execution beyond a credential-file read (CWE-78)",
        "body": "Technique: metacharacters start a second command, or a template expression evaluates code. A shell "
        "command whose target is a credential file is one case. So is any command after a semicolon, pipe, or "
        "logical operator, and a template that multiplies, calls an object, or reads a file. Naming a command in a "
        "sentence, with no metacharacter and no template, is discussion. Response: block when the value executes.",
    },
    {
        "id": "threat:log4j",
        "threat_type": "injection",
        "title": "Log4j lookup injection (CWE-117, CVE-2021-44228)",
        "body": "Technique: a lookup of the form ${jndi:...} asks the logging library to fetch and load a remote "
        "class. Wrappers such as ${lower:j}ndi and percent-encoding are the same technique once unfolded. A sentence "
        "that names the lookup without the ${} shape is discussion. Response: block when the value has the shape.",
    },
    {
        "id": "threat:log-forging",
        "threat_type": "log_forging",
        "title": "Log forging via CRLF (CWE-117)",
        "body": "Technique: a carriage return or line feed, raw or percent-encoded, splits a log line, and the next "
        "line is a log stamp: a level word, a bracketed level such as [INFO] or [AUDIT], or a timestamp followed by "
        "a level. The stamp counts in any field, including free text. The sink is a "
        "header, a query parameter, an identity field, or free text. A newline that only continues a value, with no "
        "log stamp on the next line, is not this technique. Response: block "
        "when the next line is a log stamp.",
    },
    {
        "id": "threat:cookie-payload",
        "threat_type": "deserialization",
        "title": "Cookie that carries a payload (CWE-565, CWE-502)",
        "body": "Technique: the cookie is not a session identifier. It is a serialized object, SQL, a script, a "
        "directory climb, or a command. A normal opaque session token is not this technique. Whether a well-formed "
        "token was issued to a different user cannot be decided from one request. Response: block when the cookie "
        "or the login field is itself a payload.",
    },
    {
        "id": "threat:deserialization",
        "threat_type": "deserialization",
        "title": "Insecure deserialization (CWE-502)",
        "body": "Technique: a value is a serialized object graph the server may load. Python pickle often begins "
        "with a base64 prefix of the protocol header. Java serialization begins with the aced0005 magic or its "
        "base64 form. The sink can be a cookie, a body field, or a query value. Response: block when that shape is present.",
    },
    {
        "id": "threat:ssrf",
        "threat_type": "ssrf",
        "title": "Server-side request forgery (CWE-918, OWASP API7:2023)",
        "body": "Technique: an http or https URL points at the cloud metadata address, loopback, a private "
        "address, or a file URL. The URL can sit in any field, including one that is not declared as a URL. In a "
        "field whose job is a URL or file, the visible text of a markdown link counts too. In a note or "
        "description, only the link target counts. A private address that is only the visible text of a link to a "
        "public host, inside a note or description, is not this technique. The host of the incoming request itself "
        "is not this technique. A sentence that mentions an address without a URL is discussion. Response: block "
        "when the value that would be requested is one of those destinations.",
    },
    {
        "id": "threat:ssti",
        "threat_type": "injection",
        "title": "Server-side template injection (CWE-94)",
        "body": "Technique: a template expression evaluates on the server. The shape is a matched pair of double "
        "braces, a ${...} expression that computes or calls, or a #{...} expression. One pair of ordinary braces "
        "in prose is not this technique. Response: block when the value contains the expression.",
    },
    {
        "id": "threat:benign-noise",
        "threat_type": "benign",
        "title": "Common benign anomalies",
        "body": "Punctuation, apostrophes, and a single relative link that stays inside public application content "
        "are normal. So is prose that names an attack technique without performing it. These are not attacks. They "
        "become attacks when the value climbs out of a legitimate directory, decodes into such a climb, or forms a "
        "query or command of its own. Judge that shape. Do not treat a technique label in a sentence as the technique.",
    },
]


def _threat_doc(t: dict[str, str]) -> str:
    return f"{t['title']}\nThreat type: {t['threat_type']}\n{t['body']}"


# ---------------------------------------------------------------------- engine


@dataclass(slots=True)
class RagContext:
    schema_loaded: bool
    route_template: str | None
    schema_findings: list[str]
    endpoint_docs: list[str]
    threat_docs: list[str]

    def as_prompt_section(self) -> str:
        lines = ["## OpenAPI schema validation"]
        lines += [f"- {f}" for f in self.schema_findings]
        lines.append("\n## Declared endpoint specifications (retrieved)")
        lines += [f"---\n{d}" for d in self.endpoint_docs] or ["- none"]
        lines.append("\n## Relevant threat patterns (retrieved)")
        lines += [f"---\n{d}" for d in self.threat_docs] or ["- none"]
        return "\n".join(lines)


def _sanitize_source(source: str) -> str:
    """Restrict source names to a safe file-name alphabet (they become `<source>.json` on disk)."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", source)[:64].strip("_")
    return cleaned or "default"


@dataclass(slots=True)
class IngestResult:
    source: str
    title: str
    version: str
    routes: int
    operations: list[str]


class RAGEngine:
    def __init__(self, store: VectorStore, schema_dir: Path | None = None) -> None:
        self.store = store
        self.routes = RouteIndex()
        self._schema_dir = schema_dir

    @classmethod
    def create(cls, *, chroma_dir: str, enable_chroma: bool = True) -> "RAGEngine":
        embedder = HashingEmbedder()
        store: VectorStore
        if enable_chroma:
            try:
                Path(chroma_dir).mkdir(parents=True, exist_ok=True)
                store = ChromaVectorStore(embedder, chroma_dir)
            except Exception as exc:  # noqa: BLE001 - any Chroma failure falls back to memory
                logger.warning("Chroma unavailable (%s); using in-memory vector store", exc)
                store = InMemoryVectorStore(embedder)
        else:
            store = InMemoryVectorStore(embedder)
        schema_dir = Path(chroma_dir).parent / "schemas"
        return cls(store, schema_dir)

    @property
    def backend(self) -> str:
        return type(self.store).__name__

    def bootstrap(self) -> None:
        """Seed threat patterns and reload any previously ingested schemas."""
        self.store.upsert(
            [t["id"] for t in THREAT_PATTERNS],
            [_threat_doc(t) for t in THREAT_PATTERNS],
            [{"kind": "threat", "threat_type": t["threat_type"]} for t in THREAT_PATTERNS],
        )
        if self._schema_dir and self._schema_dir.is_dir():
            for path in sorted(self._schema_dir.glob("*.json")):
                try:
                    self.ingest_openapi(path.read_bytes(), source=path.stem, persist=False)
                except OpenAPIError as exc:
                    logger.warning("skipping stored schema %s: %s", path.name, exc)

    def ingest_openapi(self, raw: bytes | str, *, source: str = "default", persist: bool = True) -> IngestResult:
        source = _sanitize_source(source)
        doc = load_schema_document(raw)
        routes = parse_routes(doc, source)
        if not routes:
            raise OpenAPIError("schema declares no operations")

        self.store.delete_where({"kind": "endpoint", "source": source})
        self.store.upsert(
            [f"endpoint:{source}:{r.method}:{r.template}" for r in routes],
            [route_document(r) for r in routes],
            [{"kind": "endpoint", "source": source, "method": r.method, "template": r.template} for r in routes],
        )
        self.routes.replace_source(source, routes)

        if persist and self._schema_dir:
            self._schema_dir.mkdir(parents=True, exist_ok=True)
            (self._schema_dir / f"{source}.json").write_text(json.dumps(doc), encoding="utf-8")

        info = doc.get("info") or {}
        logger.info("ingested OpenAPI source=%s operations=%d", source, len(routes))
        return IngestResult(
            source=source,
            title=str(info.get("title", "")),
            version=str(info.get("version", "")),
            routes=len(routes),
            operations=[f"{r.method} {r.template}" for r in routes][:200],
        )

    def remove_source(self, source: str) -> bool:
        source = _sanitize_source(source)
        removed = self.routes.remove_source(source)
        self.store.delete_where({"kind": "endpoint", "source": source})
        if self._schema_dir:
            path = self._schema_dir / f"{source}.json"
            if path.resolve().parent == self._schema_dir.resolve() and path.is_file():
                path.unlink()
                removed = True
        return removed

    def search_threats(self, query: str, k: int = 3) -> list[str]:
        """Vector search over seeded threat documents. The query is truncated and not executed."""
        text = " ".join(query.split())[:200]
        if not text:
            return []
        return [hit.document for hit in self.store.query(text, k, {"kind": "threat"})]

    def describe_endpoint(self, method: str, path: str) -> list[str]:
        """Return the declared operation for method+path, plus at most one similar operation."""
        docs: list[str] = []
        if self.routes.loaded:
            match = self.routes.match(method, path.split("?", 1)[0])
            if match and match.spec:
                docs.append(route_document(match.spec))
            elif match:
                allowed = ", ".join(sorted(set(match.allowed_methods))) or "none"
                docs.append(f"Path {match.template} is declared, but {method.upper()} is not. Allowed methods: {allowed}.")
            else:
                docs.append(f"No declared operation for {method.upper()} {path}.")
        hits = self.store.query(f"{method} {path}", 2, {"kind": "endpoint"}) if self.routes.loaded else []
        for hit in hits:
            if hit.document not in docs:
                docs.append(hit.document)
            if len(docs) >= 2:
                break
        return docs[:2]

    def retrieve_context(self, req: SuspiciousRequest, k_threats: int = 3, k_endpoints: int = 2, *, prefetch: bool = True) -> RagContext:
        """Blocking retrieval; call from a worker thread (`asyncio.to_thread`)."""
        match = self.routes.match(req.method, req.path) if self.routes.loaded else None
        findings = validate_against_spec(match, req.query, req.body_preview, self.routes.loaded)
        template = match.template if match else heuristic_template(req.path)[0]

        endpoint_docs: list[str] = []
        if match and match.spec:
            endpoint_docs.append(route_document(match.spec))
        if prefetch and self.routes.loaded and len(endpoint_docs) < k_endpoints:
            hits = self.store.query(f"{req.method} {template}", k_endpoints, {"kind": "endpoint"})
            endpoint_docs += [h.document for h in hits if h.document not in endpoint_docs]
        endpoint_docs = endpoint_docs[:k_endpoints]

        threat_docs: list[str] = []
        if prefetch:
            query_text = " ".join(
                [
                    req.fast_verdict.threat_hint or "",
                    " ".join(req.fast_verdict.reasons),
                    " ".join(findings),
                    req.path,
                    req.query[:300],
                    req.body_preview[:300],
                ]
            )
            threat_docs = [h.document for h in self.store.query(query_text, k_threats, {"kind": "threat"})]
        return RagContext(
            schema_loaded=self.routes.loaded,
            route_template=template,
            schema_findings=findings,
            endpoint_docs=endpoint_docs,
            threat_docs=threat_docs,
        )
