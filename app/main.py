"""SentinelAgent reverse proxy.

Hot path per request (no network calls except one pipelined Redis round trip):
  blocklist check -> sliding-window rate -> fast ML/signature filter -> forward or 403.
Flagged requests are handed to the async agent without waiting for it.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app import dashboard
from app.agent_worker import AgentWorker, MetaModelClient, redact_headers
from app.config import Settings, get_settings
from app.ml_filter import FastFilter
from app.models import Action, SecurityEvent, SuspiciousRequest, Tier
from app.rag_engine import RAGEngine, needs_agent_review, undeclared_body_fields
from app.redis_store import RedisStore, RequestState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("sentinel.proxy")

PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
SECURITY_HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
# Body snippet kept for blocked requests and requests sent to the model. Clean traffic stores status and size only.
RESPONSE_PREVIEW_BYTES = 800


class BodyTooLarge(Exception):
    pass


def _error(status: int, error: str, message: str, request_id: str, extra_headers: dict[str, str] | None = None) -> JSONResponse:
    """Uniform error payload that never reveals detection internals."""
    return JSONResponse(
        status_code=status,
        content={"error": error, "message": message, "request_id": request_id},
        headers={**SECURITY_HEADERS, "X-Sentinel-Request-Id": request_id, **(extra_headers or {})},
    )


def forbidden(request_id: str) -> JSONResponse:
    return _error(403, "forbidden", "Request blocked by security policy.", request_id)


def normalize_proxy_path(raw_path: str) -> str:
    """Collapse scheme-relative targets so inspection and forwarding use the same path.

    httpx reparses `//host/rest` as a protocol-relative URL and forwards only `/rest`,
    which would let a payload be checked against one path and delivered to another.
    """
    path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
    return "/" + path.lstrip("/")


def path_is_inspected(path: str, prefixes: tuple[str, ...]) -> bool:
    """True when this path should be judged. Empty prefixes judge every path."""
    if not prefixes:
        return True
    normalized = path if path.startswith("/") else f"/{path}"
    normalized = normalized.rstrip("/") or "/"
    return any(normalized == prefix or normalized.startswith(prefix + "/") for prefix in prefixes)


def same_upstream(url: httpx.URL, base: httpx.URL) -> bool:
    return (url.scheme, url.host, url.port) == (base.scheme, base.host, base.port)


def _header_bytes(value: str) -> bytes:
    return value.replace("\r", "").replace("\n", "").encode("latin-1", errors="replace")


def client_ip(request: Request, trust_forwarded_for: bool) -> str:
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge
        chunks.append(chunk)
    return b"".join(chunks)


_REWRITTEN_HEADERS = HOP_BY_HOP | {"host", "content-length", "x-forwarded-for", "x-forwarded-proto", "x-forwarded-host"}


def upstream_headers(request: Request, ip: str) -> list[tuple[bytes, bytes]]:
    headers = [(k, v) for k, v in request.headers.raw if k.decode("latin-1").lower() not in _REWRITTEN_HEADERS]
    prior = request.headers.get("x-forwarded-for", "")
    forwarded = f"{prior}, {ip}" if prior else ip
    headers.append((b"x-forwarded-for", _header_bytes(forwarded)))
    headers.append((b"x-forwarded-proto", _header_bytes(request.url.scheme)))
    if request.headers.get("host"):
        headers.append((b"x-forwarded-host", _header_bytes(request.headers["host"])))
    return headers


def create_app(
    settings: Settings | None = None,
    *,
    redis_client: Redis | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    llm_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store_kwargs = dict(
            rate_window_seconds=settings.rate_window_seconds,
            bola_window_seconds=settings.bola_window_seconds,
            history_size=settings.history_size,
            max_events=settings.max_events,
        )
        store = RedisStore(redis_client, **store_kwargs) if redis_client else RedisStore.from_url(settings.redis_url, **store_kwargs)
        try:
            await store.ping()
        except RedisError as exc:
            logger.warning("Redis unreachable at startup (%s); fail_mode=%s", exc, settings.fail_mode)

        fast = FastFilter(
            suspicious_threshold=settings.suspicious_threshold,
            hard_block_threshold=settings.hard_block_threshold,
            bola_distinct_threshold=settings.bola_distinct_threshold,
        )
        rag = await asyncio.to_thread(RAGEngine.create, chroma_dir=settings.chroma_dir, enable_chroma=settings.enable_chroma)
        await asyncio.to_thread(rag.bootstrap)

        llm: MetaModelClient | None = None
        if settings.llm_enabled:
            llm = MetaModelClient(
                api_key=settings.meta_api_key,
                base_url=settings.meta_api_url,
                model=settings.meta_model,
                timeout=settings.llm_timeout_seconds,
                max_tokens=settings.llm_max_tokens,
                max_retries=settings.llm_max_retries,
                transport=llm_transport,
            )
        else:
            logger.warning("META_API_KEY not set: fast filter only, no LLM-driven blocklisting")
        if settings.admin_token == "change-me":
            logger.warning("ADMIN_TOKEN is the default value; set a strong token before exposing the dashboard")
        if not settings.enforce_blocks:
            logger.warning("ENFORCE_BLOCKS=false; blocklist decisions are logged but not applied")

        worker: AgentWorker | None = None
        http: httpx.AsyncClient | None = None
        try:
            worker = AgentWorker(store=store, rag=rag, llm=llm, settings=settings)
            worker.start()
            http = httpx.AsyncClient(
                base_url=settings.upstream_base_url,
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
                transport=upstream_transport,
                follow_redirects=False,
            )
            app.state.settings = settings
            app.state.store = store
            app.state.fast_filter = fast
            app.state.rag = rag
            app.state.worker = worker
            app.state.http = http
            logger.info(
                "SentinelAgent ready: upstream=%s vector_store=%s llm=%s",
                settings.upstream_base_url,
                rag.backend,
                settings.meta_model if llm else "disabled",
            )
            yield
        finally:
            if worker is not None:
                await worker.stop()
            elif llm is not None:
                await llm.aclose()
            if http is not None:
                await http.aclose()
            await store.close()

    app = FastAPI(
        title="SentinelAgent",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/sentinel/docs",
        openapi_url="/sentinel/openapi.json",
        redoc_url=None,
    )
    app.include_router(dashboard.router)

    @app.get("/sentinel/health", tags=["sentinel"])
    async def health(request: Request) -> dict[str, object]:
        state = request.app.state
        try:
            redis_ok = await state.store.ping()
        except RedisError:
            redis_ok = False
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "llm_enabled": state.worker.llm_enabled,
            "model": state.settings.meta_model,
            "vector_store": state.rag.backend,
            "schema_sources": state.rag.routes.sources,
            "agent_queue_depth": state.worker.queue_depth,
        }

    @app.api_route("/{full_path:path}", methods=PROXY_METHODS, include_in_schema=False)
    async def proxy(request: Request, full_path: str) -> Response:
        started = time.perf_counter()
        state = request.app.state
        cfg: Settings = state.settings
        request_id = uuid.uuid4().hex
        ip = client_ip(request, cfg.trust_forwarded_for)
        method = request.method
        # Inspect and forward the raw (undecoded) path so encoded payloads stay visible.
        raw_path = normalize_proxy_path(request.scope.get("raw_path", b"").decode("latin-1") or request.url.path)
        query = request.scope.get("query_string", b"").decode("latin-1")
        decoded_path = unquote(raw_path)
        if decoded_path == "/sentinel" or decoded_path.startswith("/sentinel/"):
            return _error(404, "not_found", "Unknown SentinelAgent endpoint.", request_id)

        if not path_is_inspected(decoded_path, cfg.inspect_prefixes):
            return await _forward_uninspected(state, request, cfg, ip, method, raw_path, query, request_id, started)

        route = state.rag.routes.classify(method, decoded_path)
        template = route.template
        history_entry = f"{int(time.time())} {method} {raw_path[:200]}" + (f"?{query[:200]}" if query else "")

        try:
            rs = await state.store.check_request(
                ip, history_entry=history_entry, route_template=template, object_id=route.object_id
            )
        except RedisError as exc:
            if cfg.fail_mode == "closed":
                logger.error("Redis unavailable, failing closed: %s", exc)
                return _error(503, "unavailable", "Security service temporarily unavailable.", request_id)
            logger.warning("Redis unavailable, failing open: %s", exc)
            rs = RequestState(block_info=None, window_rate=0, distinct_objects=0)

        if rs.block_info and cfg.enforce_blocks:
            return forbidden(request_id)

        if rs.window_rate > cfg.rate_limit_per_window:
            if rs.window_rate == cfg.rate_limit_per_window + 1:
                # Record only the first excess request per window to avoid flooding the event log.
                await _record(
                    state.store, ip, method, raw_path, Tier.SUSPICIOUS, 0.0, Action.BLOCKED, "rate_abuse",
                    f"rate limit exceeded: >{cfg.rate_limit_per_window} requests / {cfg.rate_window_seconds}s",
                    request_id=request_id,
                    response_status=429,
                    response_preview="Request blocked by security policy.",
                )
            return _error(
                429, "too_many_requests", "Rate limit exceeded.", request_id,
                {"Retry-After": str(cfg.rate_window_seconds)},
            )

        try:
            body = await read_body(request, cfg.max_body_bytes)
        except BodyTooLarge:
            return _error(413, "payload_too_large", "Request body too large.", request_id)

        undeclared, privileged = undeclared_body_fields(route.spec, body, request.headers.get("content-type", ""))
        fast: FastFilter = state.fast_filter
        features = fast.extract_features(
            method,
            raw_path,
            query,
            body,
            window_rate=rs.window_rate,
            route_known=route.known,
            matched_route=template,
            undeclared_fields=undeclared,
            privileged_fields=privileged,
            headers=dict(request.headers),
        )
        verdict = fast.evaluate(features, distinct_objects=rs.distinct_objects)
        review_anyway = needs_agent_review(route.spec, rs.distinct_objects)

        queued = verdict.tier is not Tier.CLEAN or review_anyway
        if queued:
            state.worker.submit(
                SuspiciousRequest(
                    request_id=request_id,
                    client_ip=ip,
                    method=method,
                    path=raw_path,
                    query=query,
                    headers=redact_headers(dict(request.headers)),
                    body_preview=body[:4096].decode("utf-8", errors="replace"),
                    features=features,
                    fast_verdict=verdict,
                ),
                template,
            )

        guard_ms = (time.perf_counter() - started) * 1000
        target = raw_path + (f"?{query}" if query else "")
        if verdict.tier is Tier.HARD_BLOCK:
            await _record(
                state.store, ip, method, target, verdict.tier, verdict.score, Action.BLOCKED,
                verdict.threat_hint, "; ".join(verdict.reasons), request_id=request_id,
                response_status=403,
                response_preview="Request blocked by security policy.",
            )
            return forbidden(request_id)

        await _record(
            state.store, ip, method, target, verdict.tier, verdict.score, Action.FORWARDED,
            verdict.threat_hint, "; ".join(verdict.reasons), request_id=request_id,
            extra={"pending_agent": queued, "fast_reasons": verdict.reasons},
        )
        return await _forward(
            state.http, request, ip, method, raw_path, query, body, request_id, guard_ms, verdict.tier,
            store=state.store, keep_preview=queued,
        )

    return app


async def _forward_uninspected(
    state: Any,
    request: Request,
    cfg: Settings,
    ip: str,
    method: str,
    raw_path: str,
    query: str,
    request_id: str,
    started: float,
) -> Response:
    """Forward a page or static file. A blocklisted client is still denied."""
    if cfg.enforce_blocks:
        try:
            if await state.store.get_block(ip):
                return forbidden(request_id)
        except RedisError as exc:
            if cfg.fail_mode == "closed":
                logger.error("Redis unavailable, failing closed: %s", exc)
                return _error(503, "unavailable", "Security service temporarily unavailable.", request_id)
            logger.warning("Redis unavailable, failing open: %s", exc)
    try:
        body = await read_body(request, cfg.max_body_bytes)
    except BodyTooLarge:
        return _error(413, "payload_too_large", "Request body too large.", request_id)
    guard_ms = (time.perf_counter() - started) * 1000
    return await _forward(
        state.http, request, ip, method, raw_path, query, body, request_id, guard_ms, "passthrough"
    )


async def _record(
    store: RedisStore,
    ip: str,
    method: str,
    path: str,
    tier: Tier,
    score: float,
    action: Action,
    threat: str | None,
    reasoning: str,
    *,
    request_id: str = "",
    extra: dict | None = None,
    response_status: int | None = None,
    response_preview: str = "",
) -> None:
    try:
        await store.record_event(
            SecurityEvent(
                request_id=request_id,
                client_ip=ip,
                method=method,
                path=path,
                tier=tier,
                fast_score=score,
                action=action,
                threat_type=threat,
                reasoning=reasoning,
                source="fast_filter",
                response_status=response_status,
                response_preview=response_preview,
                extra=extra or {},
            )
        )
    except RedisError:
        logger.exception("failed to record event for %s", ip)


async def _forward(
    http: httpx.AsyncClient,
    request: Request,
    ip: str,
    method: str,
    raw_path: str,
    query: str,
    body: bytes,
    request_id: str,
    guard_ms: float,
    tier: Tier | str,
    *,
    store: RedisStore | None = None,
    keep_preview: bool = False,
) -> Response:
    target = raw_path + (f"?{query}" if query else "")
    try:
        upstream_req = http.build_request(method, target, headers=upstream_headers(request, ip), content=body or None)
    except httpx.InvalidURL:
        return _error(400, "bad_request", "Request target could not be forwarded.", request_id)
    if not same_upstream(upstream_req.url, http.base_url):
        logger.warning("refusing to forward %s off the configured upstream", upstream_req.url)
        return _error(400, "bad_request", "Request target could not be forwarded.", request_id)
    try:
        upstream = await http.send(upstream_req, stream=True)
    except httpx.TimeoutException:
        return _error(504, "gateway_timeout", "Upstream service timed out.", request_id)
    except httpx.HTTPError as exc:
        logger.error("upstream error for %s %s: %s", method, raw_path, exc)
        return _error(502, "bad_gateway", "Upstream service unavailable.", request_id)

    async def tracked():
        preview = bytearray()
        total = 0
        limit = RESPONSE_PREVIEW_BYTES if keep_preview else 0
        try:
            async for chunk in upstream.aiter_raw():
                total += len(chunk)
                if len(preview) < limit:
                    preview.extend(chunk[: limit - len(preview)])
                yield chunk
        finally:
            if store is not None:
                text = preview.decode("utf-8", "replace") if preview else ""
                try:
                    await store.attach_response(request_id, upstream.status_code, total, text)
                except RedisError:
                    logger.exception("failed to store response for %s", request_id)
            await upstream.aclose()

    response = StreamingResponse(tracked(), status_code=upstream.status_code)
    response.raw_headers = [
        (k, v) for k, v in upstream.headers.raw if k.lower().decode("latin-1") not in HOP_BY_HOP
    ] + [
        (b"x-sentinel-request-id", request_id.encode()),
        (b"x-sentinel-guard-ms", f"{guard_ms:.3f}".encode()),
        (b"x-sentinel-tier", (tier.value if isinstance(tier, Tier) else tier).encode()),
    ]
    return response


app = create_app()
