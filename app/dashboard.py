"""Admin dashboard and management API (threat log, blocklist, OpenAPI ingestion)."""

from __future__ import annotations

import asyncio
import ipaddress
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from app.rag_engine import OpenAPIError

ADMIN_COOKIE = "sentinel_admin"
MAX_SCHEMA_BYTES = 5 * 1024 * 1024

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
templates.env.filters["ts"] = lambda v: datetime.fromtimestamp(float(v), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def require_admin(request: Request) -> None:
    """Accept the admin token from the X-Admin-Token header, the session cookie, or ?token=."""
    expected: str = request.app.state.settings.admin_token
    supplied = (
        request.headers.get("x-admin-token")
        or request.cookies.get(ADMIN_COOKIE)
        or request.query_params.get("token")
        or ""
    )
    if not expected or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="admin token required")


router = APIRouter(prefix="/sentinel/admin", tags=["sentinel-admin"], dependencies=[Depends(require_admin)])


class BlockRequest(BaseModel):
    ip: str
    reason: str = "manual block"
    ttl_seconds: int | None = Field(None, gt=0)


def _validate_ip(ip: str) -> str:
    try:
        return str(ipaddress.ip_address(ip))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid IP address") from exc


async def _safe(coro: Any, default: Any) -> Any:
    try:
        return await coro
    except RedisError:
        return default


async def _redis(coro: Any) -> Any:
    try:
        return await coro
    except RedisError as exc:
        raise HTTPException(status_code=503, detail="security state store unavailable") from exc


@router.get("", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page(request: Request, limit: int = Query(100, ge=1, le=500)) -> Response:
    if "token" in request.query_params:
        # Exchange the URL token for an HttpOnly cookie, then drop it from the address bar.
        resp = RedirectResponse("/sentinel/admin", status_code=303)
        resp.set_cookie(
            ADMIN_COOKIE,
            request.query_params["token"],
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/sentinel",
        )
        return resp

    state = request.app.state
    events = await _safe(state.store.recent_events(limit), [])
    blocked = await _safe(state.store.list_blocked(), [])
    stats = await _safe(state.store.stats(), {})
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "events": events,
            "blocked": blocked,
            "stats": stats,
            "llm_enabled": state.worker.llm_enabled,
            "model": state.settings.baseten_model,
            "vector_store": state.rag.backend,
            "schema_sources": state.rag.routes.sources,
            "queue_depth": state.worker.queue_depth,
            "flash": request.query_params.get("msg"),
        },
    )


@router.get("/events")
async def list_events(request: Request, limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
    events = await _redis(request.app.state.store.recent_events(limit))
    return [e.model_dump() for e in events]


@router.get("/stats")
async def get_stats(request: Request) -> dict[str, Any]:
    state = request.app.state
    return {
        "counters": await _redis(state.store.stats()),
        "agent": {
            "llm_enabled": state.worker.llm_enabled,
            "queue_depth": state.worker.queue_depth,
            "processed": state.worker.processed,
            "dropped": state.worker.dropped,
        },
    }


@router.get("/blocklist")
async def get_blocklist(request: Request) -> list[dict[str, Any]]:
    return await _redis(request.app.state.store.list_blocked())


@router.post("/blocklist", status_code=201)
async def add_block(request: Request, body: BlockRequest) -> dict[str, Any]:
    state = request.app.state
    ip = _validate_ip(body.ip)
    ttl = body.ttl_seconds or state.settings.block_ttl_seconds
    await _redis(state.store.block_ip(ip, reason=body.reason, threat_type="manual", confidence=1.0, ttl=ttl))
    return {"blocked": ip, "ttl_seconds": ttl}


@router.delete("/blocklist/{ip}")
async def remove_block(request: Request, ip: str) -> dict[str, Any]:
    removed = await _redis(request.app.state.store.unblock_ip(_validate_ip(ip)))
    if not removed:
        raise HTTPException(status_code=404, detail="IP not in blocklist")
    return {"unblocked": ip}


@router.post("/blocklist/{ip}/delete", include_in_schema=False)
async def remove_block_form(request: Request, ip: str) -> Response:
    """HTML-form variant of DELETE used by the dashboard."""
    ip = _validate_ip(ip)
    await _redis(request.app.state.store.unblock_ip(ip))
    return _redirect_with_message(f"Unblocked {ip}")


@router.get("/openapi")
async def list_schemas(request: Request) -> dict[str, Any]:
    return {"sources": request.app.state.rag.routes.sources}


@router.post("/openapi")
async def ingest_openapi(request: Request, source: str = Query("default", max_length=64)) -> Response:
    """Ingest an OpenAPI 3 / Swagger 2 document (JSON or YAML).

    Accepts a raw request body (application/json, application/yaml, text/plain) or a
    multipart form with a `file` field. Re-uploading the same `source` replaces it.
    """
    content_type = request.headers.get("content-type", "")
    from_form = content_type.startswith("multipart/form-data")
    if from_form:
        form = await request.form()
        upload = form.get("file")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=422, detail="multipart upload requires a 'file' field")
        raw = await upload.read(MAX_SCHEMA_BYTES + 1)
        source = str(form.get("source") or source)
    else:
        raw = await request.body()
    if len(raw) > MAX_SCHEMA_BYTES:
        raise HTTPException(status_code=413, detail="schema too large")

    rag = request.app.state.rag
    try:
        result = await asyncio.to_thread(rag.ingest_openapi, raw, source=source)
    except OpenAPIError as exc:
        if from_form:
            return _redirect_with_message(f"Schema rejected: {str(exc)[:120]}")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if from_form:
        return _redirect_with_message(f"Ingested {result.routes} operations from '{result.source}'")
    return JSONResponse(
        {
            "source": result.source,
            "title": result.title,
            "version": result.version,
            "operations": result.routes,
            "routes": result.operations,
        }
    )


@router.delete("/openapi/{source}")
async def delete_schema(request: Request, source: str) -> dict[str, Any]:
    rag = request.app.state.rag
    removed = await asyncio.to_thread(rag.remove_source, source)
    if not removed:
        raise HTTPException(status_code=404, detail="unknown schema source")
    return {"removed": source}


def _redirect_with_message(message: str) -> RedirectResponse:
    return RedirectResponse(f"/sentinel/admin?msg={quote_plus(message)}", status_code=303)
