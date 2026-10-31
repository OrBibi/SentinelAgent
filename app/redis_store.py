"""Redis-backed state: sliding windows, session history, blocklist, and security events.

The hot path issues a single pipelined round trip per request (`check_request`).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from redis.asyncio import Redis

from app.models import SecurityEvent

logger = logging.getLogger("sentinel.redis")
PREFIX = "sentinel"


def _k(*parts: str) -> str:
    return ":".join((PREFIX, *parts))


def _load_object(raw: str | None) -> dict[str, Any] | None:
    """Parse a Redis JSON object, treating corrupt values as absent instead of crashing the proxy."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("ignoring corrupt redis json")
        return None
    return data if isinstance(data, dict) else None


@dataclass(slots=True)
class RequestState:
    """Per-request state read from Redis on the hot path."""

    block_info: dict[str, Any] | None
    window_rate: int
    distinct_objects: int
    watched: bool = False


class RedisStore:
    def __init__(
        self,
        client: Redis,
        *,
        rate_window_seconds: int,
        bola_window_seconds: int,
        history_size: int,
        max_events: int,
    ) -> None:
        self._r = client
        self._rate_window = rate_window_seconds
        self._bola_window = bola_window_seconds
        self._history_size = history_size
        self._max_events = max_events

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> "RedisStore":
        if url.startswith("memory://"):
            # Single-process in-memory Redis for demos and tests (requires `fakeredis`).
            try:
                from fakeredis import FakeAsyncRedis
            except ImportError as exc:
                raise RuntimeError("REDIS_URL=memory:// requires the fakeredis package") from exc
            return cls(FakeAsyncRedis(decode_responses=True), **kwargs)
        client = Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            health_check_interval=30,
        )
        return cls(client, **kwargs)

    async def ping(self) -> bool:
        return bool(await self._r.ping())

    async def close(self) -> None:
        await self._r.aclose()

    # ------------------------------------------------------------------ hot path

    async def check_request(
        self,
        ip: str,
        *,
        history_entry: str,
        route_template: str | None = None,
        object_id: str | None = None,
        principal: str | None = None,
        now: float | None = None,
    ) -> RequestState:
        """Read the blocklist, bump the sliding windows, and append session history in one round trip."""
        now = time.time() if now is None else now
        rate_key = _k("rate", ip)
        hist_id = principal or ip
        hist_key = _k("hist", hist_id)

        pipe = self._r.pipeline(transaction=False)
        pipe.get(_k("block", ip))
        ip_block_index = 0
        principal_block_index: int | None = None
        watch_index: int | None = None
        if principal:
            pipe.get(_k("blockid", principal))
            principal_block_index = 1
            pipe.get(_k("watch", principal))
            watch_index = 2
        pipe.zremrangebyscore(rate_key, 0, now - self._rate_window)
        pipe.zadd(rate_key, {f"{now:.6f}:{uuid.uuid4().hex[:8]}": now})
        rate_index = len(pipe.command_stack)
        pipe.zcard(rate_key)
        pipe.expire(rate_key, self._rate_window + 1)
        pipe.lpush(hist_key, history_entry)
        pipe.ltrim(hist_key, 0, self._history_size - 1)
        pipe.expire(hist_key, 3600)

        distinct_index: int | None = None
        if route_template and object_id:
            obj_key = _k("obj", principal or ip, route_template)
            pipe.zremrangebyscore(obj_key, 0, now - self._bola_window)
            pipe.zadd(obj_key, {object_id: now})
            distinct_index = len(pipe.command_stack)
            pipe.zcard(obj_key)
            pipe.expire(obj_key, self._bola_window + 1)
        pipe.hincrby(_k("stats"), "requests", 1)

        results = await pipe.execute()
        block_info = _load_object(results[ip_block_index])
        if block_info is None and principal_block_index is not None:
            block_info = _load_object(results[principal_block_index])
        distinct = int(results[distinct_index]) if distinct_index is not None else 0
        watched = watch_index is not None and _load_object(results[watch_index]) is not None
        return RequestState(
            block_info=block_info,
            window_rate=int(results[rate_index]),
            distinct_objects=distinct,
            watched=watched,
        )

    # ------------------------------------------------------------------ blocklist

    async def block_ip(
        self,
        ip: str,
        *,
        reason: str,
        threat_type: str,
        confidence: float,
        ttl: int,
        principal: str | None = None,
    ) -> None:
        payload = {
            "ip": ip,
            "reason": reason[:500],
            "threat_type": threat_type,
            "confidence": round(confidence, 3),
            "blocked_at": time.time(),
            "ttl": ttl,
        }
        if principal:
            payload["principal"] = principal
        await self._r.set(_k("block", ip), json.dumps(payload), ex=ttl)
        if principal:
            await self._r.set(_k("blockid", principal), json.dumps(payload), ex=ttl)

    async def refresh_block(self, ip: str, ttl: int) -> bool:
        return bool(await self._r.expire(_k("block", ip), ttl))

    async def object_ids(self, ip: str, template: str, limit: int = 20, principal: str | None = None) -> list[str]:
        """Ids this client already touched on one route template. Does not record a new id."""
        members = await self._r.zrange(_k("obj", principal or ip, template), 0, max(limit, 1) - 1)
        return [member.decode("utf-8") if isinstance(member, bytes) else str(member) for member in members]

    async def watch_principal(self, principal: str, *, reason: str, ttl: int) -> None:
        """Mark this caller so the next write is held for a verdict."""
        if not principal:
            return
        payload = {"principal": principal, "reason": reason[:500], "watched_at": time.time(), "ttl": ttl}
        await self._r.set(_k("watch", principal), json.dumps(payload), ex=ttl)

    async def is_watched(self, principal: str | None) -> bool:
        if not principal:
            return False
        return _load_object(await self._r.get(_k("watch", principal))) is not None

    async def mark_campaign(self, actor: str, ttl: int) -> bool:
        """True when this actor has no campaign mark in the cooldown window."""
        if not actor or ttl <= 0:
            return True
        return bool(await self._r.set(_k("campaign", actor), "1", ex=ttl, nx=True))

    async def get_block(self, ip: str, principal: str | None = None) -> dict[str, Any] | None:
        """Read the blocklist entry without counting the request toward rate or history."""
        info = _load_object(await self._r.get(_k("block", ip)))
        if info is None and principal:
            info = _load_object(await self._r.get(_k("blockid", principal)))
        return info

    async def unblock_ip(self, ip: str) -> bool:
        return bool(await self._r.delete(_k("block", ip)))

    async def list_blocked(self, limit: int = 200) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        async for key in self._r.scan_iter(match=_k("block", "*"), count=200):
            raw = await self._r.get(key)
            info = _load_object(raw)
            if info:
                info["ttl_remaining"] = await self._r.ttl(key)
                entries.append(info)
            if len(entries) >= limit:
                break
        return sorted(entries, key=lambda e: e.get("blocked_at", 0), reverse=True)

    # ------------------------------------------------------------------ history and events

    async def get_history(self, ip: str, limit: int = 15, principal: str | None = None) -> list[str]:
        return list(await self._r.lrange(_k("hist", principal or ip), 0, limit - 1))

    async def record_event(self, event: SecurityEvent) -> None:
        pipe = self._r.pipeline(transaction=False)
        pipe.lpush(_k("events"), event.model_dump_json())
        pipe.ltrim(_k("events"), 0, self._max_events - 1)
        pipe.hincrby(_k("stats"), event.action.value, 1)
        await pipe.execute()

    async def update_pending_event(self, request_id: str, event: SecurityEvent) -> bool:
        """Replace the queued row for this request once the agent decides."""
        if not request_id:
            return False
        key = _k("events")
        raw = await self._r.lrange(key, 0, self._max_events - 1)
        for index, item in enumerate(raw):
            try:
                current = SecurityEvent.model_validate_json(item)
            except ValidationError:
                continue
            if current.request_id != request_id or not current.extra.get("pending_agent"):
                continue
            merged = current.model_copy(
                update={
                    "action": event.action,
                    "threat_type": event.threat_type,
                    "confidence_score": event.confidence_score,
                    "reasoning": event.reasoning,
                    "source": event.source,
                    "extra": {**current.extra, **event.extra, "pending_agent": False},
                }
            )
            pipe = self._r.pipeline(transaction=False)
            pipe.lset(key, index, merged.model_dump_json())
            pipe.hincrby(_k("stats"), event.action.value, 1)
            await pipe.execute()
            return True
        return False

    async def attach_response(self, request_id: str, status: int, size: int, preview: str) -> None:
        """Store the upstream status, size, and an optional body snippet on an existing row."""
        if not request_id:
            return
        key = _k("events")
        raw = await self._r.lrange(key, 0, self._max_events - 1)
        for index, item in enumerate(raw):
            try:
                current = SecurityEvent.model_validate_json(item)
            except ValidationError:
                continue
            if current.request_id != request_id:
                continue
            merged = current.model_copy(
                update={"response_status": status, "response_bytes": size, "response_preview": preview}
            )
            await self._r.lset(key, index, merged.model_dump_json())
            return

    async def recent_events(self, limit: int = 100) -> list[SecurityEvent]:
        raw = await self._r.lrange(_k("events"), 0, limit - 1)
        events: list[SecurityEvent] = []
        for item in raw:
            try:
                events.append(SecurityEvent.model_validate_json(item))
            except ValidationError:
                logger.warning("skipping corrupt security event")
        return events

    async def stats(self) -> dict[str, int]:
        raw = await self._r.hgetall(_k("stats"))
        return {k: int(v) for k, v in raw.items()}

    async def incr_stat(self, name: str, amount: int = 1) -> None:
        await self._r.hincrby(_k("stats"), name, amount)
