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
        now: float | None = None,
    ) -> RequestState:
        """Read the blocklist, bump the sliding windows, and append session history in one round trip."""
        now = time.time() if now is None else now
        rate_key = _k("rate", ip)
        hist_key = _k("hist", ip)

        pipe = self._r.pipeline(transaction=False)
        pipe.get(_k("block", ip))
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
            obj_key = _k("obj", ip, route_template)
            pipe.zremrangebyscore(obj_key, 0, now - self._bola_window)
            pipe.zadd(obj_key, {object_id: now})
            distinct_index = len(pipe.command_stack)
            pipe.zcard(obj_key)
            pipe.expire(obj_key, self._bola_window + 1)
        pipe.hincrby(_k("stats"), "requests", 1)

        results = await pipe.execute()
        distinct = int(results[distinct_index]) if distinct_index is not None else 0
        return RequestState(
            block_info=_load_object(results[0]),
            window_rate=int(results[rate_index]),
            distinct_objects=distinct,
        )

    # ------------------------------------------------------------------ blocklist

    async def block_ip(self, ip: str, *, reason: str, threat_type: str, confidence: float, ttl: int) -> None:
        payload = {
            "ip": ip,
            "reason": reason[:500],
            "threat_type": threat_type,
            "confidence": round(confidence, 3),
            "blocked_at": time.time(),
            "ttl": ttl,
        }
        await self._r.set(_k("block", ip), json.dumps(payload), ex=ttl)

    async def refresh_block(self, ip: str, ttl: int) -> bool:
        return bool(await self._r.expire(_k("block", ip), ttl))

    async def get_block(self, ip: str) -> dict[str, Any] | None:
        return _load_object(await self._r.get(_k("block", ip)))

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

    async def get_history(self, ip: str, limit: int = 15) -> list[str]:
        return list(await self._r.lrange(_k("hist", ip), 0, limit - 1))

    async def record_event(self, event: SecurityEvent) -> None:
        pipe = self._r.pipeline(transaction=False)
        pipe.lpush(_k("events"), event.model_dump_json())
        pipe.ltrim(_k("events"), 0, self._max_events - 1)
        pipe.hincrby(_k("stats"), event.action.value, 1)
        await pipe.execute()

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
