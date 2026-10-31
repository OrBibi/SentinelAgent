"""Request-log campaign detection over Redis events."""

from __future__ import annotations

import asyncio
import time

import httpx
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.log_analyst import scan_events
from app.main import create_app
from app.models import Action, SecurityEvent, Tier
from tests.test_proxy import ADMIN, AUTH, make_settings, store_upstream


def _event(
    *,
    ip: str,
    method: str = "GET",
    path: str = "/api/users/51",
    action: Action = Action.FORWARDED,
    status: int | None = 200,
    principal: str = "51",
    threat: str | None = None,
    source: str = "fast_filter",
    ts: float | None = None,
) -> SecurityEvent:
    return SecurityEvent(
        client_ip=ip,
        method=method,
        path=path,
        tier=Tier.CLEAN,
        fast_score=0.0,
        action=action,
        threat_type=threat,
        source=source,
        response_status=status,
        principal=principal,
        timestamp=ts if ts is not None else time.time(),
    )


def test_shopper_on_one_ip_is_not_a_campaign() -> None:
    now = 1_000.0
    events = [
        _event(ip="198.51.100.10", path="/api/orders/1", ts=now - 3),
        _event(ip="198.51.100.10", path="/api/orders/2", ts=now - 2),
        _event(ip="198.51.100.10", method="POST", path="/api/comments", ts=now - 1),
    ]
    assert scan_events(events, now=now) == []


def test_multi_ip_same_principal_is_a_campaign() -> None:
    now = 1_000.0
    events = [
        _event(ip="198.51.100.1", path="/api/users/52", ts=now - 2),
        _event(ip="198.51.100.2", path="/api/users/53", ts=now - 1),
    ]
    findings = scan_events(events, now=now)
    assert len(findings) == 1
    assert findings[0].actor == "51"
    assert findings[0].signals == ("multi_ip",)


def test_three_denies_then_ok_is_a_campaign() -> None:
    now = 1_000.0
    events = [
        _event(ip="198.51.100.1", path="/api/users/52", action=Action.BLOCKED, status=403, ts=now - 4),
        _event(ip="198.51.100.1", path="/api/users/53", action=Action.BLOCKED, status=403, ts=now - 3),
        _event(ip="198.51.100.1", path="/api/users/54", action=Action.BLOCKED, status=403, ts=now - 2),
        _event(ip="198.51.100.1", path="/api/users/51", status=200, ts=now - 1),
    ]
    findings = scan_events(events, now=now)
    assert findings and "deny_then_ok" in findings[0].signals


def test_mass_assignment_alert_then_write_is_not_enough() -> None:
    now = 1_000.0
    events = [
        _event(
            ip="198.51.100.1",
            method="POST",
            path="/api/comments",
            action=Action.ALERT,
            threat="mass_assignment",
            ts=now - 2,
        ),
        _event(ip="198.51.100.1", method="PATCH", path="/api/users/51", ts=now - 1),
    ]
    assert scan_events(events, now=now) == []


def test_injection_alert_then_write_is_a_campaign() -> None:
    now = 1_000.0
    events = [
        _event(
            ip="198.51.100.1",
            method="POST",
            path="/api/comments",
            action=Action.ALERT,
            threat="injection",
            ts=now - 2,
        ),
        _event(ip="198.51.100.1", method="PATCH", path="/api/users/51", ts=now - 1),
    ]
    findings = scan_events(events, now=now)
    assert findings and "alert_then_write" in findings[0].signals


def test_log_agent_rows_are_ignored() -> None:
    now = 1_000.0
    events = [
        _event(ip="198.51.100.1", source="log_agent", ts=now - 2),
        _event(ip="198.51.100.2", source="log_agent", ts=now - 1),
    ]
    assert scan_events(events, now=now) == []


def test_scan_request_log_blocklists_and_records_campaign(tmp_path) -> None:
    redis = FakeAsyncRedis(decode_responses=True)
    app = create_app(
        make_settings(tmp_path),
        redis_client=redis,
        upstream_transport=httpx.MockTransport(store_upstream),
        llm_transport=httpx.MockTransport(lambda _req: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": '{"is_malicious":false,"confidence_score":0.2,"threat_type":"benign","reasoning":"ok"}'}}]},
        )),
    )
    with TestClient(app, client=("198.51.100.40", 5000)) as c:
        async def plant() -> None:
            store = c.app.state.store
            now = time.time()
            await store.record_event(_event(ip="198.51.100.1", path="/api/users/52", ts=now - 2))
            await store.record_event(_event(ip="198.51.100.2", path="/api/users/53", ts=now - 1))
            findings = await c.app.state.worker.scan_request_log()
            assert findings and findings[0].signals == ("multi_ip",)
            assert await store.is_watched("51")

        asyncio.run(plant())
        events = c.get("/sentinel/admin/events", headers=ADMIN).json()
        assert any(e["source"] == "log_agent" and e["threat_type"] == "campaign" for e in events)
        assert c.get("/api/users/51", headers=AUTH).status_code == 403
