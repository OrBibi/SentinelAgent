"""End-to-end proxy tests with in-memory Redis and mocked upstream / Baseten transports."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import Settings
from app.main import create_app, normalize_proxy_path, same_upstream
from app.redis_store import RedisStore

ADMIN = {"X-Admin-Token": "test-token"}
DEMO_SCHEMA = Path(__file__).resolve().parent.parent / "examples" / "demo_openapi.yaml"


def upstream_handler(request: httpx.Request) -> httpx.Response:
    body = json.dumps({"path": request.url.raw_path.decode(), "xff": request.headers.get("x-forwarded-for")}).encode()
    # A streamed body mirrors real transports; `content=` responses are pre-read and cannot be re-streamed.
    return httpx.Response(
        200,
        headers=[("content-type", "application/json"), ("set-cookie", "a=1"), ("set-cookie", "b=2")],
        stream=httpx.ByteStream(body),
    )


def llm_handler_factory(calls: list[dict]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        prompt = body["messages"][1]["content"]
        if "path traversal signature" in prompt:
            verdict = {"is_malicious": True, "confidence_score": 0.96, "threat_type": "path_traversal", "reasoning": "dot-dot"}
        else:
            verdict = {"is_malicious": False, "confidence_score": 0.3, "threat_type": "benign", "reasoning": "ok"}
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": json.dumps(verdict)}}]})

    return handler


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(
        upstream_base_url="http://upstream.test",
        redis_url="memory://",
        meta_api_key="test-key",
        meta_api_url="http://llm.test/v1",
        meta_model="muse-spark-1.3",
        enforce_blocks=True,
        admin_token="test-token",
        chroma_dir=str(tmp_path / "chroma"),
        enable_chroma=False,
        agent_cooldown_seconds=0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def llm_calls() -> list[dict]:
    return []


@pytest.fixture
def client(tmp_path: Path, llm_calls: list[dict]) -> Iterator[TestClient]:
    app = create_app(
        make_settings(tmp_path),
        redis_client=FakeAsyncRedis(decode_responses=True),
        upstream_transport=httpx.MockTransport(upstream_handler),
        llm_transport=httpx.MockTransport(llm_handler_factory(llm_calls)),
    )
    with TestClient(app, client=("198.51.100.10", 5000)) as c:
        yield c


def wait_for_agent(c: TestClient, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = c.get("/sentinel/admin/stats", headers=ADMIN).json()["agent"]
        if stats["queue_depth"] == 0 and stats["processed"] > 0:
            return
        time.sleep(0.05)
    raise AssertionError("agent did not finish in time")


def test_clean_request_is_forwarded_with_guard_headers(client: TestClient, llm_calls: list[dict]) -> None:
    r = client.get("/api/users/42?page=1")
    assert r.status_code == 200
    assert r.json() == {"path": "/api/users/42?page=1", "xff": "198.51.100.10"}
    assert r.headers["x-sentinel-tier"] == "clean"
    assert float(r.headers["x-sentinel-guard-ms"]) >= 0
    assert r.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert llm_calls == []


def test_traversal_blocks_immediately_then_agent_blocklists(client: TestClient, llm_calls: list[dict]) -> None:
    r = client.get("/api/files?name=../../etc/passwd")
    assert r.status_code == 403
    assert set(r.json()) == {"error", "message", "request_id"}
    assert "traversal" not in r.text

    wait_for_agent(client)
    assert len(llm_calls) == 1
    assert llm_calls[0]["model"] == "muse-spark-1.3"
    assert "<request>" in llm_calls[0]["messages"][1]["content"]

    assert client.get("/api/users/42").status_code == 403
    blocked = client.get("/sentinel/admin/blocklist", headers=ADMIN).json()
    assert blocked[0]["ip"] == "198.51.100.10" and blocked[0]["threat_type"] == "path_traversal"

    events = client.get("/sentinel/admin/events", headers=ADMIN).json()
    assert {(e["source"], e["action"]) for e in events} == {("fast_filter", "blocked"), ("meta_agent", "blocklisted")}

    assert client.delete("/sentinel/admin/blocklist/198.51.100.10", headers=ADMIN).status_code == 200
    assert client.get("/api/users/42").status_code == 200


def test_enforce_blocks_false_logs_decision_without_denying_later_requests(
    tmp_path: Path, llm_calls: list[dict]
) -> None:
    app = create_app(
        make_settings(tmp_path, enforce_blocks=False),
        redis_client=FakeAsyncRedis(decode_responses=True),
        upstream_transport=httpx.MockTransport(upstream_handler),
        llm_transport=httpx.MockTransport(llm_handler_factory(llm_calls)),
    )
    with TestClient(app, client=("198.51.100.10", 5000)) as client:
        assert client.get("/api/files?name=../../etc/passwd").status_code == 403
        wait_for_agent(client)
        assert client.get("/sentinel/admin/blocklist", headers=ADMIN).json() == []
        assert client.get("/api/users/42").status_code == 200
        events = client.get("/sentinel/admin/events", headers=ADMIN).json()
        logged = [e for e in events if e["source"] == "meta_agent"]
        assert logged[0]["action"] == "blocklisted"
        assert logged[0]["extra"]["enforced"] is False


def test_suspicious_request_is_forwarded_without_waiting(client: TestClient) -> None:
    client.post("/sentinel/admin/openapi?source=demo", headers=ADMIN, content=DEMO_SCHEMA.read_bytes())
    r = client.delete("/api/users/42")
    assert r.status_code == 200
    assert r.headers["x-sentinel-tier"] == "suspicious"

    wait_for_agent(client)
    events = client.get("/sentinel/admin/events", headers=ADMIN).json()
    assert events[0]["action"] == "ignored"
    assert any("method not declared" in f for f in events[0]["extra"]["schema_findings"])
    assert client.get("/api/users/42").status_code == 200


def test_openapi_ingestion_endpoint(client: TestClient) -> None:
    assert client.post("/sentinel/admin/openapi", content=DEMO_SCHEMA.read_bytes()).status_code == 401
    r = client.post("/sentinel/admin/openapi?source=demo", headers=ADMIN, content=DEMO_SCHEMA.read_bytes())
    assert r.status_code == 200 and r.json()["operations"] == 21
    files = {"file": ("api.yaml", DEMO_SCHEMA.read_bytes(), "application/yaml")}
    r = client.post("/sentinel/admin/openapi", headers=ADMIN, files=files, data={"source": "uploaded"}, follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/sentinel/admin/openapi", headers=ADMIN).json()["sources"] == {"demo": 21, "uploaded": 21}
    assert client.post("/sentinel/admin/openapi", headers=ADMIN, content=b"[1, 2]").status_code == 422


def test_rate_limit_returns_429(tmp_path: Path) -> None:
    app = create_app(
        make_settings(tmp_path, rate_limit_per_window=5, meta_api_key=""),
        redis_client=FakeAsyncRedis(decode_responses=True),
        upstream_transport=httpx.MockTransport(upstream_handler),
    )
    with TestClient(app) as c:
        codes = [c.get("/api/orders").status_code for _ in range(8)]
    assert codes == [200] * 5 + [429] * 3


def test_admin_and_reserved_paths(client: TestClient) -> None:
    assert client.get("/sentinel/admin/events").status_code == 401
    assert client.get("/sentinel/admin/events", headers={"X-Admin-Token": "wrong"}).status_code == 401
    assert client.get("/sentinel/unknown").status_code == 404
    r = client.get("/sentinel/admin?token=test-token", follow_redirects=False)
    assert r.status_code == 303 and "httponly" in r.headers["set-cookie"].lower()
    assert client.get("/sentinel/admin").status_code == 200
    assert client.get("/sentinel/health").json()["redis"] is True


def test_scheme_relative_target_collapses_to_the_inspected_path() -> None:
    assert normalize_proxy_path("//api/users/1") == "/api/users/1"
    assert normalize_proxy_path("///api/users/1") == "/api/users/1"
    assert normalize_proxy_path("/api/users/1") == "/api/users/1"
    base = httpx.URL("http://upstream.test")
    assert same_upstream(httpx.URL("http://upstream.test/api/users/1"), base)
    assert not same_upstream(httpx.URL("http://evil.com/api/users/1"), base)
    assert not same_upstream(httpx.URL("https://upstream.test/api/users/1"), base)


def test_memory_redis_and_corrupt_block_entry() -> None:
    import asyncio

    async def inner() -> None:
        store = RedisStore.from_url(
            "memory://",
            rate_window_seconds=10,
            bola_window_seconds=60,
            history_size=5,
            max_events=10,
        )
        assert await store.ping()
        planted = await store.check_request("203.0.113.9", history_entry="GET /a", route_template="/orders/{id}", object_id="7")
        assert planted.window_rate == 1 and planted.distinct_objects == 1 and planted.block_info is None
        again = await store.check_request("203.0.113.9", history_entry="GET /b", route_template="/orders/{id}", object_id="8")
        assert again.distinct_objects == 2
        await store._r.set("sentinel:block:203.0.113.9", "{not-json")
        corrupt = await store.check_request("203.0.113.9", history_entry="GET /c")
        assert corrupt.block_info is None and corrupt.window_rate == 3
        await store.close()

    asyncio.run(inner())


def test_redis_outage_fails_open_or_closed(tmp_path: Path) -> None:
    class BrokenRedis(FakeAsyncRedis):
        def pipeline(self, *args, **kwargs):  # type: ignore[override]
            raise RedisConnectionError("down")

    for mode, expected in (("open", 200), ("closed", 503)):
        app = create_app(
            make_settings(tmp_path, fail_mode=mode, meta_api_key=""),
            redis_client=BrokenRedis(decode_responses=True),
            upstream_transport=httpx.MockTransport(upstream_handler),
        )
        with TestClient(app) as c:
            assert c.get("/api/orders").status_code == expected
            # Signature blocking keeps working even when Redis is down.
            if mode == "open":
                assert c.get("/api/files?name=../../etc/passwd").status_code == 403
