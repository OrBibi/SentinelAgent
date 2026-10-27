"""Unit tests for the synchronous filter, schema index, verdict parsing, and mitigation policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.agent_worker import MitigationPolicy, build_messages, normalize_threat_type, parse_verdict
from app.ml_filter import FastFilter, heuristic_template
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


def test_injection_is_suspicious_not_hard_blocked(ff: FastFilter) -> None:
    v = verdict_for(ff, "POST", "/login", body=b'{"user":"admin\' OR 1=1 --"}')
    assert v.tier is Tier.SUSPICIOUS
    assert v.threat_hint == "injection"


def test_schema_violation_and_bola_signals(ff: FastFilter) -> None:
    assert verdict_for(ff, "DELETE", "/api/users/42", route_known=False).threat_hint == "schema_violation"
    bola = verdict_for(ff, "GET", "/api/orders/77", distinct_objects=20)
    assert bola.tier is Tier.SUSPICIOUS and bola.threat_hint == "bola"


def test_privileged_mass_assignment_is_suspicious(ff: FastFilter) -> None:
    v = verdict_for(ff, "PATCH", "/api/users/1", undeclared_fields=["role"], privileged_fields=["role"])
    assert v.tier is Tier.SUSPICIOUS and v.threat_hint == "mass_assignment"


def test_compiled_forest_matches_sklearn(ff: FastFilter) -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        row = tuple(float(x) for x in rng.uniform(0, 6, size=9))
        expected = -ff.forest.score_samples(np.array([row]))[0]
        assert ff._compiled.anomaly_score(row) == pytest.approx(expected, rel=1e-9)


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


@pytest.mark.parametrize(
    ("verdict", "findings", "tier", "expected"),
    [
        (None, None, Tier.SUSPICIOUS, Action.ALERT),
        (AgentVerdict(is_malicious=False, confidence_score=0.9), None, Tier.SUSPICIOUS, Action.IGNORED),
        (AgentVerdict(is_malicious=True, confidence_score=0.9, threat_type="other"), None, Tier.SUSPICIOUS, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="bola"), None, Tier.SUSPICIOUS, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="other"), None, Tier.SUSPICIOUS, Action.ALERT),
        (
            AgentVerdict(is_malicious=True, confidence_score=0.75, threat_type="mass_assignment"),
            ["undeclared body fields (possible mass assignment): role"],
            Tier.SUSPICIOUS,
            Action.BLOCKLISTED,
        ),
        (AgentVerdict(is_malicious=True, confidence_score=0.72, threat_type="other"), None, Tier.HARD_BLOCK, Action.BLOCKLISTED),
        (AgentVerdict(is_malicious=True, confidence_score=0.5, threat_type="path_traversal"), None, Tier.HARD_BLOCK, Action.ALERT),
    ],
)
def test_mitigation_policy(verdict: AgentVerdict | None, findings: list[str] | None, tier: Tier, expected: Action) -> None:
    policy = MitigationPolicy(block_confidence=0.85, critical_confidence=0.7)
    action, _ = policy.decide(verdict, _req(tier), _ctx(findings))
    assert action is expected
