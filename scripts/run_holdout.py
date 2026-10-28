"""Score one holdout product with its own OpenAPI file and labeled samples.

Clean requests are judged by the fast filter only. Everything else goes through the
locked agent policy, without the fast-filter numeric score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent_worker import (  # noqa: E402
    MetaModelClient,
    MitigationPolicy,
    build_messages,
    investigate,
)
from app.ml_filter import FastFilter  # noqa: E402
from app.models import SuspiciousRequest, Tier  # noqa: E402
from app.rag_engine import RAGEngine, load_schema_document, needs_agent_review, review_classes, undeclared_body_fields  # noqa: E402

POLICY = MitigationPolicy(block_confidence=0.85, critical_confidence=0.7)
SAMPLE_FILES = ("benign.json", "suspicious_benign.json", "malicious.json")


def server_prefix(doc: dict) -> str:
    for server in doc.get("servers") or []:
        url = str(server.get("url", "")) if isinstance(server, dict) else ""
        path = urlparse(url).path if "://" in url else url
        path = path.rstrip("/")
        if path:
            return path
    return ""


def query_string(sample: dict) -> str:
    query = sample.get("query") or ""
    if isinstance(query, dict):
        return urlencode({key: "" if value is None else value for key, value in query.items()}, doseq=True)
    return str(query)


def request_path(sample: dict, prefix: str) -> str:
    path = str(sample.get("path") or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    if prefix and not (path == prefix or path.startswith(prefix + "/")):
        return prefix + path
    return path


def load_samples(directory: Path, prefix: str) -> list[dict]:
    session_dir = directory / "sessions"
    session_files = sorted(session_dir.glob("*.json")) if session_dir.is_dir() else []
    if session_files:
        return load_sessions(session_files, prefix)
    samples: list[dict] = []
    for name in SAMPLE_FILES:
        path = directory / "samples" / name
        if not path.exists():
            continue
        batch = json.loads(path.read_text(encoding="utf-8"))
        for sample in batch:
            item = dict(sample)
            item["path"] = request_path(sample, prefix)
            item["query"] = query_string(sample)
            item["client_ip"] = str(sample.get("client_ip") or "203.0.113.50")
            samples.append(item)
    return samples


def load_sessions(paths: list[Path], prefix: str) -> list[dict]:
    samples: list[dict] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        sessions = document if isinstance(document, list) else [document]
        for session in sessions:
            client_ip = str(session.get("client_ip") or "203.0.113.50")
            for step in session.get("steps") or []:
                item = dict(step)
                item["path"] = request_path(step, prefix)
                item["query"] = query_string(step)
                item["client_ip"] = client_ip
                item["session_id"] = session.get("session_id")
                samples.append(item)
    return samples


def assign_object_history(engine: RAGEngine, samples: list[dict]) -> None:
    """Replay each client_ip in file order, the same way Redis would see one machine."""
    seen: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        route = engine.routes.classify(sample["method"], sample["path"])
        template = route.template or sample["path"]
        client_ip = str(sample.get("client_ip") or "203.0.113.50")
        key = (client_ip, template)
        prior = sorted(seen.get(key, set()))
        sample["prior_object_ids"] = prior
        sample["route_template"] = template
        sample["client_ip"] = client_ip
        if route.object_id:
            bucket = seen.setdefault(key, set())
            bucket.add(route.object_id)
            sample["distinct_objects"] = len(bucket)
        else:
            sample["distinct_objects"] = 0
        sample["review"] = sorted(review_classes(route.spec))


def score_sample(engine: RAGEngine, fast: FastFilter, sample: dict):
    method = sample["method"]
    path = sample["path"]
    query = sample.get("query") or ""
    body = b"" if not sample.get("body") else json.dumps(sample["body"]).encode()
    route = engine.routes.classify(method, path)
    content_type = "application/json" if body else ""
    undeclared, privileged = undeclared_body_fields(route.spec, body, content_type)
    features = fast.extract_features(
        method,
        path,
        query,
        body,
        window_rate=int(sample.get("window_rate") or 3),
        route_known=route.known,
        matched_route=route.template,
        undeclared_fields=undeclared,
        privileged_fields=privileged,
        headers=sample.get("headers") or None,
    )
    verdict = fast.evaluate(features, distinct_objects=int(sample.get("distinct_objects") or 0))
    return route, features, verdict, body


def caught(row: dict) -> bool:
    return row["tier"] == Tier.HARD_BLOCK.value or row.get("action") == "blocklisted"


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for label in ("malicious", "benign", "suspicious_benign"):
        group = [row for row in rows if row["label"] == label]
        blocked = [row for row in group if caught(row)]
        summary[label] = {
            "total": len(group),
            "blocked": len(blocked),
            "forwarded": len(group) - len(blocked),
            "tiers": dict(Counter(row["tier"] for row in group)),
        }
    malicious = summary["malicious"]["total"]
    benign_total = summary["benign"]["total"] + summary["suspicious_benign"]["total"]
    tp = summary["malicious"]["blocked"]
    fp = summary["benign"]["blocked"] + summary["suspicious_benign"]["blocked"]
    scored = malicious + benign_total
    summary["defense"] = {
        "tp": tp,
        "fn": malicious - tp,
        "fp": fp,
        "tn": benign_total - fp,
        "accuracy": round((tp + benign_total - fp) / scored, 3) if scored else 0,
    }
    return summary


async def run_agent(
    engine: RAGEngine, fast: FastFilter, samples: list[dict], *, force_agent: bool = False
) -> list[dict]:
    client = MetaModelClient.from_env(timeout=120, max_tokens=4096, max_retries=2)
    if client is None:
        raise SystemExit("META_API_KEY is empty")
    semaphore = asyncio.Semaphore(3)

    async def one(sample: dict) -> dict:
        _route, features, fast_verdict, body = score_sample(engine, fast, sample)
        base = {
            "id": sample["id"],
            "label": sample["label"],
            "note": sample.get("note", ""),
            "method": sample["method"],
            "path": sample["path"],
            "tier": fast_verdict.tier.value,
            "fast_score": fast_verdict.score,
            "reasons": fast_verdict.reasons,
        }
        if fast_verdict.tier is Tier.HARD_BLOCK and not force_agent:
            base["action"] = "blocked"
            print(sample["id"], "blocked", flush=True)
            return base
        review = frozenset(sample.get("review") or [])
        distinct = int(sample.get("distinct_objects") or 0)
        route = engine.routes.classify(sample["method"], sample["path"])
        if review != review_classes(route.spec):
            review = review_classes(route.spec)
        if (
            not force_agent
            and fast_verdict.tier is Tier.CLEAN
            and not needs_agent_review(route.spec, distinct)
        ):
            base["action"] = "forwarded"
            print(sample["id"], "forwarded", flush=True)
            return base
        request = SuspiciousRequest(
            client_ip=str(sample.get("client_ip") or "203.0.113.50"),
            method=sample["method"],
            path=sample["path"],
            query=sample.get("query") or "",
            body_preview=body[:2000].decode("utf-8", "replace"),
            headers={key: str(value)[:300] for key, value in (sample.get("headers") or {}).items()},
            features=features,
            fast_verdict=fast_verdict,
        )
        context = await asyncio.to_thread(engine.retrieve_context, request)
        prior = list(sample.get("prior_object_ids") or [])
        messages = build_messages(
            request, [], context, include_fast_score=False, object_ids=prior, review=review
        )
        ledger = {"template": sample.get("route_template") or "", "ids": prior}
        async with semaphore:
            verdict, raw, tool_trace, error, chain = await investigate(
                client, engine, messages, req=request, ctx=context, ledger=ledger
            )
        decided, rationale = POLICY.decide(verdict, request, context)
        if error:
            rationale = f"{rationale}; {error}"
        action = decided.value if force_agent or fast_verdict.tier is not Tier.HARD_BLOCK else "blocked"
        print(sample["id"], action, flush=True)
        base.update(
            {
                "action": action,
                "agent_action": decided.value,
                "rationale": rationale,
                "tool_trace": tool_trace,
                "chain": chain,
                "verdict": None
                if verdict is None
                else {
                    "is_malicious": verdict.is_malicious,
                    "confidence_score": verdict.confidence_score,
                    "threat_type": verdict.threat_type,
                    "recommended_action": verdict.recommended_action,
                    "reasoning": verdict.reasoning,
                    "raw": raw[:500],
                    "error": error,
                },
            }
        )
        return base

    try:
        return list(await asyncio.gather(*(one(sample) for sample in samples)))
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct-malicious", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    document = load_schema_document((directory / "openapi.yaml").read_bytes())
    prefix = server_prefix(document)
    samples = load_samples(directory, prefix)
    engine = RAGEngine.create(chroma_dir=str(ROOT / "data" / "holdout_chroma"), enable_chroma=False)
    engine.bootstrap()
    engine.ingest_openapi((directory / "openapi.yaml").read_bytes(), source=directory.name, persist=False)
    fast = FastFilter(suspicious_threshold=0.45, hard_block_threshold=0.9, bola_distinct_threshold=15)
    assign_object_history(engine, samples)
    preview = []
    for sample in samples:
        _route, _features, verdict, _body = score_sample(engine, fast, sample)
        preview.append((sample["label"], verdict.tier.value))
    print("prefix", prefix or "(none)", "samples", len(samples))
    print("tiers", dict(Counter(preview)))
    review_n = sum(1 for sample in samples if needs_agent_review(engine.routes.classify(sample["method"], sample["path"]).spec, int(sample.get("distinct_objects") or 0)))
    print("review_routes", review_n)
    if args.dry_run:
        return
    if args.direct_malicious:
        samples = [sample for sample in samples if sample["label"] == "malicious"]
    rows = asyncio.run(run_agent(engine, fast, samples, force_agent=args.direct_malicious))
    report = {"prefix": prefix, "direct_malicious": args.direct_malicious, "summary": summarize(rows), "rows": rows}
    out = directory / ("malicious_direct_llm.json" if args.direct_malicious else "results.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(report["summary"], ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()
