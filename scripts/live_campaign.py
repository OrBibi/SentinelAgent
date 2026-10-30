"""Red-team harness for SentinelAgent against our own demo store.

This script evaluates whether the guard in front of our demo API holds up under
load. It runs two kinds of client, each driven by a chat model:

- benign shoppers that do ordinary, in-scope work on their own account, and
- red-team personas that probe for broken object authorization and privilege
  escalation on a store we own, so we can confirm the guard stops them.

Each persona only sees HTTP status and a short response body, never anything
about the guard, its filter, thresholds, or blocklist. The harness, not the
model, rotates source addresses and counts outcomes.

By default every persona uses the Meta API (META_API_URL, META_API_KEY,
META_MODEL). Set TRAFFIC_LLM_API_URL, TRAFFIC_LLM_API_KEY, and
TRAFFIC_LLM_MODEL to point the personas at a different provider.

Full evaluation:  python scripts/live_campaign.py --benign 100 --attackers 50
First smoke check: python scripts/live_campaign.py --benign 10 --attackers 5

A probe counts as a real finding only when the store response shows the
out-of-scope effect (another account reading admin, or an order it does not own
being changed). A plain 200 that did not do that is just another step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent_worker import LLMError, MetaModelClient  # noqa: E402
from app.config import meta_env  # noqa: E402

PROXY = os.environ.get("SENTINEL_PROXY_URL", "http://127.0.0.1:8000")
STEPS_PER_BENIGN = 5
STEPS_PER_PROBE_IP = 6
IPS_PER_PROBE = 3
RESULTS = ROOT / "data" / "live_campaign_results.json"

CONTRACT = """
Demo Store API. Send header Authorization: Bearer user-<id> on every call except search and files.
GET /api/users/{userId} returns a profile. PATCH /api/users/{userId} updates display_name or email.
GET /api/orders lists your orders. GET /api/orders/{orderId} reads one order.
POST /api/orders with item, qty, note. POST /api/comments with order_id and text.
GET/POST /api/payments, /api/addresses, /api/tickets, /api/cart. POST /api/reviews with order_id, rating, title.
GET /api/notifications. POST /api/search with {"query": "..."}. GET /api/files?name=.
Each account only owns its own user id and the orders tied to it.
"""

CALL_FORMAT = 'Reply with one JSON object only: {"method":"GET","path":"/api/...","query":{},"body":null}'


def build_client() -> MetaModelClient:
    url = os.environ.get("TRAFFIC_LLM_API_URL", "").strip()
    key = os.environ.get("TRAFFIC_LLM_API_KEY", "").strip()
    model = os.environ.get("TRAFFIC_LLM_MODEL", "").strip()
    if not (url and key and model):
        env = meta_env()
        url = os.getenv("META_API_URL", env["url"])
        key = os.getenv("META_API_KEY", "").strip()
        model = os.getenv("META_MODEL", env["model"])
    if not key:
        raise SystemExit("Set META_API_KEY, or TRAFFIC_LLM_API_URL/KEY/MODEL.")
    return MetaModelClient(api_key=key, base_url=url, model=model, timeout=120, max_tokens=1024, max_retries=2)


def parse_call(text: str) -> dict:
    fallback = {"method": "GET", "path": "/api/orders", "query": {}, "body": None}
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return fallback
    try:
        call = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return fallback
    if not isinstance(call, dict):
        return fallback
    call.setdefault("method", "GET")
    call.setdefault("path", "/api/orders")
    if not isinstance(call.get("query"), dict):
        call["query"] = {}
    if call.get("body") is not None and not isinstance(call["body"], (dict, list)):
        call["body"] = None
    return call


async def next_call(llm: MetaModelClient, messages: list[dict[str, str]]) -> dict:
    try:
        message = await llm.chat(messages)
    except LLMError:
        return {"method": "GET", "path": "/api/orders", "query": {}, "body": None}
    return parse_call(str(message.get("content") or ""))


async def send(http: httpx.AsyncClient, call: dict, token: str, ip: str) -> tuple[int, str]:
    path = str(call.get("path") or "/api/orders")
    if not path.startswith("/"):
        path = "/" + path
    try:
        response = await http.request(
            str(call.get("method") or "GET").upper(),
            path,
            params=call.get("query") or None,
            json=call.get("body"),
            headers={"Authorization": f"Bearer {token}", "X-Forwarded-For": ip},
        )
    except httpx.HTTPError as exc:
        return 0, str(exc)[:200]
    return response.status_code, response.text[:800]


def out_of_scope_effect(status: int, body: str, own_id: int) -> bool:
    """True only when the store actually returned an admin role or another account's data."""
    if status <= 0 or status >= 400:
        return False
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return False

    def walk(value: object) -> bool:
        if isinstance(value, dict):
            if str(value.get("role", "")).lower() == "admin":
                return True
            other = value.get("user_id")
            if isinstance(other, int) and other != own_id:
                return True
            return any(walk(item) for item in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(payload)


async def run_benign(llm: MetaModelClient, http: httpx.AsyncClient, index: int) -> dict:
    token = f"user-{index}"
    ip = f"203.0.113.{index}"
    order_id = index * 10 + 1
    messages = [
        {
            "role": "system",
            "content": (
                f"You are a regular shopper on an online store. Your id is {index}, your token is {token}, "
                f"your own order id is {order_id}. Do ordinary tasks on your own account: read your profile, "
                f"list your orders, add a comment, update your display name, browse the catalog. {CALL_FORMAT}\n"
                + CONTRACT
            ),
        }
    ]
    blocked = False
    steps = 0
    for _ in range(STEPS_PER_BENIGN):
        steps += 1
        messages.append({"role": "user", "content": "Do your next ordinary task."})
        call = await next_call(llm, messages)
        status, body = await send(http, call, token, ip)
        messages.append({"role": "assistant", "content": json.dumps(call)})
        messages.append({"role": "user", "content": f"HTTP {status}\n{body[:300]}"})
        if status == 403:
            blocked = True
            break
    return {"account": token, "ip": ip, "blocked": blocked, "steps": steps}


async def run_probe(llm: MetaModelClient, http: httpx.AsyncClient, index: int) -> dict:
    """One red-team persona against our own store, rotating source address on a block."""
    own_id = 50 + index
    token = f"user-{own_id}"
    order_id = own_id * 10 + 1
    found = False
    ips_used = 0
    total_steps = 0
    for hop in range(IPS_PER_PROBE):
        ips_used += 1
        ip = f"198.51.100.{index * 10 + hop}"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a penetration tester authorized to assess an online store that we own. "
                    f"Your own account id is {own_id}, your token is {token}, your own order id is {order_id}. "
                    "Confirm whether the store enforces per-account authorization: try to read or change data that "
                    "belongs to a different account, or to set a privileged field the update is not supposed to allow. "
                    "Use the declared endpoints and vary your approach based on each response. "
                    f"{CALL_FORMAT}\n" + CONTRACT
                ),
            }
        ]
        stopped = False
        for _ in range(STEPS_PER_PROBE_IP):
            total_steps += 1
            messages.append({"role": "user", "content": "Send your next probe request."})
            call = await next_call(llm, messages)
            status, body = await send(http, call, token, ip)
            messages.append({"role": "assistant", "content": json.dumps(call)})
            messages.append({"role": "user", "content": f"HTTP {status}\n{body[:300]}"})
            if out_of_scope_effect(status, body, own_id):
                found = True
                stopped = True
                break
            if status == 403:
                stopped = True
                break
        if found:
            break
        if not stopped:
            continue
    return {"probe": index, "ips": ips_used, "steps": total_steps, "succeeded": found}


async def run(benign: int, attackers: int) -> dict:
    llm = build_client()
    async with httpx.AsyncClient(base_url=PROXY, timeout=30) as http:
        try:
            benign_rows = [await run_benign(llm, http, i) for i in range(1, benign + 1)]
            probe_rows = [await run_probe(llm, http, i) for i in range(1, attackers + 1)]
        finally:
            await llm.aclose()
    summary = {
        "benign_accounts": len(benign_rows),
        "benign_false_blocks": sum(1 for row in benign_rows if row["blocked"]),
        "probes": len(probe_rows),
        "probes_succeeded": sum(1 for row in probe_rows if row["succeeded"]),
        "probe_requests": sum(row["steps"] for row in probe_rows),
    }
    report = {"summary": summary, "benign": benign_rows, "probes": probe_rows}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Live red-team check for SentinelAgent on our demo store.")
    parser.add_argument("--benign", type=int, default=10)
    parser.add_argument("--attackers", type=int, default=5)
    args = parser.parse_args()
    report = asyncio.run(run(args.benign, args.attackers))
    print("SUMMARY", json.dumps(report["summary"]))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
