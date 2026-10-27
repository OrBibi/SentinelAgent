"""Offline stand-in for Baseten's Chat Completions API, for demos without an API key.

Returns a verdict in the same JSON contract the real model is prompted for, based on
simple keyword checks of the prompt. Run: uvicorn examples.mock_baseten:app --port 9100
and set BASETEN_BASE_URL=http://localhost:9100/v1 with any non-empty BASETEN_API_KEY.
"""

from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request

app = FastAPI(title="Mock Baseten")


def _verdict(prompt: str) -> dict[str, object]:
    # Match only phrases emitted by the fast filter and schema validator, not retrieved threat docs.
    p = prompt.lower()
    if "path traversal signature" in p:
        return {"is_malicious": True, "confidence_score": 0.97, "threat_type": "path_traversal",
                "reasoning": "Dot-dot segments target files outside the web root."}
    if "distinct object ids on one route" in p:
        return {"is_malicious": True, "confidence_score": 0.9, "threat_type": "bola",
                "reasoning": "The client enumerates many object IDs on one route, a BOLA pattern."}
    if "undeclared body fields (possible mass assignment)" in p:
        return {"is_malicious": True, "confidence_score": 0.78, "threat_type": "mass_assignment",
                "reasoning": "The body sets fields not declared in the OpenAPI schema."}
    if "path is not declared in any loaded" in p or "method not declared for /" in p:
        return {"is_malicious": True, "confidence_score": 0.6, "threat_type": "api_intrusion",
                "reasoning": "The request targets an operation absent from the API inventory."}
    if "injection signature" in p:
        return {"is_malicious": True, "confidence_score": 0.88, "threat_type": "injection",
                "reasoning": "Quote-balanced tautology indicates SQL injection."}
    return {"is_malicious": False, "confidence_score": 0.2, "threat_type": "benign",
            "reasoning": "No concrete malicious intent found."}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> dict[str, object]:
    body = await request.json()
    prompt = "\n".join(m.get("content", "") for m in body.get("messages", []))
    content = json.dumps(_verdict(prompt))
    return {
        "id": "mock-1",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model"),
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }
