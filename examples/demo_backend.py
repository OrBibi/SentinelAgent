"""Minimal upstream API used to demo SentinelAgent locally.

Run: uvicorn examples.demo_backend:app --port 9000
"""

from __future__ import annotations

from fastapi import FastAPI, Request

app = FastAPI(title="Demo Store API")


@app.get("/api/users/{user_id}")
async def get_user(user_id: int) -> dict[str, object]:
    return {"id": user_id, "display_name": f"user-{user_id}"}


@app.patch("/api/users/{user_id}")
async def update_user(user_id: int, request: Request) -> dict[str, object]:
    return {"id": user_id, "updated": await request.json()}


@app.get("/api/orders")
async def list_orders(page: int = 1, status: str | None = None) -> dict[str, object]:
    return {"page": page, "status": status, "orders": [{"id": 1}, {"id": 2}]}


@app.get("/api/orders/{order_id}")
async def get_order(order_id: int) -> dict[str, object]:
    return {"id": order_id, "item": "book", "qty": 1}


@app.post("/api/orders", status_code=201)
async def create_order(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/files")
async def download_file(name: str) -> dict[str, str]:
    return {"name": name, "content": "public document"}
