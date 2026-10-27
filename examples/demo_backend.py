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


@app.get("/api/users/{user_id}/orders")
async def list_user_orders(user_id: int) -> dict[str, object]:
    return {"user_id": user_id, "orders": [{"id": 10, "item": "notebook"}]}


@app.post("/api/search")
async def search_catalog(request: Request) -> dict[str, object]:
    body = await request.json()
    return {"query": body.get("query", ""), "hits": [{"item": "notebook"}]}


@app.post("/api/comments", status_code=201)
async def add_comment(request: Request) -> dict[str, object]:
    return {"saved": await request.json()}


@app.get("/api/payments")
async def list_payments(page: int = 1) -> dict[str, object]:
    return {"page": page, "payments": [{"id": 1, "amount": 12.5, "currency": "USD"}]}


@app.post("/api/payments", status_code=201)
async def create_payment(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/payments/{payment_id}")
async def get_payment(payment_id: int) -> dict[str, object]:
    return {"id": payment_id, "amount": 12.5, "currency": "USD"}


@app.get("/api/addresses")
async def list_addresses() -> dict[str, object]:
    return {"addresses": [{"id": 1, "city": "Haifa"}]}


@app.post("/api/addresses", status_code=201)
async def create_address(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/tickets")
async def list_tickets() -> dict[str, object]:
    return {"tickets": [{"id": 1, "subject": "Late delivery"}]}


@app.post("/api/tickets", status_code=201)
async def create_ticket(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: int) -> dict[str, object]:
    return {"id": ticket_id, "subject": "Late delivery", "body": "Package was a day late."}


@app.post("/api/reviews", status_code=201)
async def create_review(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/cart")
async def get_cart() -> dict[str, object]:
    return {"items": [{"item": "notebook", "qty": 1}]}


@app.post("/api/cart", status_code=201)
async def add_cart_item(request: Request) -> dict[str, object]:
    return {"created": await request.json()}


@app.get("/api/notifications")
async def list_notifications(unread: bool = False) -> dict[str, object]:
    return {"unread": unread, "items": [{"id": 1, "text": "Order shipped"}]}


@app.get("/api/files")
async def download_file(name: str) -> dict[str, str]:
    return {"name": name, "content": "public document"}
