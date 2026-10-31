"""Stateful Demo Store API used behind SentinelAgent.

Run: uvicorn examples.demo_backend:app --port 9000

Bearer token ``user-12`` is user 12. Authentication is required. This store is
an evaluation fixture for the guard in front of it: object reads (and profile
writes) are not re-checked against the caller, and file download is an in-memory
map with no extra path filter. Extra fields on a profile update are stored as
sent, so a campaign can see whether a write actually landed.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request

app = FastAPI(title="Demo Store API")

CATALOG = [
    {"item": "notebook", "price": 12.5},
    {"item": "book", "price": 18.0},
    {"item": "pen", "price": 3.5},
]


def _seed() -> None:
    for user_id in range(1, 101):
        USERS[user_id] = {
            "id": user_id,
            "display_name": f"user-{user_id}",
            "email": f"user{user_id}@demo-store.test",
            "role": "customer",
        }
        order_id = user_id * 10 + 1
        ORDERS[order_id] = {
            "id": order_id,
            "user_id": user_id,
            "item": "notebook",
            "qty": 1,
            "status": "open",
            "note": "",
        }
        PAYMENTS[order_id] = {
            "id": order_id,
            "user_id": user_id,
            "order_id": order_id,
            "amount": 12.5,
            "currency": "USD",
        }
        ADDRESSES[user_id] = [
            {"id": user_id, "user_id": user_id, "line1": f"{user_id} Harbor St", "city": "Haifa", "postal_code": "31000"}
        ]
        TICKETS[user_id] = [
            {
                "id": user_id,
                "user_id": user_id,
                "subject": "Order update",
                "body": "The notebook shipped.",
            }
        ]
        CARTS[user_id] = [{"item": "notebook", "qty": 1}]
        NOTES[user_id] = [{"id": 1, "text": "Order shipped", "unread": False}]


USERS: dict[int, dict] = {}
ORDERS: dict[int, dict] = {}
PAYMENTS: dict[int, dict] = {}
ADDRESSES: dict[int, list] = {}
TICKETS: dict[int, list] = {}
CARTS: dict[int, list] = {}
NOTES: dict[int, list] = {}
_seed()


def caller_id(request: Request) -> int:
    header = request.headers.get("authorization", "")
    token = header.split(" ", 1)[1].strip() if header.lower().startswith("bearer ") else ""
    if token.startswith("user-") and token[5:].isdigit():
        user_id = int(token[5:])
        if user_id in USERS:
            return user_id
    raise HTTPException(status_code=401, detail="bearer token user-<id> required")


def _order_lookup(order_id: int) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="order not found")
    return order


@app.get("/api/users/{user_id}")
async def get_user(user_id: int, request: Request) -> dict:
    caller_id(request)
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="user not found")
    return USERS[user_id]


@app.patch("/api/users/{user_id}")
async def update_user(user_id: int, request: Request) -> dict:
    caller_id(request)
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="user not found")
    body = await request.json()
    user = USERS[user_id]
    for key, value in body.items():
        if key != "id":
            user[key] = value
    return user


@app.get("/api/orders")
async def list_orders(request: Request, page: int = 1, status: str | None = None) -> dict:
    owner = caller_id(request)
    orders = [order for order in ORDERS.values() if order["user_id"] == owner]
    if status:
        orders = [order for order in orders if order["status"] == status]
    return {"page": page, "status": status, "orders": orders}


@app.get("/api/orders/{order_id}")
async def get_order(order_id: int, request: Request) -> dict:
    caller_id(request)
    return _order_lookup(order_id)


@app.post("/api/orders", status_code=201)
async def create_order(request: Request) -> dict:
    owner = caller_id(request)
    body = await request.json()
    order_id = max(ORDERS) + 1
    order = {
        "id": order_id,
        "user_id": owner,
        "item": body.get("item", "notebook"),
        "qty": body.get("qty", 1),
        "status": "open",
        "note": body.get("note", ""),
    }
    ORDERS[order_id] = order
    return order


@app.get("/api/users/{user_id}/orders")
async def list_user_orders(user_id: int, request: Request) -> dict:
    caller_id(request)
    if user_id not in USERS:
        raise HTTPException(status_code=404, detail="user not found")
    orders = [order for order in ORDERS.values() if order["user_id"] == user_id]
    return {"user_id": user_id, "orders": orders}


@app.post("/api/search")
async def search_catalog(request: Request) -> dict:
    body = await request.json()
    query = str(body.get("query", ""))
    hits = [item for item in CATALOG if query.lower() in item["item"]] or CATALOG[:1]
    return {"query": query, "hits": hits}


@app.post("/api/comments", status_code=201)
async def add_comment(request: Request) -> dict:
    caller_id(request)
    body = await request.json()
    order = _order_lookup(int(body.get("order_id", 0)))
    order["note"] = str(body.get("text", ""))
    return {"order_id": order["id"], "text": order["note"]}


@app.get("/api/payments")
async def list_payments(request: Request, page: int = 1) -> dict:
    owner = caller_id(request)
    payments = [payment for payment in PAYMENTS.values() if payment["user_id"] == owner]
    return {"page": page, "payments": payments}


@app.post("/api/payments", status_code=201)
async def create_payment(request: Request) -> dict:
    owner = caller_id(request)
    body = await request.json()
    order = _order_lookup(int(body.get("order_id", 0)))
    payment_id = max(PAYMENTS) + 1
    payment = {
        "id": payment_id,
        "user_id": owner,
        "order_id": order["id"],
        "amount": body.get("amount", 12.5),
        "currency": body.get("currency", "USD"),
    }
    PAYMENTS[payment_id] = payment
    return payment


@app.get("/api/payments/{payment_id}")
async def get_payment(payment_id: int, request: Request) -> dict:
    caller_id(request)
    payment = PAYMENTS.get(payment_id)
    if payment is None:
        raise HTTPException(status_code=404, detail="payment not found")
    return payment


@app.get("/api/addresses")
async def list_addresses(request: Request) -> dict:
    owner = caller_id(request)
    return {"addresses": ADDRESSES.get(owner, [])}


@app.post("/api/addresses", status_code=201)
async def create_address(request: Request) -> dict:
    owner = caller_id(request)
    body = await request.json()
    entry = {
        "id": owner * 1000 + len(ADDRESSES[owner]) + 1,
        "user_id": owner,
        "line1": body.get("line1", ""),
        "city": body.get("city", ""),
        "postal_code": body.get("postal_code", ""),
    }
    ADDRESSES[owner].append(entry)
    return entry


@app.get("/api/tickets")
async def list_tickets(request: Request) -> dict:
    owner = caller_id(request)
    return {"tickets": TICKETS.get(owner, [])}


@app.post("/api/tickets", status_code=201)
async def create_ticket(request: Request) -> dict:
    owner = caller_id(request)
    body = await request.json()
    entry = {
        "id": owner * 1000 + len(TICKETS[owner]) + 1,
        "user_id": owner,
        "subject": body.get("subject", ""),
        "body": body.get("body", ""),
    }
    TICKETS[owner].append(entry)
    return entry


@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: int, request: Request) -> dict:
    caller_id(request)
    for tickets in TICKETS.values():
        for ticket in tickets:
            if ticket["id"] == ticket_id:
                return ticket
    raise HTTPException(status_code=404, detail="ticket not found")


@app.post("/api/reviews", status_code=201)
async def create_review(request: Request) -> dict:
    caller_id(request)
    body = await request.json()
    order = _order_lookup(int(body.get("order_id", 0)))
    return {"order_id": order["id"], "rating": body.get("rating", 5), "title": body.get("title", "")}


@app.get("/api/cart")
async def get_cart(request: Request) -> dict:
    owner = caller_id(request)
    return {"items": CARTS.get(owner, [])}


@app.post("/api/cart", status_code=201)
async def add_cart_item(request: Request) -> dict:
    owner = caller_id(request)
    body = await request.json()
    item = {"item": body.get("item", "notebook"), "qty": body.get("qty", 1)}
    CARTS[owner].append(item)
    return item


@app.get("/api/notifications")
async def list_notifications(request: Request, unread: bool = False) -> dict:
    owner = caller_id(request)
    items = NOTES.get(owner, [])
    if unread:
        items = [item for item in items if item.get("unread")]
    return {"unread": unread, "items": items}


DOCUMENTS = {
    "catalog.pdf": "public catalog document",
    "faq.txt": "Frequently asked questions about notebooks",
    "shipping.txt": "Standard shipping is five days",
    "shipping-v2.txt": "Express shipping is two days",
}


@app.get("/api/files")
async def download_file(name: str) -> dict[str, str]:
    if not name:
        raise HTTPException(status_code=404, detail="file not found")
    if name in DOCUMENTS:
        return {"name": name, "content": DOCUMENTS[name]}
    return {"name": name, "content": "EVAL_INTERNAL_DOCUMENT"}
