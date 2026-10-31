"""Proxy-only ownership hints: caller principal vs user-scoped ids and owner fields.

This is not application authorization. It only compares identifiers the request,
token, or JSON body already expose. Opaque sessions and resources with no owner
field cannot be decided here.
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any
from urllib.parse import parse_qsl

# Path/query/body names that refer to the caller, not to an order or ticket id.
_USER_SCOPED = frozenset({"userid", "accountid", "customerid"})
# JSON keys that name the owner of a record. `id` is omitted: it is often the object id.
_OWNER_KEYS = ("user_id", "owner_id", "account_id", "customer_id")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_user_scoped_name(name: str) -> bool:
    return _norm(name) in _USER_SCOPED


def _same_id(principal: str, value: object) -> bool:
    text = str(value).strip()
    if not text:
        return True
    if text == principal:
        return True
    if text == f"user-{principal}" or principal == f"user-{text}":
        return True
    return False


def extract_principal(authorization: str | None) -> str | None:
    """Return a caller id from a Bearer token. The raw secret is never returned.

    `Bearer user-51` yields `51`. A JWT payload is decoded without verifying a
    signature; `sub`, `user_id`, or `uid` is used as an identity hint only.
    """
    if not authorization:
        return None
    raw = authorization.strip()
    token = raw.split(" ", 1)[1].strip() if raw.lower().startswith("bearer ") else raw
    if not token:
        return None
    if token.lower().startswith("user-") and len(token) > 5:
        rest = token[5:]
        if rest.isdigit() or rest.isalnum():
            return rest
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("sub", "user_id", "uid"):
        value = payload.get(key)
        if value in (None, ""):
            continue
        text = str(value)
        if text.lower().startswith("user-") and len(text) > 5:
            return text[5:]
        return text
    return None


def user_scoped_mismatch(
    principal: str | None,
    path_params: dict[str, str] | None = None,
    query: str = "",
    body_preview: str = "",
) -> bool:
    """True when a user-scoped identifier on the request is not the caller."""
    if not principal:
        return False
    pairs: list[tuple[str, object]] = []
    for name, value in (path_params or {}).items():
        pairs.append((name, value))
    for name, value in parse_qsl(query, keep_blank_values=True):
        pairs.append((name, value))
    if body_preview.lstrip().startswith("{"):
        try:
            payload = json.loads(body_preview)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            pairs.extend(payload.items())
    for name, value in pairs:
        if not is_user_scoped_name(str(name)) or value in (None, ""):
            continue
        if not _same_id(principal, value):
            return True
    return False


def owner_keys(declared: str | None = None) -> tuple[str, ...]:
    if declared:
        return (declared,)
    return _OWNER_KEYS


def json_owner_mismatch(raw: bytes | str, principal: str, declared_field: str | None = None) -> bool:
    """True when a JSON object (or nested list) names an owner other than the caller."""
    if not principal:
        return False
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace")
    else:
        text = raw
    stripped = text.lstrip()
    if not stripped.startswith("{") and not stripped.startswith("["):
        return False
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    keys = owner_keys(declared_field)

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key in keys:
                if key in value and value[key] not in (None, ""):
                    if not _same_id(principal, value[key]):
                        return True
            return any(walk(item) for item in value.values())
        if isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(payload)
