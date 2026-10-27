"""Shared data models for the fast path, the agent, and the dashboard."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Tier(str, Enum):
    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    HARD_BLOCK = "hard_block"


class Action(str, Enum):
    FORWARDED = "forwarded"
    BLOCKED = "blocked"
    ALERT = "alert"
    BLOCKLISTED = "blocklisted"
    IGNORED = "ignored"


class RequestFeatures(BaseModel):
    """Numeric and textual features extracted from a single HTTP request."""

    method: str
    path: str
    query: str = ""
    body_length: int = 0
    query_length: int = 0
    special_char_ratio: float = 0.0
    traversal_hits: int = 0
    injection_hits: int = 0
    numeric_id_segments: int = 0
    window_rate: int = 0
    route_known: bool | None = None
    matched_route: str | None = None
    undeclared_fields: list[str] = Field(default_factory=list)
    privileged_fields: list[str] = Field(default_factory=list)


class FastVerdict(BaseModel):
    """Result of the synchronous in-process filter."""

    tier: Tier
    score: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    threat_hint: str | None = None


class AgentVerdict(BaseModel):
    """Structured verdict returned by the Meta model."""

    is_malicious: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    threat_type: str = "unknown"
    reasoning: str = ""


class SuspiciousRequest(BaseModel):
    """Snapshot of a request handed to the async agent."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    client_ip: str
    method: str
    path: str
    query: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    body_preview: str = ""
    features: RequestFeatures
    fast_verdict: FastVerdict
    received_at: float = Field(default_factory=time.time)


class SecurityEvent(BaseModel):
    """Entry stored in Redis and displayed on the dashboard."""

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = Field(default_factory=time.time)
    client_ip: str
    method: str
    path: str
    tier: Tier
    fast_score: float
    action: Action
    threat_type: str | None = None
    confidence_score: float | None = None
    reasoning: str = ""
    source: str = "fast_filter"
    extra: dict[str, Any] = Field(default_factory=dict)
