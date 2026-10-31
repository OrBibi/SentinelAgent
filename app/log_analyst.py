"""Scan the Redis request log for multi-step campaigns.

This is not a second model. It reads events the proxy already writes and looks
across a window instead of a single request.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass

from app.ml_filter import WRITE_METHODS
from app.models import Action, SecurityEvent

_OBJECT_ID = re.compile(r"/(\d+)(?=/|$|\?)")
# Alerts that are usually extra-field noise are left out so shoppers are not watched.
_PRIOR_THREATS = frozenset({"bola", "injection", "path_traversal", "xss", "ssrf", "schema_violation"})
_DENY_ACTIONS = frozenset({Action.BLOCKED, Action.BLOCKLISTED})


@dataclass(slots=True, frozen=True)
class CampaignFinding:
    actor: str
    ips: tuple[str, ...]
    signals: tuple[str, ...]
    event_count: int
    reason: str


def actor_of(event: SecurityEvent) -> str:
    return (event.principal or "").strip() or event.client_ip


def _object_ids(path: str) -> list[str]:
    return _OBJECT_ID.findall(path)


def scan_events(
    events: list[SecurityEvent],
    *,
    now: float | None = None,
    window_seconds: float = 600.0,
    min_ips: int = 2,
    min_denies: int = 3,
    min_object_ids: int = 5,
) -> list[CampaignFinding]:
    """Return campaign findings for actors in the recent request log."""
    if now is None:
        now = time.time()
    rows = [
        event
        for event in events
        if event.source != "log_agent" and (now - event.timestamp) <= window_seconds
    ]
    by_actor: dict[str, list[SecurityEvent]] = defaultdict(list)
    for event in rows:
        by_actor[actor_of(event)].append(event)

    findings: list[CampaignFinding] = []
    for actor, group in by_actor.items():
        finding = _inspect_actor(actor, group, min_ips=min_ips, min_denies=min_denies, min_object_ids=min_object_ids)
        if finding is not None:
            findings.append(finding)
    return findings


def _inspect_actor(
    actor: str,
    group: list[SecurityEvent],
    *,
    min_ips: int,
    min_denies: int,
    min_object_ids: int,
) -> CampaignFinding | None:
    ordered = sorted(group, key=lambda event: event.timestamp)
    ips = tuple(dict.fromkeys(event.client_ip for event in ordered if event.client_ip))
    object_ids = {oid for event in ordered for oid in _object_ids(event.path)}
    signals: list[str] = []

    if len(ips) >= min_ips:
        signals.append("multi_ip")

    denies = 0
    saw_prior = False
    deny_then_ok = False
    alert_then_write = False
    for event in ordered:
        denied = event.action in _DENY_ACTIONS or event.response_status == 403
        if denied:
            denies += 1
        if event.response_status == 200 and denies >= min_denies:
            deny_then_ok = True
        threat = event.threat_type or ""
        if event.action in _DENY_ACTIONS or (event.action is Action.ALERT and threat in _PRIOR_THREATS):
            saw_prior = True
        if saw_prior and event.method.upper() in WRITE_METHODS and not denied:
            alert_then_write = True

    if deny_then_ok:
        signals.append("deny_then_ok")
    if alert_then_write:
        signals.append("alert_then_write")
    if len(object_ids) >= min_object_ids and denies >= 2:
        signals.append("object_enum")

    if not signals:
        return None
    reason = f"request-log campaign for {actor}: {', '.join(signals)}"
    return CampaignFinding(
        actor=actor,
        ips=ips,
        signals=tuple(signals),
        event_count=len(ordered),
        reason=reason,
    )
