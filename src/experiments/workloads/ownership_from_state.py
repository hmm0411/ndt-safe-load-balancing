"""Build an OwnershipTimeline from /api/v1/state data, without guessing."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from src.experiments.workloads.setup_probe import OwnershipTimeline

ROLLED_BACK = {"FAILED", "ROLLING_BACK", "RESTORED"}


def _epoch(text: str) -> float:
    return datetime.fromisoformat(text).timestamp()


def _event_time(tx: Mapping[str, Any], state: str) -> float | None:
    for event in tx.get("history", []):
        if event["state"] == state:
            return _epoch(event["at"])
    return None


def build_timeline(
    ownership: Mapping[str, str],
    transactions: Iterable[Mapping[str, Any]],
    since: float,
) -> OwnershipTimeline:
    """ownership is the mapping at `since`. A committed migration changes the
    owner at its VERIFYING event, when the target is already MASTER."""
    timeline = OwnershipTimeline()
    for switch_id, controller_id in ownership.items():
        timeline.record(since, switch_id, controller_id)
    changes: list[tuple[float, str, str]] = []
    for tx in transactions:
        if tx.get("state") != "COMMITTED":
            continue
        at = _event_time(tx, "VERIFYING")
        if at is None:
            raise ValueError(f"{tx['transaction_id']} is COMMITTED without VERIFYING")
        if at >= since:
            changes.append((at, str(tx["switch_id"]), str(tx["target_controller"])))
    for at, switch_id, controller_id in sorted(changes):
        timeline.record(at, switch_id, controller_id)
    return timeline


def ambiguous_windows(
    transactions: Iterable[Mapping[str, Any]], since: float
) -> list[tuple[str, float, float]]:
    """Role-switch attempts that were rolled back: owner unclear in the window."""
    windows: list[tuple[str, float, float]] = []
    for tx in transactions:
        if tx.get("state") not in ROLLED_BACK:
            continue
        start = _event_time(tx, "ROLE_SWITCHING")
        if start is None:
            continue
        end = _epoch(tx["history"][-1]["at"])
        if end >= since:
            windows.append((str(tx["switch_id"]), start, end))
    return sorted(windows, key=lambda w: (w[1], w[0]))
