"""Flow-setup probe: samples, ownership at send time, per-controller window."""
from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class SetupSample:
    """One probe flow through one switch.

    controller_id is the owner of the switch when the probe was sent.
    latency_ms is None when the probe timed out.
    """

    switch_id: str
    controller_id: str
    sent_at: float
    latency_ms: float | None

    @property
    def timed_out(self) -> bool:
        return self.latency_ms is None


@dataclass(frozen=True)
class SetupWindowStats:
    p95_ms: float
    samples: int
    timeouts: int


class OwnershipTimeline:
    """Owner of each switch over time; a change applies from its time on."""

    def __init__(self) -> None:
        self._changes: dict[str, list[tuple[float, str]]] = {}

    def record(self, at: float, switch_id: str, controller_id: str) -> None:
        changes = self._changes.setdefault(switch_id, [])
        if changes and at < changes[-1][0]:
            raise ValueError("ownership changes must be recorded in time order")
        changes.append((at, controller_id))

    def owner_at(self, switch_id: str, at: float) -> str | None:
        changes = self._changes.get(switch_id, [])
        index = bisect.bisect_right([when for when, _ in changes], at)
        return changes[index - 1][1] if index else None


def make_sample(
    timeline: OwnershipTimeline,
    switch_id: str,
    sent_at: float,
    latency_ms: float | None,
) -> SetupSample:
    owner = timeline.owner_at(switch_id, sent_at)
    if owner is None:
        raise ValueError(f"owner of {switch_id} unknown at {sent_at}")
    return SetupSample(switch_id, owner, sent_at, latency_ms)


def summarize_window(
    samples: Iterable[SetupSample],
    controller_id: str,
    start: float,
    end: float,
    timeout_ms: float,
) -> SetupWindowStats | None:
    """P95 of one controller over [start, end); a timeout counts as timeout_ms.

    Returns None, never 0.0, when the window has no sample for the controller.
    """
    if timeout_ms <= 0:
        raise ValueError("timeout_ms must be > 0")
    own = [
        s for s in samples
        if s.controller_id == controller_id and start <= s.sent_at < end
    ]
    if not own:
        return None
    values = sorted(
        timeout_ms if s.latency_ms is None else s.latency_ms for s in own
    )
    rank = max(1, math.ceil(0.95 * len(values)))
    return SetupWindowStats(
        p95_ms=values[rank - 1],
        samples=len(own),
        timeouts=sum(1 for s in own if s.timed_out),
    )
