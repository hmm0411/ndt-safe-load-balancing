"""Latest per-controller flow-setup stats, merged into a snapshot while fresh."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Lock

from src.schemas.snapshot import NetworkSnapshot


@dataclass(frozen=True)
class SetupStats:
    p95_ms: float
    samples: int
    timeouts: int
    window_end: datetime


class SetupProbeStore:
    """A controller without fresh stats keeps None; a stale value is never used."""

    def __init__(self, max_age_ms: float) -> None:
        if max_age_ms <= 0:
            raise ValueError("max_age_ms must be > 0")
        self._max_age_ms = max_age_ms
        self._stats: dict[str, SetupStats] = {}
        self._lock = Lock()

    def update(self, stats: Mapping[str, SetupStats]) -> None:
        """Replace everything: a controller missing from `stats` has no stats."""
        for item in stats.values():
            if item.window_end.tzinfo is None:
                raise ValueError("window_end must carry a timezone")
        with self._lock:
            self._stats = dict(stats)

    def overlay(self, snapshot: NetworkSnapshot) -> NetworkSnapshot:
        with self._lock:
            stats = dict(self._stats)
        controllers = []
        for controller in snapshot.controllers:
            item = stats.get(controller.controller_id)
            if item is not None:
                age_ms = (snapshot.created_at - item.window_end).total_seconds() * 1000.0
                if age_ms <= self._max_age_ms:
                    controller = replace(
                        controller,
                        setup_p95_ms=item.p95_ms,
                        setup_samples=item.samples,
                        setup_timeouts=item.timeouts,
                    )
            controllers.append(controller)
        return replace(snapshot, controllers=controllers)
