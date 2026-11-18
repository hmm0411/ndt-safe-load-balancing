"""One push cycle: read new samples, attach owners, post per-controller stats."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.workloads.ownership_from_state import (
    ambiguous_windows,
    build_timeline,
)
from src.experiments.workloads.setup_probe_push import (
    RawSample,
    SampleTail,
    payload,
    to_samples,
    window_stats,
)


class PushLoop:
    def __init__(
        self,
        probe_dir: Path,
        initial_ownership: Mapping[str, str],
        since: float,
        window_seconds: float,
        timeout_seconds: float,
    ) -> None:
        if window_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError("window_seconds and timeout_seconds must be > 0")
        self.probe_dir = probe_dir
        self.initial = dict(initial_ownership)
        self.since = since
        self.window = window_seconds
        self.timeout_ms = timeout_seconds * 1000.0
        self.controllers = sorted(set(self.initial.values()))
        self._tails: dict[Path, SampleTail] = {}
        self._raws: list[RawSample] = []

    def cycle(
        self,
        transactions: Sequence[Mapping[str, Any]],
        post: Callable[[dict[str, Any]], object],
        now: float,
    ) -> dict[str, Any]:
        for path in sorted(self.probe_dir.glob("qos_s*.jsonl")):
            if path not in self._tails:
                switch_id = path.stem.removeprefix("qos_")
                self._tails[path] = SampleTail(path, switch_id)
        for tail in self._tails.values():
            self._raws.extend(tail.read_new())
        start = now - self.window
        self._raws = [r for r in self._raws if r.sent_at >= start]
        timeline = build_timeline(self.initial, transactions, self.since)
        samples, unknown = to_samples(self._raws, timeline)
        stats, excluded = window_stats(
            samples,
            ambiguous_windows(transactions, self.since),
            start,
            now,
            self.timeout_ms,
            self.controllers,
        )
        post(payload(stats, datetime.fromtimestamp(now, timezone.utc)))
        return {
            "stats": {c: [s.p95_ms, s.samples, s.timeouts] for c, s in stats.items()},
            "unknown_owner": unknown,
            "excluded_rolled_back": excluded,
        }
