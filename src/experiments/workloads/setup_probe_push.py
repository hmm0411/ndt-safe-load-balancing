"""Incremental reading of qos_s<i>.jsonl and per-controller window stats."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiments.workloads.setup_probe import (
    OwnershipTimeline,
    SetupSample,
    SetupWindowStats,
    make_sample,
    summarize_window,
)


@dataclass(frozen=True)
class RawSample:
    """A probe result before its owner is known."""

    switch_id: str
    sent_at: float
    latency_ms: float | None


def raw_from_row(row: Mapping[str, Any], switch_id: str) -> RawSample:
    observed = datetime.fromisoformat(str(row["observed_at"]))
    if observed.tzinfo is None:
        raise ValueError("observed_at must carry a timezone")
    latency = row.get("flow_setup_latency_ms")
    value: float | None = None
    if row.get("success") is True and latency is not None:
        value = float(latency)
    return RawSample(switch_id, observed.timestamp(), value)


class SampleTail:
    """Reads only the complete new lines of one file on each call."""

    def __init__(self, path: Path, switch_id: str) -> None:
        self.path = path
        self.switch_id = switch_id
        self._offset = 0

    def read_new(self) -> list[RawSample]:
        if not self.path.exists():
            return []
        with self.path.open("rb") as handle:
            handle.seek(self._offset)
            data = handle.read()
        end = data.rfind(b"\n") + 1
        self._offset += end
        lines = data[:end].decode("utf-8").splitlines()
        return [
            raw_from_row(json.loads(line), self.switch_id)
            for line in lines
            if line.strip()
        ]


def to_samples(
    raws: Iterable[RawSample], timeline: OwnershipTimeline
) -> tuple[list[SetupSample], int]:
    """Attach the owner at send time; samples sent before it is known are counted."""
    samples: list[SetupSample] = []
    unknown = 0
    for raw in raws:
        if timeline.owner_at(raw.switch_id, raw.sent_at) is None:
            unknown += 1
            continue
        samples.append(
            make_sample(timeline, raw.switch_id, raw.sent_at, raw.latency_ms)
        )
    return samples, unknown


def window_stats(
    samples: Sequence[SetupSample],
    ambiguous: Iterable[tuple[str, float, float]],
    start: float,
    end: float,
    timeout_ms: float,
    controller_ids: Iterable[str],
) -> tuple[dict[str, SetupWindowStats], int]:
    """Stats per controller; a sample inside a rolled-back window is left out."""
    windows = list(ambiguous)
    kept = [
        s
        for s in samples
        if not any(w[0] == s.switch_id and w[1] <= s.sent_at <= w[2] for w in windows)
    ]
    result: dict[str, SetupWindowStats] = {}
    for controller_id in controller_ids:
        stats = summarize_window(kept, controller_id, start, end, timeout_ms)
        if stats is not None:
            result[controller_id] = stats
    return result, len(samples) - len(kept)


def payload(
    stats: Mapping[str, SetupWindowStats], window_end: datetime
) -> dict[str, Any]:
    return {
        "controllers": {
            cid: {
                "p95_ms": item.p95_ms,
                "samples": item.samples,
                "timeouts": item.timeouts,
                "window_end": window_end.isoformat(),
            }
            for cid, item in stats.items()
        }
    }
