"""Turn qos_probe rows into SetupSample, with the owner at send time."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiments.workloads.setup_probe import (
    OwnershipTimeline,
    SetupSample,
    make_sample,
)


def row_to_sample(
    row: dict[str, Any], timeline: OwnershipTimeline, switch_id: str
) -> SetupSample:
    """observed_at is taken just before the send; a failed row is a timeout."""
    observed = datetime.fromisoformat(str(row["observed_at"]))
    if observed.tzinfo is None:
        raise ValueError("observed_at must carry a timezone")
    latency = row.get("flow_setup_latency_ms")
    value: float | None = None
    if row.get("success") is True and latency is not None:
        value = float(latency)
    return make_sample(timeline, switch_id, observed.timestamp(), value)


def read_samples(
    path: Path, timeline: OwnershipTimeline, switch_id: str
) -> list[SetupSample]:
    samples: list[SetupSample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                samples.append(row_to_sample(json.loads(line), timeline, switch_id))
    return samples
