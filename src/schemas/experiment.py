from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ExperimentMetadata:
    run_id: str
    experiment_type: str

    topology: str
    workload_pattern: str
    protocol: str

    target_controller: str

    source_host: str
    target_host: str

    target_new_flow_rate: float

    duration_seconds: float
    sample_interval_seconds: float

    seed: int
    git_commit: str

    started_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        return data