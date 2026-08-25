from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkloadSample:
    # identifiers
    run_id: str
    observed_at: datetime

    # traffic endpoints
    source_host: str
    source_ip: str

    target_host: str
    target_ip: str
    target_port: int

    # workload definition
    protocol: str
    pattern: str

    target_new_flow_rate: float

    # ground truth metrics of generated traffic
    emitted_new_flow_rate: float

    interval_seconds: float

    attempted_flows: int
    emitted_flows: int
    send_errors: int

    # quality metrics of the workload generator
    late_events: int
    max_schedule_lag_ms: float

    # cumulative metrics of the workload generator
    cumulative_attempted_flows: int
    cumulative_emitted_flows: int
    cumulative_send_errors: int

    # port range of the workload generator (if applicable)
    first_source_port: int | None = None
    last_source_port: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        return data