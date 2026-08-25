from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkloadSample:
    run_id: str
    observed_at: datetime

    source_host: str
    source_ip: str

    target_host: str
    target_ip: str
    target_port: int

    protocol: str
    pattern: str

    target_new_flow_rate: float
    emitted_new_flow_rate: float

    interval_seconds: float

    attempted_flows: int
    emitted_flows: int
    send_errors: int

    late_events: int
    max_schedule_lag_ms: float

    cumulative_attempted_flows: int
    cumulative_emitted_flows: int
    cumulative_send_errors: int

    first_source_port: int | None = None
    last_source_port: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        return data