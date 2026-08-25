from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from __future__ import annotations


@dataclass(frozen=True)
class ControllerTelemetry:
    controller_id: str
    observed_at: datetime
    ingested_at: datetime
    packet_in_total: int
    packet_in_rate: float
    processed_packet_in_total: int
    processed_packet_in_rate: float
    flow_mod_total: int
    flow_mod_rate: float
    process_cpu_percent: float
    process_memory_rss_mb: float
    response_mean_ms: float
    response_p95_ms: float
    managed_switch_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        data["ingested_at"] = self.ingested_at.isoformat()
        return data


@dataclass(frozen=True)
class SwitchTelemetry:
    switch_id: str
    controller_id: str
    observed_at: datetime
    packet_in_total: int
    packet_in_rate: float
    processed_packet_in_total: int
    processed_packet_in_rate: float
    flow_mod_total: int
    flow_mod_rate: float
    control_load_share: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        return data
