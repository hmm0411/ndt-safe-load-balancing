from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import datetime
from typing import Any

from src.schemas.snapshot import (
    OwnershipState,
)
from src.schemas.telemetry import (
    SwitchTelemetry,
)

# ControllerTelemetry is a subset of SwitchTelemetry, but we don't want to import SwitchTelemetry into the controller module, so we define a separate class here.
# = physical/raw observed state

# ControllerTwinState is a subset of ControllerTelemetry, but we don't want to import ControllerTelemetry into the twin module, so we define a separate class here.
# = derived/processed digital-twin state

@dataclass(frozen=True)
class ControllerTwinState:
    controller_id: str

    processed_packet_in_rate: float

    safe_capacity_pps: float

    utilization: float

    collection_latency_ms: float

    flow_mod_rate: float

    process_cpu_percent: float

    process_memory_rss_mb: float

    response_p95_ms: float

    managed_switch_count: int

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwinQuality:
    age_of_twin_ms: float

    twinning_rate: float

    completeness_ratio: float

    synchronization_jitter_ms: float

    consistent: bool

    valid: bool

    # Soft quality in [0, 1] (configs/quality.yaml). It never replaces
    # `valid` (the Hard Gate); None when no quality config is loaded.
    consistency_score: float = 1.0

    quality_score: float | None = None

    quality_components: dict[str, float] = field(default_factory=dict)

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwinState:
    snapshot_id: str

    created_at: datetime

    topology_version: int

    ownership_version: int

    controllers: list[
        ControllerTwinState
    ]

    switches: list[
        SwitchTelemetry
    ]

    ownership: list[
        OwnershipState
    ]

    quality: TwinQuality

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "snapshot_id":
                self.snapshot_id,

            "created_at":
                self.created_at.isoformat(),

            "topology_version":
                self.topology_version,

            "ownership_version":
                self.ownership_version,

            "controllers": [
                item.to_dict()
                for item
                in self.controllers
            ],

            "switches": [
                item.to_dict()
                for item
                in self.switches
            ],

            "ownership": [
                item.to_dict()
                for item
                in self.ownership
            ],

            "quality":
                self.quality.to_dict(),
        }