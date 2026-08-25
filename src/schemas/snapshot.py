from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from src.schemas.telemetry import ControllerTelemetry, SwitchTelemetry
from __future__ import annotations

@dataclass(frozen=True)
class OwnershipState:
    switch_id: str
    owner_controller_id: str
    source_role: str
    target_role: str
    generation_id: int
    ownership_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class SnapshotQuality:
    fresh: bool
    complete: bool
    consistent: bool
    valid: bool
    missing_fields: list[str] = field(default_factory=list)
    stale_sources: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    max_data_age_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class NetworkSnapshot:
    snapshot_id: str
    created_at: datetime
    topology_version: int
    ownership_version: int
    controllers: list[ControllerTelemetry]
    switches: list[SwitchTelemetry]
    ownership: list[OwnershipState]
    quality: SnapshotQuality

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
            "topology_version": self.topology_version,
            "ownership_version": self.ownership_version,
            "controllers": [x.to_dict() for x in self.controllers],
            "switches": [x.to_dict() for x in self.switches],
            "ownership": [x.to_dict() for x in self.ownership],
            "quality": self.quality.to_dict(),
        }
