"""Identity of one decision cycle (Week 13, Day 5).

A cycle is pinned to ONE snapshot: forecast, candidates, outcome predictions, Gate
and ranking all read the state named here. The decision_id is a pure function of the
mode and the snapshot, so asking twice for the same snapshot is the same decision
(idempotency) and can never execute a second migration.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

DRY_RUN = "dry_run"
EXECUTE = "execute"
MODES = (DRY_RUN, EXECUTE)
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def make_decision_id(mode: str, snapshot_id: str) -> str:
    """dec-<mode>-<snapshot id>, safe as a directory name."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must be a non-empty string")
    return f"dec-{mode}-{_UNSAFE.sub('_', snapshot_id.strip())}"


@dataclass(frozen=True)
class DecisionContext:
    decision_id: str
    mode: str
    scenario: str
    snapshot_id: str
    snapshot_created_at: datetime
    decided_at: datetime
    forecast_snapshot_id: str | None = None
    forecast_model_version: str | None = None

    def __post_init__(self) -> None:
        if self.decision_id != make_decision_id(self.mode, self.snapshot_id):
            raise ValueError("decision_id does not match mode and snapshot_id")
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("scenario must be a non-empty string")
        for name in ("snapshot_created_at", "decided_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ValueError(f"{name} must be a timezone-aware datetime")

    @classmethod
    def pin(cls, mode: str, scenario: str, snapshot_id: str, snapshot_created_at: datetime,
            decided_at: datetime) -> DecisionContext:
        return cls(make_decision_id(mode, snapshot_id), mode, scenario, snapshot_id,
                   snapshot_created_at, decided_at)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["snapshot_created_at"] = self.snapshot_created_at.isoformat()
        data["decided_at"] = self.decided_at.isoformat()
        return data
