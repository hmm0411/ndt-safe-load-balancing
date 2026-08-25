from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4
from src.common.enums import TransactionState
from __future__ import annotations

@dataclass(frozen=True)
class TransactionEvent:
    state: TransactionState
    at: datetime
    detail: Optional[str] = None

    def to_dict(self):
        return {
            "state": self.state.value,
            "at": self.at.isoformat(),
            "detail": self.detail,
        }

@dataclass
class MigrationTransaction:
    switch_id: str
    source_controller: str
    target_controller: str
    transaction_id: str = field(default_factory=lambda: f"tx-{uuid4().hex[:12]}")
    generation_id: int | None = None
    state: TransactionState = TransactionState.PREPARING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_reason: str | None = None
    history: list[TransactionEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "switch_id": self.switch_id,
            "source_controller": self.source_controller,
            "target_controller": self.target_controller,
            "generation_id": self.generation_id,
            "state": self.state.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "failure_reason": self.failure_reason,
            "history": [x.to_dict() for x in self.history],
        }
