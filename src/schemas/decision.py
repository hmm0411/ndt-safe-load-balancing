from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class ReasonCode(str, Enum):
    """Machine-readable reason a candidate was rejected."""

    INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
    QUALITY_BELOW_POLICY = "QUALITY_BELOW_POLICY"
    SOURCE_NOT_AT_RISK = "SOURCE_NOT_AT_RISK"
    DESTINATION_DISCONNECTED = "DESTINATION_DISCONNECTED"
    DESTINATION_ALREADY_AT_RISK = "DESTINATION_ALREADY_AT_RISK"
    OWNER_MISMATCH = "OWNER_MISMATCH"
    ROLE_STATE_INVALID = "ROLE_STATE_INVALID"
    ACTIVE_TRANSACTION_CONFLICT = "ACTIVE_TRANSACTION_CONFLICT"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    MISSING_CAPACITY = "MISSING_CAPACITY"
    OUTCOME_PREDICTOR_NOT_READY = "OUTCOME_PREDICTOR_NOT_READY"
    DST_UTILIZATION_UNSAFE = "DST_UTILIZATION_UNSAFE"
    DST_RESPONSE_UNSAFE = "DST_RESPONSE_UNSAFE"
    NO_SOURCE_RELIEF = "NO_SOURCE_RELIEF"
    INSUFFICIENT_IMPROVEMENT = "INSUFFICIENT_IMPROVEMENT"


STRUCTURAL_REASONS: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.INVALID_SNAPSHOT,
        ReasonCode.SOURCE_NOT_AT_RISK,
        ReasonCode.DESTINATION_DISCONNECTED,
        ReasonCode.OWNER_MISMATCH,
        ReasonCode.ROLE_STATE_INVALID,
        ReasonCode.ACTIVE_TRANSACTION_CONFLICT,
        ReasonCode.COOLDOWN_ACTIVE,
        ReasonCode.MISSING_CAPACITY,
    }
)


@dataclass(frozen=True)
class MigrationCandidate:
    """One (switch, source -> target) migration option for one snapshot."""

    candidate_id: str
    snapshot_id: str
    switch_id: str
    source_controller: str
    target_controller: str
    source_utilization: float
    target_utilization: float
    switch_control_load_share: float
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        return data


@dataclass(frozen=True)
class EvaluatedCandidate:
    """A candidate plus the Outcome Predictor's point predictions."""

    candidate: MigrationCandidate
    predicted_src_utilization: float
    predicted_dst_utilization: float
    predicted_src_response_p95_ms: float
    predicted_dst_response_p95_ms: float
    predicted_imbalance: float
    utilization_margin: float
    response_margin_ms: float
    outcome_model_version: str
    outcome_inference_ms: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["candidate"] = self.candidate.to_dict()
        return data


@dataclass(frozen=True)
class SafetyDecision:
    """Result of the Safety Gate for one evaluated candidate."""

    candidate_id: str
    safe: bool
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    predicted_dst_utilization: float
    utilization_margin: float
    dst_upper_bound: float
    predicted_dst_response_p95_ms: float
    response_margin_ms: float
    response_upper_bound_ms: float | None
    predicted_src_relief: float
    quality_score: float
    consistency_score: float
    policy_version: str
    outcome_model_version: str
    decision_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["passed_checks"] = list(self.passed_checks)
        data["failed_checks"] = list(self.failed_checks)
        return data
