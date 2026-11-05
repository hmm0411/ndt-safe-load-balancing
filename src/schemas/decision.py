"""Week 13 decision contracts: candidate -> evaluated candidate -> safety decision.

A candidate a = (switch, source, target) exists only while the source has an overload
risk. The Outcome Predictor turns it into an EvaluatedCandidate and the Safety Gate
returns one SafetyDecision per candidate. Utilization is never clamped to [0, 1].
Shared schema: a change needs both owners' review (CLAUDE.md section 2).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

# Reason codes are counted in reports: stable strings, never renamed.
INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
QUALITY_BELOW_POLICY = "QUALITY_BELOW_POLICY"
SOURCE_NOT_AT_RISK = "SOURCE_NOT_AT_RISK"
OWNER_MISMATCH = "OWNER_MISMATCH"
ROLE_STATE_INVALID = "ROLE_STATE_INVALID"
SWITCH_DISCONNECTED = "SWITCH_DISCONNECTED"
MISSING_CAPACITY = "MISSING_CAPACITY"
DESTINATION_DISCONNECTED = "DESTINATION_DISCONNECTED"
DESTINATION_ALREADY_AT_RISK = "DESTINATION_ALREADY_AT_RISK"
ACTIVE_TRANSACTION_CONFLICT = "ACTIVE_TRANSACTION_CONFLICT"
COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
OUTCOME_PREDICTOR_NOT_READY = "OUTCOME_PREDICTOR_NOT_READY"
DST_UTILIZATION_UNSAFE = "DST_UTILIZATION_UNSAFE"
DST_RESPONSE_UNSAFE = "DST_RESPONSE_UNSAFE"
NO_SOURCE_RELIEF = "NO_SOURCE_RELIEF"
INSUFFICIENT_IMPROVEMENT = "INSUFFICIENT_IMPROVEMENT"

# Rejected by the generator's structural pre-check, before any prediction.
STRUCTURAL_REASONS: tuple[str, ...] = (
    SOURCE_NOT_AT_RISK, OWNER_MISMATCH, ROLE_STATE_INVALID, SWITCH_DISCONNECTED,
    MISSING_CAPACITY,
)
# Safety Gate checks in evaluation order and the reason each one reports.
CHECK_REASONS: dict[str, str] = {
    "snapshot_valid": INVALID_SNAPSHOT,
    "quality_policy": QUALITY_BELOW_POLICY,
    "destination_active": DESTINATION_DISCONNECTED,
    "destination_not_at_risk": DESTINATION_ALREADY_AT_RISK,
    "no_conflict": ACTIVE_TRANSACTION_CONFLICT,
    "cooldown_clear": COOLDOWN_ACTIVE,
    "outcome_ready": OUTCOME_PREDICTOR_NOT_READY,
    "dst_utilization": DST_UTILIZATION_UNSAFE,
    "dst_response": DST_RESPONSE_UNSAFE,
    "source_relief_positive": NO_SOURCE_RELIEF,
    "minimum_improvement": INSUFFICIENT_IMPROVEMENT,
}
GATE_CHECKS: tuple[str, ...] = tuple(CHECK_REASONS)
REASON_CODES: tuple[str, ...] = STRUCTURAL_REASONS + tuple(CHECK_REASONS.values())
# A check is SKIPPED, never silently passed, when its limit is unset in the policy
# (NOT_CONFIGURED) or its input does not exist, e.g. in offline replay (NOT_AVAILABLE).
NOT_CONFIGURED = "NOT_CONFIGURED"
NOT_AVAILABLE = "NOT_AVAILABLE"
SKIP_REASONS: tuple[str, ...] = (NOT_CONFIGURED, NOT_AVAILABLE)


def _text(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _number(name: str, value: Any, minimum: float = 0.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")


def make_candidate_id(snapshot_id: str, switch_id: str, source: str, target: str) -> str:
    return f"{snapshot_id}:{switch_id}:{source}->{target}"


@dataclass(frozen=True)
class MigrationCandidate:
    """a = (switch, source, target) on one pinned snapshot; no score (no ranking yet)."""

    snapshot_id: str
    switch_id: str
    source_controller: str
    target_controller: str
    source_utilization: float
    target_utilization: float
    switch_control_load_share: float
    generated_at: datetime
    candidate_id: str = ""

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "switch_id", "source_controller", "target_controller"):
            _text(name, getattr(self, name))
        if self.source_controller == self.target_controller:
            raise ValueError("source_controller and target_controller must differ")
        _number("source_utilization", self.source_utilization)
        _number("target_utilization", self.target_utilization)
        _number("switch_control_load_share", self.switch_control_load_share)
        if self.switch_control_load_share > 1.0:
            raise ValueError("switch_control_load_share must be <= 1")
        if not isinstance(self.generated_at, datetime) or self.generated_at.tzinfo is None:
            raise ValueError("generated_at must be a timezone-aware datetime")
        expected = make_candidate_id(self.snapshot_id, self.switch_id,
                                     self.source_controller, self.target_controller)
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", expected)
        elif self.candidate_id != expected:
            raise ValueError(f"candidate_id must be {expected!r}, got {self.candidate_id!r}")

    @property
    def moved_utilization(self) -> float:
        """Load the action takes off the source, in source-U units (share x U_src)."""
        return self.switch_control_load_share * self.source_utilization

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["generated_at"] = self.generated_at.isoformat()
        data["moved_utilization"] = self.moved_utilization
        return data


_PREDICTION_FIELDS = {
    "predicted_src_utilization": "predicted_src_utilization",
    "predicted_dst_utilization": "predicted_dst_utilization",
    "predicted_src_response_p95_ms": "predicted_src_response_p95_ms",
    "predicted_dst_response_p95_ms": "predicted_dst_response_p95_ms",
    "predicted_imbalance": "predicted_imbalance",
    "utilization_margin": "dst_utilization_margin",
    "response_margin_ms": "response_margin_ms",
    "outcome_inference_ms": "inference_time_ms",
}


@dataclass(frozen=True)
class EvaluatedCandidate:
    """A candidate with its READY outcome prediction (NOT_READY never gets here)."""

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

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MigrationCandidate):
            raise ValueError("candidate must be a MigrationCandidate")
        for name in _PREDICTION_FIELDS:
            _number(name, getattr(self, name))
        _text("outcome_model_version", self.outcome_model_version)

    @property
    def dst_upper_bound(self) -> float:
        return self.predicted_dst_utilization + self.utilization_margin

    @property
    def response_upper_bound_ms(self) -> float:
        return self.predicted_dst_response_p95_ms + self.response_margin_ms

    @property
    def predicted_src_change(self) -> float:
        """U_src before minus predicted U_src after. Includes workload drift, so it is
        reported, not gated (raw-v2: the source rose after 57/300 committed migrations)."""
        return self.candidate.source_utilization - self.predicted_src_utilization

    @classmethod
    def from_prediction(cls, candidate: MigrationCandidate, prediction: Any) -> EvaluatedCandidate:
        """From a PostMigrationOutcomePrediction of the same snapshot and action."""
        if getattr(prediction, "status", None) != "READY":
            raise ValueError(f"prediction is not READY: {getattr(prediction, 'reason', None)}")
        for name in ("snapshot_id", "switch_id", "source_controller", "target_controller"):
            if getattr(prediction, name, None) != getattr(candidate, name):
                raise ValueError(f"prediction {name} does not match the candidate")
        values = {ours: getattr(prediction, theirs) for ours, theirs in _PREDICTION_FIELDS.items()}
        missing = sorted(k for k, v in values.items() if v is None)
        if missing:
            raise ValueError(f"READY prediction without {missing}")
        return cls(candidate=candidate, outcome_model_version=str(prediction.model_version),
                   **values)


@dataclass(frozen=True)
class SafetyDecision:
    """One verdict per candidate: safe only when no check failed and every check ran."""

    candidate_id: str
    safe: bool
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    skipped_checks: tuple[tuple[str, str], ...]
    policy_version: str
    outcome_model_version: str | None
    decision_time_ms: float
    predicted_dst_utilization: float | None = None
    utilization_margin: float | None = None
    dst_upper_bound: float | None = None
    predicted_dst_response_p95_ms: float | None = None
    response_margin_ms: float | None = None
    response_upper_bound_ms: float | None = None
    moved_utilization: float | None = None
    quality_score: float | None = None
    consistency_score: float | None = None

    def __post_init__(self) -> None:
        _text("candidate_id", self.candidate_id)
        _text("policy_version", self.policy_version)
        _number("decision_time_ms", self.decision_time_ms)
        skipped = [check for check, _ in self.skipped_checks]
        for name, items, known in (("passed_checks", self.passed_checks, GATE_CHECKS),
                                   ("failed_checks", self.failed_checks, REASON_CODES),
                                   ("skipped_checks", skipped, GATE_CHECKS)):
            unknown = sorted(set(items) - set(known))
            if unknown:
                raise ValueError(f"{name}: unknown {unknown}")
            if len(set(items)) != len(items):
                raise ValueError(f"{name}: duplicates")
        bad_skip = sorted({why for _, why in self.skipped_checks} - set(SKIP_REASONS))
        if bad_skip:
            raise ValueError(f"skipped_checks: reasons must be in {SKIP_REASONS}, got {bad_skip}")
        if set(self.passed_checks) & set(skipped):
            raise ValueError("a check cannot be both passed and skipped")
        if self.safe != (not self.failed_checks):
            raise ValueError("safe must be True exactly when failed_checks is empty")
        if self.safe and set(self.passed_checks) | set(skipped) != set(GATE_CHECKS):
            raise ValueError("a safe decision needs every gate check passed or skipped")
        for name in ("predicted_dst_utilization", "utilization_margin", "dst_upper_bound",
                     "predicted_dst_response_p95_ms", "response_margin_ms",
                     "response_upper_bound_ms", "moved_utilization", "quality_score",
                     "consistency_score"):
            value = getattr(self, name)
            if value is not None:
                _number(name, value)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["skipped_checks"] = [list(item) for item in self.skipped_checks]
        return data
