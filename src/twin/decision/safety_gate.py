"""Safety Gate v1 (Week 13, Day 3): ordered, non-compensatory checks on one candidate.

A candidate is safe only when no check fails; a check whose limit or input is missing is
SKIPPED with its reason (NOT_CONFIGURED / NOT_AVAILABLE), never recorded as passed. The
11 checks run in the order of src.schemas.decision.GATE_CHECKS. By default every check
runs, so an offline replay records all failing reasons; stop_at_first_failure=True gives
the short-circuit a runtime caller may want. Ranking is separate and never looks here.
A prediction that is not READY rejects the candidate (OUTCOME_PREDICTOR_NOT_READY): the
Gate never falls back to another model. Utilization is not clamped; boundary is inclusive.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from src.schemas.decision import (
    CHECK_REASONS,
    GATE_CHECKS,
    NOT_AVAILABLE,
    NOT_CONFIGURED,
    EvaluatedCandidate,
    MigrationCandidate,
    SafetyDecision,
)
from src.twin.decision.safety_policy import EPSILON, SafetyPolicy, within_limit

READY = "READY"
PASSED = "passed"
FAILED = "failed"
SKIPPED = "skipped"
# (status, why): why is the skip reason for SKIPPED and "" otherwise.
CheckResult = tuple[str, str]
_PASS: CheckResult = (PASSED, "")
_FAIL: CheckResult = (FAILED, "")
_NOT_CONFIGURED: CheckResult = (SKIPPED, NOT_CONFIGURED)
_NOT_AVAILABLE: CheckResult = (SKIPPED, NOT_AVAILABLE)


@dataclass(frozen=True)
class GateContext:
    """What the caller knows about the system at decision time (not about the outcome)."""

    snapshot_valid: bool
    destination_connected: bool
    active_transaction: bool
    # Seconds since the last migration finished (global cooldown); None = none yet.
    seconds_since_last_migration: float | None = None
    # Forecast risk of the destination; None = no forecast (offline replay).
    destination_forecast_risk: bool | None = None
    quality_score: float | None = None
    consistency_score: float | None = None

    def __post_init__(self) -> None:
        elapsed = self.seconds_since_last_migration
        if elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("seconds_since_last_migration must be finite and >= 0")


def _pass_if(condition: bool) -> CheckResult:
    return _PASS if condition else _FAIL


class SafetyGate:
    def __init__(self, policy: SafetyPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        candidate: MigrationCandidate,
        prediction: Any,
        context: GateContext,
        *,
        stop_at_first_failure: bool = False,
    ) -> SafetyDecision:
        """One verdict for one candidate and its PostMigrationOutcomePrediction (or None)."""
        started = time.perf_counter()
        evaluated = self._evaluated(candidate, prediction)
        checks = {
            "snapshot_valid": lambda: _pass_if(context.snapshot_valid),
            "quality_policy": lambda: self._quality(context, prediction),
            "destination_active": lambda: self._destination_active(context),
            "destination_not_at_risk": lambda: self._destination_risk(context),
            "no_conflict": lambda: _pass_if(not context.active_transaction),
            "cooldown_clear": lambda: self._cooldown(context),
            "outcome_ready": lambda: _pass_if(evaluated is not None),
            "dst_utilization": lambda: self._utilization(evaluated),
            "dst_response": lambda: self._response(evaluated),
            "source_relief_positive": lambda: self._relief_positive(candidate),
            "minimum_improvement": lambda: self._minimum_improvement(candidate),
        }
        passed: list[str] = []
        failed: list[str] = []
        skipped: list[tuple[str, str]] = []
        for check in GATE_CHECKS:
            status, why = checks[check]()
            if status == PASSED:
                passed.append(check)
            elif status == SKIPPED:
                skipped.append((check, why))
            else:
                failed.append(CHECK_REASONS[check])
                if stop_at_first_failure:
                    break
        model_version = (
            evaluated.outcome_model_version
            if evaluated is not None
            else _text_or_none(getattr(prediction, "model_version", None))
        )
        quality = context.quality_score
        if quality is None:
            quality = getattr(prediction, "quality_score", None)
        consistency = context.consistency_score
        if consistency is None:
            consistency = getattr(prediction, "consistency_score", None)
        return SafetyDecision(
            candidate_id=candidate.candidate_id,
            safe=not failed,
            passed_checks=tuple(passed),
            failed_checks=tuple(failed),
            skipped_checks=tuple(skipped),
            policy_version=self.policy.version,
            outcome_model_version=model_version,
            decision_time_ms=(time.perf_counter() - started) * 1000.0,
            predicted_dst_utilization=_pick(evaluated, "predicted_dst_utilization"),
            utilization_margin=_pick(evaluated, "utilization_margin"),
            dst_upper_bound=_pick(evaluated, "dst_upper_bound"),
            predicted_dst_response_p95_ms=_pick(evaluated, "predicted_dst_response_p95_ms"),
            response_margin_ms=_pick(evaluated, "response_margin_ms"),
            response_upper_bound_ms=_pick(evaluated, "response_upper_bound_ms"),
            moved_utilization=candidate.moved_utilization,
            quality_score=quality,
            consistency_score=consistency,
        )

    def _evaluated(self, candidate: MigrationCandidate, prediction: Any) -> EvaluatedCandidate | None:
        """The READY prediction of this candidate from the policy's model, else None."""
        if prediction is None or getattr(prediction, "status", None) != READY:
            return None
        if getattr(prediction, "model_version", None) != self.policy.outcome_model_version:
            return None
        try:
            return EvaluatedCandidate.from_prediction(candidate, prediction)
        except ValueError:
            return None

    def _quality(self, context: GateContext, prediction: Any) -> CheckResult:
        limit = self.policy.minimum_quality_score.value
        if limit is None:
            return _NOT_CONFIGURED
        score = context.quality_score
        if score is None:
            score = getattr(prediction, "quality_score", None)
        if score is None:
            return _NOT_AVAILABLE
        return _pass_if(score + EPSILON >= limit)

    def _destination_active(self, context: GateContext) -> CheckResult:
        if not self.policy.require_destination_connected:
            return _NOT_CONFIGURED
        return _pass_if(context.destination_connected)

    def _destination_risk(self, context: GateContext) -> CheckResult:
        if not self.policy.reject_if_destination_at_risk:
            return _NOT_CONFIGURED
        if context.destination_forecast_risk is None:
            return _NOT_AVAILABLE
        return _pass_if(not context.destination_forecast_risk)

    def _cooldown(self, context: GateContext) -> CheckResult:
        limit = self.policy.cooldown_seconds.value
        if limit is None:
            return _NOT_CONFIGURED
        elapsed = context.seconds_since_last_migration
        return _pass_if(elapsed is None or elapsed >= limit)

    def _utilization(self, evaluated: EvaluatedCandidate | None) -> CheckResult:
        limit = self.policy.utilization_safe_limit.value
        if limit is None:
            return _NOT_CONFIGURED
        if evaluated is None:
            return _NOT_AVAILABLE
        return _pass_if(
            within_limit(evaluated.predicted_dst_utilization, evaluated.utilization_margin, limit)
        )

    def _response(self, evaluated: EvaluatedCandidate | None) -> CheckResult:
        limit = self.policy.response_p95_max_ms.value
        if limit is None:
            return _NOT_CONFIGURED
        if evaluated is None:
            return _NOT_AVAILABLE
        return _pass_if(
            within_limit(
                evaluated.predicted_dst_response_p95_ms, evaluated.response_margin_ms, limit
            )
        )

    def _relief_positive(self, candidate: MigrationCandidate) -> CheckResult:
        if not self.policy.require_positive_source_relief:
            return _NOT_CONFIGURED
        return _pass_if(candidate.moved_utilization > 0)

    def _minimum_improvement(self, candidate: MigrationCandidate) -> CheckResult:
        limit = self.policy.minimum_source_relief.value
        if limit is None:
            return _NOT_CONFIGURED
        return _pass_if(candidate.moved_utilization + EPSILON >= limit)


def _pick(evaluated: EvaluatedCandidate | None, name: str) -> float | None:
    return None if evaluated is None else float(getattr(evaluated, name))


def _text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
