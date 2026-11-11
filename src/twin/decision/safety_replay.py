"""Offline replay of the Safety Gate on executed migrations (Week 13, Day 3).

Each row is one COMMITTED migration of the processed outcome dataset: its pre-migration
state gives the candidate, a model prediction gives the Gate's input, and the OBSERVED
u_dst_post / r_dst95_post give the ground truth. Only executed migrations have an observed
outcome, so rejected-and-never-run candidates are not part of this evaluation (selection
effect). Gate A uses no uncertainty margin (0, 0); Gate B uses the margins of the artifact.
Checks that need data the dataset does not hold (forecast risk, cooldown) are not invented:
they stay skipped and are counted as such.
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from src.schemas.decision import MigrationCandidate
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.decision.safety_metrics import ReplayRecord, SafetyMetrics, compute_metrics
from src.twin.decision.safety_policy import Limit, SafetyPolicy
from src.twin.outcome.predictor import MODEL_NAME, MODEL_VERSION, PostMigrationOutcomePrediction

GENERATED_AT = datetime(2026, 11, 9, tzinfo=timezone.utc)  # fixed: a replay has no clock
CONTEXT = GateContext(snapshot_valid=True, destination_connected=True, active_transaction=False)


@dataclass(frozen=True)
class ReplayResult:
    metrics: SafetyMetrics
    failed_reasons: dict[str, int]
    skipped_checks: dict[str, int]
    records: tuple[ReplayRecord, ...]


def candidate_from_row(row: Mapping[str, Any]) -> MigrationCandidate:
    return MigrationCandidate(
        snapshot_id=str(row["attempt_id"]),
        switch_id=str(row["switch_id"]),
        source_controller=str(row["source_controller"]),
        target_controller=str(row["target_controller"]),
        source_utilization=float(row["src_utilization"]),
        target_utilization=float(row["dst_utilization"]),
        switch_control_load_share=float(row["switch_control_load_share"]),
        generated_at=GENERATED_AT,
    )


def with_minimum_relief(policy: SafetyPolicy, value: float | None, source: str) -> SafetyPolicy:
    """A copy of the policy with another minimum_source_relief (None = not configured)."""
    limit = Limit(None, reason=source) if value is None else Limit(value, source=source)
    return dataclasses.replace(policy, minimum_source_relief=limit)


def prediction_from_values(
    row: Mapping[str, Any], values: Mapping[str, float], m_u: float, m_r: float
) -> PostMigrationOutcomePrediction:
    return PostMigrationOutcomePrediction(
        snapshot_id=str(row["attempt_id"]),
        switch_id=str(row["switch_id"]),
        source_controller=str(row["source_controller"]),
        target_controller=str(row["target_controller"]),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        predicted_src_utilization=values["u_src_post"],
        predicted_dst_utilization=values["u_dst_post"],
        predicted_src_response_p95_ms=values["r_src95_post"],
        predicted_dst_response_p95_ms=values["r_dst95_post"],
        predicted_worst_response_p95_ms=values["r95_post"],
        predicted_imbalance=values["i_post"],
        dst_utilization_margin=m_u,
        response_margin_ms=m_r,
        inference_time_ms=0.0,  # latency is not what a replay measures; the Gate ignores it
        status="READY",
    )


def run_gate(
    rows: Sequence[Mapping[str, Any]],
    predicted: Mapping[str, Sequence[float]],
    policy: SafetyPolicy,
    m_u: float,
    m_r: float,
) -> ReplayResult:
    """Gate verdicts of every row with margins (m_u, m_r), scored against what happened."""
    if any(len(series) != len(rows) for series in predicted.values()):
        raise ValueError("every predicted series needs one value per row")
    u_limit = policy.utilization_safe_limit.value
    if u_limit is None:
        raise ValueError("the policy has no utilization limit")
    gate = SafetyGate(policy)
    records: list[ReplayRecord] = []
    failed: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for i, row in enumerate(rows):
        values = {name: float(series[i]) for name, series in predicted.items()}
        candidate = candidate_from_row(row)
        decision = gate.evaluate(candidate, prediction_from_values(row, values, m_u, m_r), CONTEXT)
        failed.update(decision.failed_checks)
        skipped.update(f"{check}:{why}" for check, why in decision.skipped_checks)
        records.append(ReplayRecord(
            candidate.candidate_id, decision.safe,
            float(row["u_dst_post"]), float(row["r_dst95_post"]),
        ))
    metrics = compute_metrics(records, u_limit, policy.response_p95_max_ms.value)
    return ReplayResult(metrics, dict(failed), dict(skipped), tuple(records))
