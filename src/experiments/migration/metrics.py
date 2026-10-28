"""Outcome metrics of one migration attempt (Week 11).

All functions are controller-agnostic (any number of controllers, any ids)
and never clamp U to [0, 1]: U = 1.1 means 10% above safe capacity.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from src.experiments.migration.schema import (
    MigrationAction,
    MigrationTransactionView,
    PostMigrationOutcome,
    QualitySummary,
    StateSnapshotView,
)

DEFAULT_WINDOW = 5


def _values(u: Mapping[str, float]) -> list[float]:
    if not u:
        raise ValueError("utilization map is empty")
    values = [float(v) for v in u.values()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"utilization must be finite, got {dict(u)}")
    return values


def imbalance(u: Mapping[str, float]) -> float:
    """I = sqrt(mean((U_c - mean U)^2)) over controllers (population std).

    2C: I = |U_1 - U_2| / 2. A good migration lowers I.
    """
    values = _values(u)
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def max_utilization(u: Mapping[str, float]) -> float:
    return max(_values(u))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _min(values: Sequence[float]) -> float | None:
    return min(values) if values else None


def _diff(after: float | None, before: float | None) -> float | None:
    return None if after is None or before is None else after - before


def summarize_quality(
    before: Sequence[StateSnapshotView],
    after: Sequence[StateSnapshotView],
    required_window: int = DEFAULT_WINDOW,
) -> QualitySummary:
    if required_window < 1:
        raise ValueError("required_window must be >= 1")

    def scores(window: Sequence[StateSnapshotView]) -> list[float]:
        return [s.quality.quality_score for s in window if s.quality.quality_score is not None]

    return QualitySummary(
        required_window=required_window,
        before_count=len(before),
        after_count=len(after),
        before_all_valid=bool(before) and all(s.quality.valid for s in before),
        after_all_valid=bool(after) and all(s.quality.valid for s in after),
        post_window_complete=len(after) >= required_window,
        before_quality_min=_min(scores(before)),
        after_quality_min=_min(scores(after)),
        before_consistency_min=_min([s.quality.consistency_score for s in before]),
        after_consistency_min=_min([s.quality.consistency_score for s in after]),
    )


def compute_outcome(
    before: Sequence[StateSnapshotView],
    after: Sequence[StateSnapshotView],
    action: MigrationAction,
    transaction: MigrationTransactionView | None,
    required_window: int = DEFAULT_WINDOW,
) -> PostMigrationOutcome:
    """Observed outcome of one attempt.

    ``before`` / ``after`` are the pre- and post-window snapshots in time
    order. An empty ``after`` (post window timed out, SKIPPED) gives None for
    every after/delta field. Eligibility for training:
    - both windows hold ``required_window`` snapshots and all pass the Hard Gate;
    - MIGRATE: the transaction is COMMITTED (RESTORED / FAILED / SKIPPED keep
      their observed numbers but are never treated as a successful migration);
    - NO_OP: no transaction; eligible as a control sample.
    """
    if not before:
        raise ValueError("before window is empty")
    if action.action_type == "MIGRATE" and transaction is None:
        raise ValueError("MIGRATE needs a transaction view")
    if action.action_type == "NO_OP" and transaction is not None:
        raise ValueError("NO_OP has no transaction")

    src, dst = action.source_controller, action.target_controller
    src_before = [s.controller(src).utilization for s in before]
    dst_before = [s.controller(dst).utilization for s in before]
    src_after = [s.controller(src).utilization for s in after]
    dst_after = [s.controller(dst).utilization for s in after]
    imb_before = [imbalance(s.utilization()) for s in before]
    imb_after = [imbalance(s.utilization()) for s in after]

    # Load of the migrated switch, read from its owner (the source) before the move.
    switch_rates: list[float] = []
    switch_shares: list[float] = []
    if action.switch_id is not None:
        for snap in before:
            entry = snap.switch_load(action.switch_id, src)
            if entry is not None:
                switch_rates.append(entry.processed_packet_in_rate)
                switch_shares.append(entry.control_load_share)

    quality = summarize_quality(before, after, required_window)
    reasons: list[str] = []
    if quality.before_count < required_window:
        reasons.append("pre_window_incomplete")
    if not quality.before_all_valid:
        reasons.append("pre_window_invalid")
    if not quality.post_window_complete:
        reasons.append("post_window_incomplete")
    elif not quality.after_all_valid:
        reasons.append("post_window_invalid")
    status = None if transaction is None else transaction.status
    if action.action_type == "MIGRATE" and status != "COMMITTED":
        reasons.append(f"transaction_{str(status).lower()}")
    if action.action_type == "MIGRATE" and action.failure_injected:
        reasons.append("failure_injected")
    if action.action_type == "MIGRATE" and not switch_rates:
        reasons.append("switch_load_missing")

    src_before_mean, dst_before_mean = _mean(src_before), _mean(dst_before)
    src_after_mean, dst_after_mean = _mean(src_after), _mean(dst_after)
    imb_before_mean, imb_after_mean = _mean(imb_before), _mean(imb_after)
    return PostMigrationOutcome(
        action_type=action.action_type,
        transaction_status=status,
        training_eligible=not reasons,
        ineligible_reasons=tuple(reasons),
        src_u_before_last=src_before[-1],
        src_u_before_mean=src_before_mean,
        src_u_after_mean=src_after_mean,
        delta_src_u_mean=_diff(src_after_mean, src_before_mean),
        delta_src_u_last=_diff(src_after_mean, src_before[-1]),
        dst_u_before_last=dst_before[-1],
        dst_u_before_mean=dst_before_mean,
        dst_u_after_mean=dst_after_mean,
        delta_dst_u_mean=_diff(dst_after_mean, dst_before_mean),
        delta_dst_u_last=_diff(dst_after_mean, dst_before[-1]),
        imbalance_before_last=imb_before[-1],
        imbalance_before_mean=imb_before_mean,
        imbalance_after_mean=imb_after_mean,
        delta_imbalance_mean=_diff(imb_after_mean, imb_before_mean),
        max_u_before_mean=_mean([max_utilization(s.utilization()) for s in before]),
        max_u_after_mean=_mean([max_utilization(s.utilization()) for s in after]),
        dst_response_p95_before_mean=_mean([s.controller(dst).response_p95_ms for s in before]),
        dst_response_p95_after_mean=_mean([s.controller(dst).response_p95_ms for s in after]),
        switch_rate_before_mean=_mean(switch_rates),
        switch_share_before_mean=_mean(switch_shares),
    )
