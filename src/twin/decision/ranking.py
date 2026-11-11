"""Ranking v1 (Week 13, Day 5): order the SAFE candidates of one decision epoch.

Safety decides feasibility, ranking decides preference: a candidate the Gate rejected is
never scored, whatever its predicted outcome. Each enabled component is a raw metric of
the predicted outcome, min-max normalized over the safe set of this epoch (equal values
give 0.5), and the total is their weighted sum with the policy's renormalized weights.
NO_MIGRATION is always available: it is chosen when nothing is safe (NO_SAFE_ACTION) or
when no safe candidate has a strictly positive value of every raw metric the policy lists
in no_migration.require_positive (NO_BENEFICIAL_ACTION).
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.schemas.decision import EvaluatedCandidate, SafetyDecision
from src.twin.decision.ranking_policy import COMPONENTS, RankingPolicy

SELECTED = "SELECTED"
NO_SAFE_ACTION = "NO_SAFE_ACTION"
NO_BENEFICIAL_ACTION = "NO_BENEFICIAL_ACTION"
EQUAL_NORM = 0.5  # min-max of a metric that does not separate the candidates
RAW_NAMES = {
    "source_relief": "source_relief",
    "destination_headroom": "destination_headroom",
    "imbalance": "imbalance_gain",
    "response": "response_headroom_ms",
    "global_improvement": "max_utilization_gain",
    "migration_cost": "migration_cost",
}


@dataclass(frozen=True)
class RankingContext:
    """What ranking needs beyond the candidates, all from the same pinned snapshot."""

    controller_utilization: Mapping[str, float]  # every controller, not only src/dst
    utilization_limit: float  # U_safe of the Safety Policy (1.0)
    response_limit_ms: float | None = None  # needed only when 'response' is enabled
    migration_cost: Mapping[str, float] | None = None  # candidate_id -> predicted cost

    def __post_init__(self) -> None:
        if not self.controller_utilization:
            raise ValueError("controller_utilization is empty")
        for cid, value in self.controller_utilization.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"utilization of {cid} must be finite and >= 0")
        if not math.isfinite(self.utilization_limit) or self.utilization_limit <= 0:
            raise ValueError("utilization_limit must be finite and > 0")


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    rank: int
    total_score: float
    switch_id: str
    source_controller: str
    target_controller: str
    source_relief_score: float | None
    destination_headroom_score: float | None
    imbalance_score: float | None
    response_score: float | None
    global_improvement_score: float | None
    migration_cost_score: float | None
    source_relief: float
    destination_headroom: float
    imbalance_gain: float
    max_utilization_gain: float
    response_headroom_ms: float | None
    migration_cost: float | None
    predicted_src_utilization: float
    predicted_dst_utilization: float
    predicted_imbalance: float
    dst_upper_bound: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankingResult:
    status: str
    selected_candidate_id: str | None
    ranked: tuple[RankedCandidate, ...]
    n_candidates: int
    n_safe: int
    policy_version: str
    safety_policy_version: str
    effective_weights: dict[str, float]
    ranking_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ranked"] = [r.to_dict() for r in self.ranked]
        return data


def minmax(values: Sequence[float]) -> list[float]:
    """Min-max to [0, 1]; a metric with one distinct value gives EQUAL_NORM to all."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return [EQUAL_NORM] * len(values)
    return [(v - low) / (high - low) for v in values]


def order_key(total: float, evaluated: EvaluatedCandidate) -> tuple[float, float, str]:
    """Total score desc (rounded so float noise is a tie), dst upper bound asc, id asc."""
    return (-round(total, 12), evaluated.dst_upper_bound, evaluated.candidate.candidate_id)


def raw_metrics(
    evaluated: EvaluatedCandidate, context: RankingContext, enabled: Sequence[str]
) -> dict[str, float]:
    """The raw value of every enabled component (bigger is better, except cost)."""
    cand = evaluated.candidate
    src, dst = cand.source_controller, cand.target_controller
    utilization = context.controller_utilization
    if src not in utilization or dst not in utilization:
        raise ValueError(f"{cand.candidate_id}: source or target missing from the snapshot")
    others = [u for cid, u in utilization.items() if cid not in (src, dst)]
    after = others + [evaluated.predicted_src_utilization, evaluated.predicted_dst_utilization]
    before_imbalance = abs(cand.source_utilization - cand.target_utilization) / 2.0
    values: dict[str, float] = {
        "source_relief": evaluated.predicted_src_change,
        "destination_headroom": context.utilization_limit - evaluated.dst_upper_bound,
        "imbalance": before_imbalance - evaluated.predicted_imbalance,
        "global_improvement": max(utilization.values()) - max(after),
    }
    if "response" in enabled:
        if context.response_limit_ms is None:
            raise ValueError("'response' is enabled but the context has no response limit")
        values["response"] = context.response_limit_ms - evaluated.response_upper_bound_ms
    if "migration_cost" in enabled:
        costs = context.migration_cost or {}
        if cand.candidate_id not in costs:
            raise ValueError(f"'migration_cost' is enabled but {cand.candidate_id} has no cost")
        values["migration_cost"] = float(costs[cand.candidate_id])
    return {name: values[name] for name in COMPONENTS if name in values}


def _safe_pairs(
    items: Sequence[tuple[SafetyDecision, EvaluatedCandidate | None]], policy: RankingPolicy
) -> list[EvaluatedCandidate]:
    safe: list[EvaluatedCandidate] = []
    for decision, evaluated in items:
        if decision.policy_version != policy.safety_policy_version:
            raise ValueError(
                f"decision of {decision.policy_version!r}, ranking expects "
                f"{policy.safety_policy_version!r}"
            )
        if not decision.safe:
            continue
        if evaluated is None:
            raise ValueError(f"safe decision {decision.candidate_id} has no evaluated candidate")
        if evaluated.candidate.candidate_id != decision.candidate_id:
            raise ValueError(f"decision {decision.candidate_id} paired with another candidate")
        safe.append(evaluated)
    return safe


def rank_candidates(
    items: Sequence[tuple[SafetyDecision, EvaluatedCandidate | None]],
    context: RankingContext,
    policy: RankingPolicy,
) -> RankingResult:
    """Rank the safe candidates of one epoch; every Gate verdict of the epoch goes in."""
    started = time.perf_counter()
    safe = _safe_pairs(items, policy)
    ids = [e.candidate.candidate_id for e in safe]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate_id among the safe candidates")
    weights = policy.effective_weights()
    enabled = list(weights)
    raws = [raw_metrics(e, context, enabled) for e in safe]
    norms = {name: minmax([r[name] for r in raws]) for name in enabled}
    if "migration_cost" in norms:  # lower cost is better
        norms["migration_cost"] = [1.0 - v for v in norms["migration_cost"]]
    rows = []
    for i, e in enumerate(safe):
        scores = {name: norms[name][i] for name in enabled}
        total = sum(weights[name] * scores[name] for name in enabled)
        rows.append((total, e, scores, raws[i]))
    rows.sort(key=lambda r: order_key(r[0], r[1]))
    ranked = tuple(_ranked(k + 1, *row) for k, row in enumerate(rows))
    selected = next(
        (r for r in ranked if all(getattr(r, RAW_NAMES[m]) > 0 for m in policy.require_positive)),
        None,
    )
    if not ranked:
        status = NO_SAFE_ACTION
    elif selected is None:
        status = NO_BENEFICIAL_ACTION
    else:
        status = SELECTED
    return RankingResult(
        status=status,
        selected_candidate_id=selected.candidate_id if selected is not None else None,
        ranked=ranked,
        n_candidates=len(items),
        n_safe=len(safe),
        policy_version=policy.version,
        safety_policy_version=policy.safety_policy_version,
        effective_weights=weights,
        ranking_time_ms=(time.perf_counter() - started) * 1000.0,
    )


def _ranked(
    rank: int,
    total: float,
    e: EvaluatedCandidate,
    scores: Mapping[str, float],
    raw: Mapping[str, float],
) -> RankedCandidate:
    c = e.candidate
    return RankedCandidate(
        candidate_id=c.candidate_id,
        rank=rank,
        total_score=total,
        switch_id=c.switch_id,
        source_controller=c.source_controller,
        target_controller=c.target_controller,
        source_relief_score=scores.get("source_relief"),
        destination_headroom_score=scores.get("destination_headroom"),
        imbalance_score=scores.get("imbalance"),
        response_score=scores.get("response"),
        global_improvement_score=scores.get("global_improvement"),
        migration_cost_score=scores.get("migration_cost"),
        source_relief=raw["source_relief"],
        destination_headroom=raw["destination_headroom"],
        imbalance_gain=raw["imbalance"],
        max_utilization_gain=raw["global_improvement"],
        response_headroom_ms=raw.get("response"),
        migration_cost=raw.get("migration_cost"),
        predicted_src_utilization=e.predicted_src_utilization,
        predicted_dst_utilization=e.predicted_dst_utilization,
        predicted_imbalance=e.predicted_imbalance,
        dst_upper_bound=e.dst_upper_bound,
    )
