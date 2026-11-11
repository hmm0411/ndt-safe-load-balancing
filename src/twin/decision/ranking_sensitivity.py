"""Ranking sensitivity (Week 13, Day 5): does the top-1 choice survive other weight sets?

One decision epoch = one recorded pre-migration TwinState: every switch of the most
loaded controller x every other controller is a candidate, the Outcome Model predicts
them in one batch, the Safety Gate judges them, and each weight set of the ranking
policy ranks the safe ones. This measures stability only: an executed attempt holds the
observed outcome of ONE action, so it cannot say which ranking is right (no regret).
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.experiments.migration.schema import MigrationAction
from src.schemas.decision import EvaluatedCandidate, SafetyDecision
from src.schemas.snapshot import OwnershipState
from src.schemas.telemetry import SwitchTelemetry
from src.schemas.twin import ControllerTwinState, TwinQuality, TwinState
from src.twin.decision.candidate_generator import generate_candidates
from src.twin.decision.ranking import RankingContext, RankingResult, rank_candidates
from src.twin.decision.ranking_policy import RankingPolicy
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.outcome.predictor import READY, PostMigrationOutcomePredictor

# Offline: no forecast of the destination and no transaction/cooldown history exist.
REPLAY_CONTEXT = GateContext(
    snapshot_valid=True, destination_connected=True, active_transaction=False
)


def _time(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _fields(cls: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    names = cls.__dataclass_fields__
    return {k: v for k, v in row.items() if k in names}


def twin_from_payload(raw: Mapping[str, Any]) -> TwinState:
    """TwinState object from one GET /api/v1/twin/state payload (dict)."""
    controllers = [ControllerTwinState(**_fields(ControllerTwinState, c)) for c in raw["controllers"]]
    switches = []
    for row in raw["switches"]:
        data = _fields(SwitchTelemetry, row)
        data["observed_at"] = _time(data["observed_at"])
        switches.append(SwitchTelemetry(**data))
    ownership = [OwnershipState(**_fields(OwnershipState, o)) for o in raw["ownership"]]
    quality = TwinQuality(**_fields(TwinQuality, raw["quality"]))
    return TwinState(
        snapshot_id=str(raw["snapshot_id"]),
        created_at=_time(raw["created_at"]),
        topology_version=int(raw["topology_version"]),
        ownership_version=int(raw["ownership_version"]),
        controllers=controllers,
        switches=switches,
        ownership=ownership,
        quality=quality,
    )


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Kendall tau-a between two orders of the same ids; None below two common ids."""
    common = [x for x in first if x in set(second)]
    if len(common) < 2:
        return None
    pos = {x: i for i, x in enumerate(second)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos[common[i]] < pos[common[j]]:
                concordant += 1
            else:
                discordant += 1
    return (concordant - discordant) / (concordant + discordant)


@dataclass(frozen=True)
class Epoch:
    epoch_id: str
    source: str
    n_candidates: int
    n_safe: int
    results: dict[str, RankingResult]


def rank_epoch(
    epoch_id: str,
    twin_raw: Mapping[str, Any],
    predictor: PostMigrationOutcomePredictor,
    gate: SafetyGate,
    policies: Mapping[str, RankingPolicy],
    now: datetime,
) -> Epoch:
    """Rank one recorded snapshot under every policy; the predictor history is ready."""
    twin = twin_from_payload(twin_raw)
    utilization = {c.controller_id: c.utilization for c in twin.controllers}
    source = max(sorted(utilization), key=lambda cid: utilization[cid])
    candidates = generate_candidates(twin, [source], now).candidates
    actions = [
        MigrationAction("MIGRATE", c.source_controller, c.target_controller, c.switch_id)
        for c in candidates
    ]
    predictions = predictor.predict_batch(twin_raw, actions, observe=False)
    items: list[tuple[SafetyDecision, EvaluatedCandidate | None]] = []
    for cand, pred in zip(candidates, predictions):
        decision = gate.evaluate(cand, pred, REPLAY_CONTEXT)
        evaluated = (
            EvaluatedCandidate.from_prediction(cand, pred) if pred.status == READY else None
        )
        items.append((decision, evaluated))
    limit = gate.policy.utilization_safe_limit.value
    context = RankingContext(controller_utilization=utilization, utilization_limit=float(limit or 1.0))
    results = {name: rank_candidates(items, context, p) for name, p in policies.items()}
    n_safe = sum(1 for decision, _ in items if decision.safe)
    return Epoch(epoch_id, source, len(candidates), n_safe, results)


def summarize(epochs: Sequence[Epoch], base: str = "base") -> dict[str, Any]:
    """Top-1 agreement and Kendall tau of each weight set against the base set."""
    if not epochs:
        return {"n_epochs": 0, "policies": {}}
    names = list(epochs[0].results)
    status = Counter(e.results[base].status for e in epochs)
    out: dict[str, Any] = {
        "n_epochs": len(epochs),
        "n_epochs_ranked": sum(1 for e in epochs if e.results[base].ranked),
        "n_epochs_two_or_more_safe": sum(1 for e in epochs if e.n_safe >= 2),
        "base_status": dict(sorted(status.items())),
        "candidates_per_epoch_mean": sum(e.n_candidates for e in epochs) / len(epochs),
        "safe_per_epoch_mean": sum(e.n_safe for e in epochs) / len(epochs),
        "policies": {},
    }
    for name in names:
        same, taus = 0, []
        compared = 0
        for e in epochs:
            ref, other = e.results[base], e.results[name]
            if e.n_safe < 2:
                continue
            compared += 1
            same += ref.selected_candidate_id == other.selected_candidate_id
            tau = kendall_tau([r.candidate_id for r in ref.ranked],
                              [r.candidate_id for r in other.ranked])
            if tau is not None and math.isfinite(tau):
                taus.append(tau)
        out["policies"][name] = {
            "effective_weights": epochs[0].results[name].effective_weights,
            "epochs_compared": compared,
            "top1_same_as_base": same,
            "top1_agreement": same / compared if compared else None,
            "kendall_tau_mean": sum(taus) / len(taus) if taus else None,
            "kendall_tau_min": min(taus) if taus else None,
        }
    return out
