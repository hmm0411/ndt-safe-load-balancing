"""Dry-run decision pipeline: candidates, pre-check, outcome, Safety Gate (Week 13, Day 4).

POST /api/v1/twin/decision/evaluate returns the funnel raw -> eligible -> safe for the
current snapshot. It never creates a transaction, never changes ownership and never sends a
role request: executing a migration stays with POST /api/v1/migrations.
"""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable

from src.experiments.migration.schema import MigrationAction
from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.forecast_service import OK
from src.orchestrator.transaction_manager import TransactionManager
from src.twin.decision.candidate_generator import generate_candidates
from src.twin.decision.candidate_precheck import PrecheckContext, precheck_all
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.decision.safety_policy import SafetyPolicy, load_safety_policy


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


class DecisionService:
    def __init__(
        self,
        store: CurrentStateStore,
        forecast_payload: Callable[[], dict[str, Any]],
        predictor: Any,
        transactions: TransactionManager,
        policy: SafetyPolicy | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.store = store
        self.forecast_payload = forecast_payload
        self.predictor = predictor
        self.transactions = transactions
        self.policy = policy or load_safety_policy()
        self.gate = SafetyGate(self.policy)
        self.clock = clock

    @staticmethod
    def _not_ready(reason: str, detail: Any = None) -> dict[str, Any]:
        return {"status": "NOT_READY", "reason": reason, "detail": detail}

    def evaluate(self) -> dict[str, Any]:
        """Body of POST /api/v1/twin/decision/evaluate. Read-only."""
        twin = self.store.twin_state()
        if twin is None:
            return self._not_ready("NO_TWIN_STATE")
        forecast = self.forecast_payload()
        if forecast.get("status") != OK:
            return self._not_ready("FORECAST_NOT_READY", forecast.get("status"))
        if forecast.get("snapshot_id") != twin.snapshot_id:
            return self._not_ready("FORECAST_SNAPSHOT_MISMATCH", forecast.get("snapshot_id"))

        # Risk per controller; a controller without a READY forecast is unknown (None).
        risk: dict[str, bool | None] = {
            str(item["controller_id"]): (
                item.get("overload_risk") if item.get("status") == "READY" else None
            )
            for item in forecast.get("controllers", [])
        }
        at_risk = sorted(cid for cid, value in risk.items() if value is True)
        now = self.clock()
        roles = self.store.role_matrix()
        last = self.transactions.last_finished_at()

        started = time.perf_counter()
        generated = generate_candidates(twin, at_risk, now)
        precheck = precheck_all(
            generated.candidates,
            twin,
            PrecheckContext(
                at_risk=frozenset(at_risk),
                role_matrix=roles,
                locked_switches=frozenset(self.transactions.locked_switches()),
                last_migration_finished_at=last,
                cooldown_seconds=self.policy.cooldown_seconds.value,
                now=now,
            ),
        )
        t_candidate = _ms(started)
        eligible = [c for c, r in zip(generated.candidates, precheck) if r.eligible]

        started = time.perf_counter()
        actions = [
            MigrationAction("MIGRATE", c.source_controller, c.target_controller, c.switch_id)
            for c in eligible
        ]
        predictions = (
            self.predictor.predict_batch(twin.to_dict(), actions, observe=False)
            if actions
            else []
        )
        t_outcome = _ms(started)
        outcome = {c.candidate_id: p for c, p in zip(eligible, predictions)}

        started = time.perf_counter()
        elapsed = None if last is None else max(0.0, (now - last).total_seconds())
        active = self.transactions.active_count() >= 1  # policy: one transaction per cycle
        decisions = {}
        for cand in eligible:
            dst_role = roles.get((cand.switch_id, cand.target_controller)) or {}
            decisions[cand.candidate_id] = self.gate.evaluate(
                cand,
                outcome[cand.candidate_id],
                GateContext(
                    snapshot_valid=twin.quality.valid,
                    destination_connected=dst_role.get("connected") is True,
                    active_transaction=active,
                    seconds_since_last_migration=elapsed,
                    destination_forecast_risk=risk.get(cand.target_controller),
                    quality_score=twin.quality.quality_score,
                    consistency_score=twin.quality.consistency_score,
                ),
            )
        t_safety = _ms(started)

        entries = []
        for cand, pre in zip(generated.candidates, precheck):
            decision = decisions.get(cand.candidate_id)
            result = outcome.get(cand.candidate_id)
            entries.append(
                {
                    "candidate": cand.to_dict(),
                    "precheck": pre.to_dict(),
                    "outcome_status": getattr(result, "status", None),
                    "outcome_reason": getattr(result, "reason", None),
                    "safety": decision.to_dict() if decision is not None else None,
                    "safe": decision is not None and decision.safe,
                }
            )
        return {
            "status": "OK",
            "snapshot_id": twin.snapshot_id,
            "policy_version": self.policy.version,
            "forecast_model_version": forecast.get("model_version"),
            "at_risk": at_risk,
            "funnel": {
                "raw": len(generated.candidates),
                "eligible": len(eligible),
                "safe": sum(1 for d in decisions.values() if d.safe),
                "dropped": len(generated.dropped),
            },
            "dropped": [list(item) for item in generated.dropped],
            "rejections": {
                "precheck": dict(Counter(x for r in precheck for x in r.reasons)),
                "safety": dict(Counter(x for d in decisions.values() for x in d.failed_checks)),
            },
            "candidates": entries,
            "timings_ms": {"t_candidate": t_candidate, "t_outcome": t_outcome, "t_safety": t_safety},
        }
