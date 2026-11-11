"""Decision pipeline of the orchestrator (Week 13, Day 4-5).

POST /api/v1/twin/decision/evaluate: funnel raw -> eligible -> safe of the current
snapshot (read-only). POST /decision/plan adds ranking and returns the selected action
without executing it (dry run). POST /decision/execute revalidates the selected action
on the latest state and hands it to the migration executor (role request, barrier,
verification, rollback stay there). Every plan/execute writes one decision record.
"""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable

from src.experiments.migration.schema import MigrationAction
from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.forecast_service import OK
from src.orchestrator.transaction_manager import TransactionManager
from src.schemas.decision import (
    ACTIVE_TRANSACTION_CONFLICT,
    COOLDOWN_ACTIVE,
    INVALID_SNAPSHOT,
    EvaluatedCandidate,
    MigrationCandidate,
    SafetyDecision,
)
from src.schemas.twin import TwinState
from src.twin.decision import closed_loop as cl
from src.twin.decision.candidate_generator import CandidateGeneration, generate_candidates
from src.twin.decision.candidate_precheck import PrecheckContext, PrecheckResult, precheck_all
from src.twin.decision.decision_context import DRY_RUN, EXECUTE, DecisionContext
from src.twin.decision.ranking import RankingContext, rank_candidates
from src.twin.decision.ranking_policy import RankingPolicy, load_ranking_policy
from src.twin.decision.safety_gate import GateContext, SafetyGate
from src.twin.decision.safety_policy import SafetyPolicy, load_safety_policy


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


@dataclass
class _Pipeline:
    """Everything one evaluation produced, before it is turned into a payload."""

    at_risk: list[str]
    risk: dict[str, bool | None]
    roles: Any
    now: datetime
    generated: CandidateGeneration
    precheck: list[PrecheckResult]
    eligible: list[MigrationCandidate]
    outcome: dict[str, Any]
    decisions: dict[str, SafetyDecision]
    t_candidate: float
    t_outcome: float
    t_safety: float


# executor(switch_id, target_controller, simulate_failure) -> body of POST /migrations
Executor = Callable[[str, str, str], dict[str, Any]]


class DecisionService:
    def __init__(
        self,
        store: CurrentStateStore,
        forecast_payload: Callable[[], dict[str, Any]],
        predictor: Any,
        transactions: TransactionManager,
        policy: SafetyPolicy | None = None,
        clock: Callable[[], datetime] = _utc_now,
        ranking_policy: RankingPolicy | None = None,
        loop_config: cl.ClosedLoopConfig | None = None,
        executor: Executor | None = None,
        record_root: Path | None = None,
    ) -> None:
        self.store = store
        self.forecast_payload = forecast_payload
        self.predictor = predictor
        self.transactions = transactions
        self.policy = policy or load_safety_policy()
        self.gate = SafetyGate(self.policy)
        self.clock = clock
        self.ranking_policy = ranking_policy or load_ranking_policy()
        self.loop_config = loop_config or cl.load_config()
        self.executor = executor
        self.enabled = self.loop_config.enabled
        self.record_root = record_root  # None = records stay in memory (tests)
        self._lock = RLock()  # serializes decisions (a migration may run inside it)
        self._state_lock = RLock()  # cache and pending windows; never held during I/O waits
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._pending: list[cl.PostWindow] = []

    @staticmethod
    def _not_ready(reason: str, detail: Any = None) -> dict[str, Any]:
        return {"status": "NOT_READY", "reason": reason, "detail": detail}

    @staticmethod
    def _risk(forecast: dict[str, Any]) -> dict[str, bool | None]:
        """Risk per controller; a controller without a READY forecast is unknown (None)."""
        return {
            str(item["controller_id"]): (
                item.get("overload_risk") if item.get("status") == "READY" else None
            )
            for item in forecast.get("controllers", [])
        }

    def _pipeline(self, twin: TwinState, forecast: dict[str, Any]) -> _Pipeline:
        """Candidates, pre-check, outcome and Safety Gate on ONE pinned snapshot."""
        risk = self._risk(forecast)
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
        decisions: dict[str, SafetyDecision] = {}
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
        return _Pipeline(at_risk, risk, roles, now, generated, precheck, eligible, outcome,
                         decisions, t_candidate, t_outcome, t_safety)

    @staticmethod
    def _entries(run: _Pipeline) -> list[dict[str, Any]]:
        entries = []
        for cand, pre in zip(run.generated.candidates, run.precheck):
            decision = run.decisions.get(cand.candidate_id)
            result = run.outcome.get(cand.candidate_id)
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
        return entries

    @staticmethod
    def _rejections(run: _Pipeline) -> dict[str, dict[str, int]]:
        return {
            "precheck": dict(Counter(x for r in run.precheck for x in r.reasons)),
            "safety": dict(Counter(x for d in run.decisions.values() for x in d.failed_checks)),
        }

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
        run = self._pipeline(twin, forecast)
        return {
            "status": "OK",
            "snapshot_id": twin.snapshot_id,
            "policy_version": self.policy.version,
            "forecast_model_version": forecast.get("model_version"),
            "at_risk": run.at_risk,
            "funnel": {
                "raw": len(run.generated.candidates),
                "eligible": len(run.eligible),
                "safe": sum(1 for d in run.decisions.values() if d.safe),
                "dropped": len(run.generated.dropped),
            },
            "dropped": [list(item) for item in run.generated.dropped],
            "rejections": self._rejections(run),
            "candidates": self._entries(run),
            "timings_ms": {"t_candidate": run.t_candidate, "t_outcome": run.t_outcome,
                           "t_safety": run.t_safety},
        }

    # ------------------------------------------------------------------ closed loop
    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        """Kill switch: False stops every plan/execute before any work."""
        self.enabled = bool(enabled)
        return {"enabled": self.enabled}

    def plan(self, scenario: str = "unlabelled") -> dict[str, Any]:
        """Body of POST /decision/plan: the selected action, nothing executed."""
        return self._decide(DRY_RUN, scenario, "none")

    def execute(self, scenario: str = "unlabelled", simulate_failure: str = "none") -> dict[str, Any]:
        """Body of POST /decision/execute: plan, revalidate, then the migration executor."""
        return self._decide(EXECUTE, scenario, simulate_failure)

    def on_twin_state(self, twin_state: dict[str, Any]) -> None:
        """Telemetry hook: completes the observed outcome of committed migrations."""
        with self._state_lock:
            for window in list(self._pending):
                actual = window.observe(twin_state)
                if actual is None:
                    continue
                self._pending.remove(window)
                record = self._cache.get(window.decision_id)
                if record is not None:
                    record["actual_post"] = actual
                    self._save(record)

    def _save(self, record: dict[str, Any]) -> None:
        with self._state_lock:
            self._cache[record["decision_id"]] = record
            self._cache.move_to_end(record["decision_id"])
            while len(self._cache) > self.loop_config.max_cached_decisions:
                self._cache.popitem(last=False)
            if self.record_root is not None:
                cl.write_record(self.record_root, record)

    def _decide(self, mode: str, scenario: str, simulate_failure: str) -> dict[str, Any]:
        with self._lock:  # one decision at a time: a second request waits, then replays
            twin = self.store.twin_state()
            if twin is None:
                return self._not_ready("NO_TWIN_STATE")
            ctx = DecisionContext.pin(mode, scenario, twin.snapshot_id, twin.created_at,
                                      self.clock())
            with self._state_lock:
                cached = self._cache.get(ctx.decision_id)
            if cached is not None:
                return {**cached, "idempotent_replay": True}
            record = self._run_decision(ctx, twin, simulate_failure)
            self._save(record)
            return record

    def _base_record(self, ctx: DecisionContext, twin: TwinState) -> dict[str, Any]:
        return {
            **ctx.to_dict(),
            "loop_version": self.loop_config.version,
            "safety_policy_version": self.policy.version,
            "ranking_policy_version": self.ranking_policy.version,
            "status": cl.NOT_READY,
            "detail": None,
            "timings_ms": {},
            "collection_latency_ms": max(
                (c.collection_latency_ms for c in twin.controllers), default=None
            ),
            "funnel": {"raw": 0, "eligible": 0, "evaluated": 0, "safe": 0},
            "reasons": {},
            "at_risk": [],
            "selected": None,
            "ranking": None,
            "candidates": [],
            "revalidation": None,
            "transaction": None,
            "actual_post": None,
            "twin_state": twin.to_dict(),
        }

    def _run_decision(self, ctx: DecisionContext, twin: TwinState,
                      simulate_failure: str) -> dict[str, Any]:
        record = self._base_record(ctx, twin)

        def stop(status: str, detail: Any = None, reasons: dict[str, int] | None = None) -> dict[str, Any]:
            record.update(status=status, detail=detail)
            if reasons:
                record["reasons"] = reasons
            return record

        if not self.enabled:
            return stop(cl.KILL_SWITCH)
        if not twin.quality.valid:  # Hard Gate: nothing runs on an invalid snapshot
            return stop(INVALID_SNAPSHOT, reasons={INVALID_SNAPSHOT: 1})
        forecast = self.forecast_payload()
        record["forecast_snapshot_id"] = forecast.get("snapshot_id")
        record["forecast_model_version"] = forecast.get("model_version")
        if forecast.get("status") != OK:
            return stop(cl.NOT_READY, f"FORECAST_NOT_READY:{forecast.get('status')}")
        if forecast.get("snapshot_id") != twin.snapshot_id:
            return stop(cl.NOT_READY, "FORECAST_SNAPSHOT_MISMATCH")
        record["timings_ms"]["forecast"] = float(forecast.get("service_time_ms") or 0.0)

        started = time.perf_counter()
        risk = self._risk(forecast)
        at_risk = sorted(cid for cid, value in risk.items() if value is True)
        record["timings_ms"]["risk"] = _ms(started)
        record["at_risk"] = at_risk
        if not at_risk:
            return stop(cl.NO_RISK)
        last = self.transactions.last_finished_at()
        elapsed = None if last is None else max(0.0, (ctx.decided_at - last).total_seconds())
        cooldown = self.policy.cooldown_seconds.value
        if self.transactions.active_count() >= 1:
            return stop(cl.DEFERRED, reasons={ACTIVE_TRANSACTION_CONFLICT: 1})
        if cooldown is not None and elapsed is not None and elapsed < cooldown:
            return stop(cl.DEFERRED, f"cooldown {elapsed:.1f}/{cooldown:.0f} s",
                        {COOLDOWN_ACTIVE: 1})

        run = self._pipeline(twin, forecast)
        items: list[tuple[SafetyDecision, EvaluatedCandidate | None]] = []
        for cand in run.eligible:
            prediction = run.outcome[cand.candidate_id]
            evaluated = None
            if getattr(prediction, "status", None) == "READY":
                try:
                    evaluated = EvaluatedCandidate.from_prediction(cand, prediction)
                except ValueError:
                    evaluated = None
            items.append((run.decisions[cand.candidate_id], evaluated))
        started = time.perf_counter()
        limit = self.policy.utilization_safe_limit.value
        ranking = rank_candidates(
            items,
            RankingContext(
                controller_utilization={c.controller_id: c.utilization for c in twin.controllers},
                utilization_limit=float(limit if limit is not None else 1.0),
            ),
            self.ranking_policy,
        )
        rejections = self._rejections(run)
        reasons = Counter(rejections["precheck"])
        reasons.update(rejections["safety"])
        record["timings_ms"].update(candidate=run.t_candidate, outcome=run.t_outcome,
                                    safety=run.t_safety, ranking=_ms(started))
        record.update(
            funnel={"raw": len(run.generated.candidates), "eligible": len(run.eligible),
                    "evaluated": sum(1 for _, e in items if e is not None),
                    "safe": ranking.n_safe},
            reasons=dict(reasons),
            candidates=self._entries(run),
            ranking=ranking.to_dict(),
        )
        if ranking.selected_candidate_id is None:
            return stop(ranking.status)
        chosen = next(r for r in ranking.ranked if r.candidate_id == ranking.selected_candidate_id)
        candidate = next(c for c in run.eligible if c.candidate_id == chosen.candidate_id)
        decision = run.decisions[chosen.candidate_id]
        record["selected"] = {
            "candidate_id": chosen.candidate_id, "switch_id": chosen.switch_id,
            "source_controller": chosen.source_controller,
            "target_controller": chosen.target_controller,
            "total_score": chosen.total_score,
            "predicted_src_utilization": chosen.predicted_src_utilization,
            "predicted_dst_utilization": chosen.predicted_dst_utilization,
            "predicted_imbalance": chosen.predicted_imbalance,
            "utilization_margin": decision.utilization_margin,
        }
        if ctx.mode == DRY_RUN:
            return stop(cl.SELECTED)
        return self._execute_selected(record, candidate, simulate_failure, stop)

    def _execute_selected(self, record: dict[str, Any], candidate: MigrationCandidate,
                          simulate_failure: str, stop: Callable[..., dict[str, Any]]) -> dict[str, Any]:
        """Revalidate on the latest state, then run the migration executor once."""
        started = time.perf_counter()
        latest = self.store.twin_state() or None
        forecast = self.forecast_payload()
        dst_risk: bool | None = None
        if latest is not None and forecast.get("status") == OK and (
            forecast.get("snapshot_id") == latest.snapshot_id
        ):
            dst_risk = self._risk(forecast).get(candidate.target_controller)
        last = self.transactions.last_finished_at()
        now = self.clock()
        elapsed = None if last is None else max(0.0, (now - last).total_seconds())
        problems: tuple[str, ...] = (INVALID_SNAPSHOT,)
        if latest is not None:
            problems = cl.revalidate(
                candidate, latest, self.store.role_matrix(), dst_risk,
                self.transactions.active_count() >= 1, elapsed,
                self.policy.cooldown_seconds.value,
            )
        record["timings_ms"]["revalidation"] = _ms(started)
        record["revalidation"] = {
            "snapshot_id": latest.snapshot_id if latest is not None else None,
            "passed": not problems, "reasons": list(problems),
        }
        if problems:
            return stop(cl.ABORT_STALE_DECISION, reasons={**record["reasons"],
                                                         **{p: 1 for p in problems}})
        if self.executor is None:
            return stop(cl.FAILED, "no migration executor configured")
        try:
            result = self.executor(candidate.switch_id, candidate.target_controller,
                                   simulate_failure)
        except Exception as exc:  # executor errors are recorded, never raised to the loop
            return stop(cl.FAILED, f"{type(exc).__name__}: {getattr(exc, 'detail', exc)}")
        tx = result.get("transaction") or {}
        duration = None
        if tx.get("started_at") and tx.get("finished_at"):
            finished = datetime.fromisoformat(tx["finished_at"])
            duration = (finished - datetime.fromisoformat(tx["started_at"])).total_seconds()
        status = str(result.get("status"))
        record["transaction"] = {
            "transaction_id": tx.get("transaction_id"), "status": status,
            "duration_s": duration, "failure_reason": result.get("failure_reason"),
            "simulate_failure": simulate_failure,
        }
        if status == cl.COMMITTED and tx.get("finished_at"):
            with self._state_lock:
                self._pending.append(cl.PostWindow(
                    record["decision_id"], candidate.source_controller,
                    candidate.target_controller, datetime.fromisoformat(tx["finished_at"]),
                    self.loop_config.post_settle_seconds,
                    self.loop_config.post_window_valid_snapshots,
                ))
        return stop(status if status in (cl.COMMITTED, cl.RESTORED) else cl.FAILED)
