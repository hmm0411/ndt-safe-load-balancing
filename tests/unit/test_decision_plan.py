"""plan / execute of the DecisionService (Week 13, Day 5): states, safety of execution."""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any

from decision_testdata import NOW, make_twin
from src.orchestrator.decision_service import DecisionService
from src.orchestrator.transaction_manager import TransactionManager
from test_decision_service import FakePredictor, FakeStore, forecast

from scripts.decision import analyze_closed_loop as acl
from src.twin.decision import closed_loop as cl


class Executor:
    def __init__(self, status: str = "COMMITTED", error: Exception | None = None) -> None:
        self.status = status
        self.error = error
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, switch_id: str, target: str, failure: str) -> dict[str, Any]:
        self.calls.append((switch_id, target, failure))
        if self.error is not None:
            raise self.error
        return {"status": self.status, "transaction": {
            "transaction_id": "tx-1", "started_at": NOW.isoformat(),
            "finished_at": (NOW + timedelta(seconds=2)).isoformat()}}


def make(executor: Any = None, **over: Any) -> tuple[DecisionService, Any]:
    twin = over.pop("twin", make_twin())
    body = over.pop("fc", forecast())
    store: Any = FakeStore(twin)
    config = dataclasses.replace(cl.load_config(), post_settle_seconds=3,
                                 post_window_valid_snapshots=2, **over.pop("config", {}))
    service = DecisionService(
        store, lambda: body, over.pop("predictor", FakePredictor()),
        over.pop("transactions", TransactionManager()), clock=lambda: NOW,
        loop_config=config, executor=executor, **over,
    )
    return service, store


class PlanTest(unittest.TestCase):
    def test_dry_run_selects_and_never_calls_the_executor(self) -> None:
        executor = Executor()
        service, _ = make(executor)
        out = service.plan("gradual")
        self.assertEqual(out["status"], "SELECTED")
        self.assertEqual(out["mode"], "dry_run")
        self.assertEqual(out["funnel"], {"raw": 4, "eligible": 4, "evaluated": 4, "safe": 4})
        self.assertEqual(out["selected"]["candidate_id"], out["ranking"]["selected_candidate_id"])
        self.assertEqual(executor.calls, [])
        self.assertIsNone(out["transaction"])
        for stage in ("forecast", "risk", "candidate", "outcome", "safety", "ranking"):
            self.assertIn(stage, out["timings_ms"])
        json.dumps(out)

    def test_terminal_states_before_any_candidate(self) -> None:
        self.assertEqual(make(fc=forecast(c1=False))[0].plan()["status"], "NO_RISK")
        self.assertEqual(make(twin=make_twin(valid=False))[0].plan()["status"],
                         "INVALID_SNAPSHOT")
        self.assertEqual(make(fc={"status": "NOT_READY"})[0].plan()["status"], "NOT_READY")
        self.assertEqual(make(twin=None)[0].plan()["reason"], "NO_TWIN_STATE")
        off, _ = make(config={"enabled": False})
        self.assertEqual(off.plan()["status"], "KILL_SWITCH")

    def test_unsafe_destination_gives_no_safe_action(self) -> None:
        out = make(predictor=FakePredictor(dst=0.99))[0].plan("no_safe_action")
        self.assertEqual(out["status"], "NO_SAFE_ACTION")
        self.assertEqual(out["reasons"].get("DST_UTILIZATION_UNSAFE"), 4)
        self.assertIsNone(out["selected"])

    def test_destination_at_risk_is_rejected_by_the_gate(self) -> None:
        out = make(fc=forecast(c2=True))[0].plan("dst_near_capacity")
        self.assertEqual(out["status"], "NO_SAFE_ACTION")
        self.assertGreater(out["reasons"].get("DESTINATION_ALREADY_AT_RISK", 0), 0)

    def test_cooldown_and_active_transaction_defer(self) -> None:
        manager = TransactionManager()
        tx = manager.create("s9", "c1", "c2")
        busy = make(transactions=manager)[0].plan("conflict")
        self.assertEqual(busy["status"], "DEFERRED")
        self.assertEqual(busy["reasons"], {"ACTIVE_TRANSACTION_CONFLICT": 1})
        # also without any risk and on an invalid snapshot: a transaction in flight wins
        quiet = make(transactions=manager, fc=forecast(c1=False))[0].plan("conflict")
        self.assertEqual(quiet["status"], "DEFERRED")
        invalid = make(transactions=manager, twin=make_twin(valid=False))[0].plan("conflict")
        self.assertEqual(invalid["status"], "DEFERRED")
        manager.finish(tx)
        tx.finished_at = NOW - timedelta(seconds=5)
        cooling = make(transactions=manager)[0].plan("cooldown")
        self.assertEqual(cooling["status"], "DEFERRED")
        self.assertEqual(cooling["reasons"], {"COOLDOWN_ACTIVE": 1})

    def test_kill_switch_at_runtime(self) -> None:
        service, store = make(Executor())
        service.set_enabled(False)
        self.assertEqual(service.execute()["status"], "KILL_SWITCH")
        service.set_enabled(True)
        store.twin = dataclasses.replace(store.twin, snapshot_id="snap-2")
        service.forecast_payload = lambda: forecast(snapshot_id="snap-2")
        self.assertEqual(service.execute()["status"], "COMMITTED")


class ExecuteTest(unittest.TestCase):
    def test_commit_then_observed_outcome_and_idempotent_replay(self) -> None:
        executor = Executor()
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = make(executor, record_root=Path(tmp))
            out = service.execute("gradual")
            self.assertEqual(out["status"], "COMMITTED")
            self.assertEqual(out["transaction"]["duration_s"], 2.0)
            self.assertTrue(out["revalidation"]["passed"])
            self.assertEqual(len(executor.calls), 1)
            again = service.execute("gradual")
            self.assertTrue(again["idempotent_replay"])
            self.assertEqual(len(executor.calls), 1)  # same snapshot: never a second migration
            path = Path(tmp) / out["decision_id"] / "record.json"
            self.assertIsNone(json.loads(path.read_text())["actual_post"])
            src, dst = out["selected"]["source_controller"], out["selected"]["target_controller"]
            for k, (a, b) in enumerate(((0.9, 0.5), (0.7, 0.6), (0.8, 0.5))):
                service.on_twin_state({
                    "snapshot_id": f"post-{k}", "quality": {"valid": True},
                    "created_at": (NOW + timedelta(seconds=2 + 3 * k)).isoformat(),
                    "controllers": [{"controller_id": src, "utilization": a},
                                    {"controller_id": dst, "utilization": b}]})
            saved = json.loads(path.read_text())
            self.assertAlmostEqual(saved["actual_post"]["u_src"], 0.75)
            self.assertAlmostEqual(saved["actual_post"]["u_dst"], 0.55)
            report = acl.analyze(acl.load_records(Path(tmp)))  # the record meets the contract
            self.assertEqual(report["scenario_gates"]["gradual"]["result"], "PASS")
            self.assertEqual(report["prediction_errors"]["n"], 1)

    def test_rollback_and_executor_errors_are_recorded(self) -> None:
        restored = make(Executor("RESTORED"))[0].execute("verification_failure", "flow_mod")
        self.assertEqual(restored["status"], "RESTORED")
        self.assertEqual(restored["transaction"]["simulate_failure"], "flow_mod")
        failed = make(Executor(error=RuntimeError("boom")))[0].execute()
        self.assertEqual(failed["status"], "FAILED")
        self.assertIn("boom", failed["detail"])
        self.assertEqual(make()[0].execute()["status"], "FAILED")  # no executor wired

    def test_stale_decision_is_aborted_before_the_executor(self) -> None:
        executor = Executor()
        service, store = make(executor)
        original = store.twin_state

        def moved() -> Any:  # after the decision pinned its snapshot, the switch moved
            if not executor.calls and service.transactions.last_finished_at() is None:
                store.twin_state = lambda: make_twin(owners={
                    "s1": "c2", "s2": "c2", "s3": "c2", "s4": "c2", "s5": "c2"})
            return original()

        store.twin_state = moved
        out = service.execute("gradual")
        self.assertEqual(out["status"], "ABORT_STALE_DECISION")
        self.assertIn("OWNER_MISMATCH", out["revalidation"]["reasons"])
        self.assertEqual(executor.calls, [])


if __name__ == "__main__":
    unittest.main()
