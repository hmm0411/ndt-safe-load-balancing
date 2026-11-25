"""Campaign methods: config, the baselines' heuristic and the service paths they take."""
from __future__ import annotations

import dataclasses
import os
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from decision_testdata import NOW, make_twin
from src.schemas.decision import MigrationCandidate
from src.twin.decision import heuristic
from src.twin.decision.methods import CONFIG_PATH, load_method, parse_methods
from test_decision_plan import Executor, make
from test_decision_service import FakePredictor, forecast

PROTOCOL = {"static", "threshold", "forecast_only", "ndt_no_feedback", "full",
            "full_analytical", "full_zero_margin", "full_random_rank", "full_no_cooldown"}


def cand(switch: str, src: str, dst: str, u_src: float, u_dst: float,
         share: float) -> MigrationCandidate:
    return MigrationCandidate("snap-1", switch, src, dst, u_src, u_dst, share, NOW)


class MethodsConfigTest(unittest.TestCase):
    def test_every_protocol_method_is_defined(self) -> None:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        methods = parse_methods(data)
        self.assertEqual(set(methods), PROTOCOL)
        self.assertEqual(methods["threshold"].trigger_utilization, 0.8)   # protocol 5, 24/11
        self.assertEqual(methods["full"].margins, "adaptive")
        self.assertEqual(methods["ndt_no_feedback"].margins, "fixed")
        # an ablation differs from Full in exactly one field
        full = methods["full"].to_dict()
        for name in ("full_analytical", "full_zero_margin", "full_random_rank",
                     "full_no_cooldown"):
            diff = {k for k, v in methods[name].to_dict().items() if full[k] != v} - {"name"}
            self.assertEqual(len(diff), 1, name)

    def test_selection_by_env_and_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NDT_DECISION_METHOD", None)
            self.assertEqual(load_method().name, "ndt_no_feedback")
        with mock.patch.dict(os.environ, {"NDT_DECISION_METHOD": "threshold"}):
            self.assertEqual(load_method().name, "threshold")
        with self.assertRaises(ValueError):
            load_method("oracle")

    def test_invalid_specs(self) -> None:
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        bad = dict(data, methods={**data["methods"], "x": {
            **data["methods"]["threshold"], "margins": "adaptive"}})
        with self.assertRaises(ValueError):
            parse_methods(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            path.write_text(yaml.safe_dump(dict(data, methods={"full": data["methods"]["full"]})))
            with self.assertRaises(ValueError):          # ndt_no_feedback must exist
                load_method("full", path)


class HeuristicTest(unittest.TestCase):
    CAP = {"c1": 900.0, "c2": 900.0, "c3": 450.0}

    def test_heaviest_switch_of_the_most_loaded_source_to_the_lightest_destination(self) -> None:
        cands = [cand("s1", "c1", "c2", 0.95, 0.5, 0.1), cand("s1", "c1", "c3", 0.95, 0.3, 0.1),
                 cand("s2", "c1", "c2", 0.95, 0.5, 0.3), cand("s2", "c1", "c3", 0.95, 0.3, 0.3),
                 cand("s9", "c4", "c2", 0.85, 0.5, 0.9)]
        choice = heuristic.choose(cands, {**self.CAP, "c4": 900.0}, room_limit=1.0)
        assert choice.candidate is not None
        self.assertEqual((choice.candidate.switch_id, choice.candidate.target_controller),
                         ("s2", "c3"))
        # c3 has half the capacity: 0.3 * 0.95 * 900 / 450 = 0.57 added
        self.assertAlmostEqual(choice.predicted_dst_utilization or 0, 0.3 + 0.57)
        self.assertAlmostEqual(choice.predicted_src_utilization or 0, 0.95 * 0.7)

    def test_no_room_and_no_candidate(self) -> None:
        tight = heuristic.choose([cand("s1", "c1", "c2", 0.95, 0.9, 0.3)], self.CAP, 1.0)
        self.assertIsNone(tight.candidate)
        self.assertEqual(tight.reason, heuristic.NO_ROOM)
        self.assertEqual(heuristic.choose([], self.CAP, 1.0).reason, heuristic.NO_CANDIDATE)


class ServiceMethodTest(unittest.TestCase):
    def test_static_never_acts(self) -> None:
        service, _ = make(Executor(), method=load_method("static"))
        out = service.execute("burst")
        self.assertEqual((out["status"], out["method"]), ("NO_RISK", "static"))

    def test_threshold_uses_measured_u_and_no_forecast_or_predictor(self) -> None:
        predictor = FakePredictor()
        executor = Executor()
        service, _ = make(executor, fc={"status": "NOT_READY"}, predictor=predictor,
                          method=load_method("threshold"))
        out = service.execute("gradual")
        self.assertEqual(out["status"], "COMMITTED")
        self.assertEqual(out["at_risk"], ["c1"])            # 0.95 >= 0.8, c2 0.40 is not
        self.assertEqual(out["selected"]["selector"], "heuristic")
        self.assertEqual(predictor.calls, 0)
        self.assertEqual(len(executor.calls), 1)
        quiet, _ = make(twin=make_twin(utilization={"c1": 0.7, "c2": 0.4}),
                        method=load_method("threshold"))
        self.assertEqual(quiet.plan()["status"], "NO_RISK")

    def test_forecast_only_takes_the_forecast_trigger_and_the_heuristic(self) -> None:
        predictor = FakePredictor(dst=0.99)                  # the gate would reject this
        service, _ = make(predictor=predictor, method=load_method("forecast_only"))
        out = service.plan("gradual")
        self.assertEqual((out["status"], predictor.calls), ("SELECTED", 0))
        self.assertEqual(make(fc=forecast(c1=False), method=load_method("forecast_only"))[0]
                         .plan()["status"], "NO_RISK")

    def test_heuristic_without_room(self) -> None:
        twin = make_twin(utilization={"c1": 0.95, "c2": 0.97})
        out = make(twin=twin, method=load_method("threshold"))[0].plan()
        self.assertEqual(out["status"], "NO_SAFE_ACTION")
        self.assertIn("NO_ROOM", out["detail"])

    def test_zero_margin_reaches_the_gate(self) -> None:
        # U_dst 0.97 + m_U 0.037 > 1 is unsafe; with the margin at 0 it passes
        fixed = make(predictor=FakePredictor(dst=0.97))[0].plan()
        zero = make(predictor=FakePredictor(dst=0.97),
                    method=load_method("full_zero_margin"))[0].plan()
        self.assertEqual((fixed["status"], fixed["funnel"]["safe"]), ("NO_SAFE_ACTION", 0))
        self.assertEqual(zero["funnel"]["safe"], 4)     # all safe; ranking then finds no gain

    def test_random_ranking_is_seeded(self) -> None:
        picks = {make(method=load_method("full_random_rank"), seed=s)[0].plan()
                 ["selected"]["candidate_id"] for s in range(12)}
        self.assertGreater(len(picks), 1)
        a = make(method=load_method("full_random_rank"), seed=5)[0].plan()["selected"]
        b = make(method=load_method("full_random_rank"), seed=5)[0].plan()["selected"]
        self.assertEqual(a, b)

    def test_no_cooldown_ablation(self) -> None:
        from src.orchestrator.transaction_manager import TransactionManager
        manager = TransactionManager()
        tx = manager.create("s9", "c1", "c2")
        manager.finish(tx)
        tx.finished_at = NOW - timedelta(seconds=5)
        cooling = make(transactions=manager)[0].plan()
        free = make(transactions=manager, method=load_method("full_no_cooldown"))[0]
        self.assertEqual(cooling["status"], "DEFERRED")
        self.assertEqual(free.plan()["status"], "SELECTED")
        self.assertEqual(free.policy.cooldown_seconds.value, 0.0)

    def test_full_feeds_the_observed_outcome_back_into_m_u(self) -> None:
        service, store = make(Executor(), method=load_method("full"))
        out = service.execute("gradual")
        self.assertEqual(out["status"], "COMMITTED")
        later = dataclasses.replace(store.twin, created_at=NOW + timedelta(seconds=10))
        for i in range(2):
            snap = dataclasses.replace(later, snapshot_id=f"post-{i}",
                                       created_at=later.created_at + timedelta(seconds=i))
            service.on_twin_state(snap.to_dict())
        record: Any = service._cache[out["decision_id"]]
        self.assertIsNotNone(record["actual_post"])
        fb = record["margin_feedback"]
        self.assertEqual(fb["mode"], "adaptive")
        self.assertEqual(fb["margin"]["updates"], 1)


if __name__ == "__main__":
    unittest.main()
