"""predict_batch gives the same answers as predict() per action (Week 13, Day 2)."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.experiments import benchmark_outcome_inference as bench
from src.experiments.migration.schema import MigrationAction
from src.twin.outcome import predictor as p

ARTIFACT = bench.ROOT / "data" / "models" / "outcome" / "post-migration-v1"
CAPACITY = bench.ROOT / "data" / "benchmarks" / "controller_capacity.json"
NUMBERS = ("predicted_src_utilization", "predicted_dst_utilization",
           "predicted_src_response_p95_ms", "predicted_dst_response_p95_ms",
           "predicted_worst_response_p95_ms", "predicted_imbalance",
           "dst_utilization_margin", "response_margin_ms")


def _migrate(switch: str, source: str = "c1", target: str = "c2") -> MigrationAction:
    return MigrationAction(action_type="MIGRATE", source_controller=source,
                           target_controller=target, switch_id=switch)


class PredictBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor, self.state = bench.build_predictor(ARTIFACT, CAPACITY)

    def assert_same(self, one: p.PostMigrationOutcomePrediction,
                    two: p.PostMigrationOutcomePrediction) -> None:
        for name in ("switch_id", "source_controller", "target_controller", "status", "reason"):
            self.assertEqual(getattr(one, name), getattr(two, name), name)
        for name in NUMBERS:
            a, b = getattr(one, name), getattr(two, name)
            if a is None or b is None:
                self.assertEqual(a, b, name)
            else:
                self.assertAlmostEqual(a, b, places=12, msg=name)

    def test_ten_candidates_match_one_by_one(self) -> None:
        actions = bench._actions(10)
        batch = self.predictor.predict_batch(self.state, actions, observe=False)
        single = [self.predictor.predict(self.state, a, observe=False) for a in actions]
        self.assertEqual(len(batch), 10)
        for one, two in zip(batch, single):
            self.assert_same(one, two)
            self.assertEqual(one.status, p.READY)
            self.assertGreater(one.inference_time_ms or 0.0, 0.0)
        self.assertEqual([b.switch_id for b in batch], [f"s{k}" for k in range(1, 11)])

    def test_not_ready_actions_keep_their_place_and_reason(self) -> None:
        no_op = MigrationAction(action_type="NO_OP", source_controller="c1",
                                target_controller="c2")
        actions = [_migrate("s1"), _migrate("s15"), no_op, _migrate("s2")]
        batch = self.predictor.predict_batch(self.state, actions, observe=False)
        single = [self.predictor.predict(self.state, a, observe=False) for a in actions]
        self.assertEqual([b.status for b in batch], [p.READY, p.NOT_READY, p.NOT_READY, p.READY])
        self.assertEqual(batch[1].reason, p.SWITCH_NOT_OWNED_BY_SOURCE)
        self.assertEqual(batch[2].reason, p.INVALID_ACTION)
        for one, two in zip(batch, single):
            self.assert_same(one, two)

    def test_invalid_snapshot_and_empty_batch(self) -> None:
        bad = dict(self.state)
        bad["quality"] = {**self.state["quality"], "valid": False}
        out = self.predictor.predict_batch(bad, bench._actions(3), observe=False)
        self.assertEqual({o.reason for o in out}, {p.INVALID_SNAPSHOT})
        self.assertEqual(self.predictor.predict_batch(self.state, [], observe=False), [])

    def test_missing_artifact_is_reported_for_every_valid_action(self) -> None:
        fresh = p.PostMigrationOutcomePredictor(Path("/nonexistent"), "cap@x")
        for state in bench._states():
            fresh.observe(state)
        out = fresh.predict_batch(self.state, [_migrate("s1"), _migrate("s15")], observe=False)
        self.assertEqual([o.reason for o in out],
                         [p.ARTIFACT_MISSING, p.SWITCH_NOT_OWNED_BY_SOURCE])


if __name__ == "__main__":
    unittest.main()
