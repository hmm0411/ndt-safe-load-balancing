"""compute_outcome for a COMMITTED migration and for NO_OP controls."""
from __future__ import annotations

import unittest

from src.experiments.migration import fixtures as fx
from src.experiments.migration.metrics import compute_outcome
from src.experiments.migration.schema import (
    MigrationAction,
    MigrationTransactionView,
    StateSnapshotView,
)

BEFORE_U = [
    {"c1": 0.78, "c2": 0.45},
    {"c1": 0.79, "c2": 0.45},
    {"c1": 0.80, "c2": 0.44},
    {"c1": 0.81, "c2": 0.46},
    {"c1": 0.82, "c2": 0.45},
]
AFTER_U = [{"c1": 0.72, "c2": 0.53}] * 5


def window(utils, start=0, **kwargs):
    return [StateSnapshotView.from_twin_state(x) for x in fx.make_window(utils, start, **kwargs)]


def after_ownership():
    owners = fx.default_ownership()
    owners["s1"] = "c2"
    return owners


class CommittedOutcomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.before = window(BEFORE_U, hot_switch=("s1", 0.10))
        self.after = window(AFTER_U, start=10, ownership=after_ownership(), ownership_version=2)
        self.action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1", selection_rule="random")
        self.tx = MigrationTransactionView.from_response(fx.committed_response())

    def test_deltas_and_imbalance(self) -> None:
        out = compute_outcome(self.before, self.after, self.action, self.tx)
        self.assertTrue(out.training_eligible)
        self.assertEqual(out.ineligible_reasons, ())
        self.assertEqual(out.transaction_status, "COMMITTED")
        self.assertAlmostEqual(out.src_u_before_mean or 0, 0.80)
        self.assertAlmostEqual(out.delta_src_u_mean or 0, -0.08)
        self.assertAlmostEqual(out.delta_dst_u_mean or 0, 0.08)
        self.assertAlmostEqual(out.imbalance_before_mean or 0, 0.175)
        self.assertAlmostEqual(out.imbalance_after_mean or 0, 0.095)
        self.assertLess(out.delta_imbalance_mean or 0, 0.0)

    def test_last_and_mean_aggregations_are_both_kept(self) -> None:
        out = compute_outcome(self.before, self.after, self.action, self.tx)
        self.assertAlmostEqual(out.src_u_before_last or 0, 0.82)
        self.assertAlmostEqual(out.delta_src_u_last or 0, -0.10)
        self.assertNotAlmostEqual(out.delta_src_u_last or 0, out.delta_src_u_mean or 0)

    def test_switch_load_is_read_from_the_source(self) -> None:
        out = compute_outcome(self.before, self.after, self.action, self.tx)
        self.assertAlmostEqual(out.switch_share_before_mean or 0, 0.10)
        self.assertAlmostEqual(out.switch_rate_before_mean or 0, 0.80 * fx.SAFE_CAPACITY * 0.10)

    def test_short_post_window_is_not_eligible(self) -> None:
        out = compute_outcome(self.before, self.after[:3], self.action, self.tx)
        self.assertFalse(out.training_eligible)
        self.assertIn("post_window_incomplete", out.ineligible_reasons)
        self.assertIsNotNone(out.delta_src_u_mean)  # observed numbers are still kept

    def test_invalid_snapshot_in_window_is_not_eligible(self) -> None:
        before = self.before[:4] + window([BEFORE_U[4]], start=4, valid=False)
        out = compute_outcome(before, self.after, self.action, self.tx)
        self.assertIn("pre_window_invalid", out.ineligible_reasons)

    def test_empty_before_window_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_outcome([], self.after, self.action, self.tx)

    def test_no_op_control(self) -> None:
        action = MigrationAction("NO_OP", "c1", "c2")
        out = compute_outcome(self.before, window(BEFORE_U, start=10), action, None)
        self.assertTrue(out.training_eligible)
        self.assertIsNone(out.transaction_status)
        self.assertIsNone(out.switch_rate_before_mean)
        with self.assertRaises(ValueError):
            compute_outcome(self.before, self.after, action, self.tx)


if __name__ == "__main__":
    unittest.main()
