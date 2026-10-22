"""RESTORED / FAILED / SKIPPED attempts are kept but never count as a migration."""
from __future__ import annotations

import unittest

from src.experiments.migration import fixtures as fx
from src.experiments.migration.metrics import compute_outcome
from src.experiments.migration.schema import (
    MigrationAction,
    MigrationTransactionView,
    StateSnapshotView,
)

BEFORE_U = [{"c1": 0.80, "c2": 0.45}] * 5


def window(utils, start=0):
    return [StateSnapshotView.from_twin_state(x) for x in fx.make_window(utils, start)]


class RestoredOutcomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.before = window(BEFORE_U)
        self.after = window(BEFORE_U, start=10)  # rollback: ownership back to c1

    def test_restored_is_not_eligible(self) -> None:
        action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1", failure_injected=True)
        tx = MigrationTransactionView.from_response(fx.restored_response())
        out = compute_outcome(self.before, self.after, action, tx)
        self.assertFalse(out.training_eligible)
        self.assertEqual(out.transaction_status, "RESTORED")
        self.assertIn("transaction_restored", out.ineligible_reasons)
        self.assertIn("failure_injected", out.ineligible_reasons)
        self.assertAlmostEqual(out.delta_src_u_mean or 0, 0.0)  # observed, not a migration effect

    def test_failed_is_not_eligible(self) -> None:
        action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1")
        tx = MigrationTransactionView.failed(fx.failed_detail())
        out = compute_outcome(self.before, self.after, action, tx)
        self.assertFalse(out.training_eligible)
        self.assertIn("transaction_failed", out.ineligible_reasons)
        self.assertEqual(tx.transaction_id, "tx-0123456789ab")

    def test_skipped_without_post_window(self) -> None:
        action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1")
        tx = MigrationTransactionView.skipped("source_not_master")
        out = compute_outcome(self.before, [], action, tx)
        self.assertFalse(out.training_eligible)
        self.assertIn("transaction_skipped", out.ineligible_reasons)
        self.assertIsNone(out.src_u_after_mean)
        self.assertIsNone(out.delta_imbalance_mean)


if __name__ == "__main__":
    unittest.main()
