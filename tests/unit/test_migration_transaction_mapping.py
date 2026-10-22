"""transaction.json -> status, and ownership after COMMITTED / RESTORED."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.writer import raw_transaction, transaction_from_raw, write_attempt


class MappingTest(unittest.TestCase):
    def test_http_mapping(self) -> None:
        cases = {
            "COMMITTED": raw_transaction(200, fx.committed_response()),
            "RESTORED": raw_transaction(200, fx.restored_response()),
            "FAILED": raw_transaction(500, {"detail": fx.failed_detail()}),
        }
        for status, raw in cases.items():
            view = transaction_from_raw(raw)
            assert view is not None
            self.assertEqual(view.status, status)
            self.assertEqual(view.transaction_id, "tx-0123456789ab")
        for code in (400, 403, 404, 409):
            view = transaction_from_raw(raw_transaction(code, {"detail": "x"}))
            assert view is not None
            self.assertEqual((view.status, view.skip_reason), ("SKIPPED", f"http_{code}"))
        skipped = transaction_from_raw(raw_transaction(skip_reason="source_not_master"))
        assert skipped is not None
        self.assertEqual(skipped.skip_reason, "source_not_master")
        self.assertIsNone(transaction_from_raw(raw_transaction()))

    def test_500_without_transaction_detail(self) -> None:
        view = transaction_from_raw(raw_transaction(500, {"detail": "boom"}))
        assert view is not None
        self.assertEqual(view.status, "FAILED")
        self.assertIsNone(view.transaction_id)


class OwnershipAfterTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def test_committed_but_switch_still_on_source(self) -> None:
        inputs = fx.attempt_inputs("committed")
        inputs["after_raw"] = fx.make_window([{"c1": 0.72, "c2": 0.53}] * 5, 10)
        from src.experiments.migration.metrics import compute_outcome, summarize_quality
        from src.experiments.migration.schema import StateSnapshotView
        before = [StateSnapshotView.from_twin_state(x) for x in inputs["before_raw"]]
        after = [StateSnapshotView.from_twin_state(x) for x in inputs["after_raw"]]
        tx = transaction_from_raw(inputs["transaction_raw"])
        inputs["outcome"] = compute_outcome(before, after, inputs["action"], tx)
        inputs["quality"] = summarize_quality(before, after)
        report = write_attempt(self.root, **inputs).validation
        self.assertFalse(report["valid"])
        self.assertIn("COMMITTED: s1 should be owned by c2 after", report["errors"])

    def test_restored_keeps_the_source(self) -> None:
        report = write_attempt(self.root, **fx.attempt_inputs("restored")).validation
        self.assertTrue(report["valid"], report["errors"])

    def test_failed_is_kept_with_a_warning(self) -> None:
        report = write_attempt(self.root, **fx.attempt_inputs("failed")).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])
        self.assertTrue(any("FAILED" in w for w in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
