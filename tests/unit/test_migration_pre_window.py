"""Pre window rules: complete, Hard-Gate valid, time ordered, switch owned by the source."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.metrics import compute_outcome, summarize_quality
from src.experiments.migration.schema import StateSnapshotView
from src.experiments.migration.validator import validate_attempt
from src.experiments.migration.writer import transaction_from_raw, write_attempt


def with_before(kind: str, before: list) -> dict:
    """attempt_inputs with another pre window and the outcome recomputed from it."""
    inputs = fx.attempt_inputs(kind)
    inputs["before_raw"] = before
    views = [StateSnapshotView.from_twin_state(x) for x in before]
    after = [StateSnapshotView.from_twin_state(x) for x in inputs["after_raw"]]
    tx = transaction_from_raw(inputs["transaction_raw"])
    inputs["outcome"] = compute_outcome(views, after, inputs["action"], tx)
    inputs["quality"] = summarize_quality(views, after)
    return inputs


class PreWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def write(self, **changes: object) -> dict:
        inputs = fx.attempt_inputs("committed")
        inputs.update(changes)
        return write_attempt(self.root, **inputs).validation

    def test_short_pre_window_is_invalid(self) -> None:
        inputs = fx.attempt_inputs("committed")
        report = self.write(before_raw=inputs["before_raw"][:4])
        self.assertFalse(report["valid"])
        self.assertTrue(any("before window has 4" in e for e in report["errors"]))

    def test_invalid_snapshot_in_pre_window(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 4) + [
            fx.make_twin_state({"c1": 0.80, "c2": 0.45}, index=4, valid=False)
        ]
        report = self.write(before_raw=before)
        self.assertIn("before window contains a snapshot that failed the Hard Gate",
                      report["errors"])

    def test_samples_out_of_order(self) -> None:
        before = list(reversed(fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5)))
        report = self.write(before_raw=before)
        self.assertIn("before: created_at is not strictly increasing", report["errors"])

    def test_repeated_sample_is_rejected(self) -> None:
        window = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 4)
        report = self.write(before_raw=window + [window[-1]])
        self.assertTrue(any("duplicate snapshot_id" in e for e in report["errors"]))

    def test_switch_not_owned_by_source(self) -> None:
        owners = fx.default_ownership()
        owners["s1"] = "c2"
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5, ownership=owners)
        report = self.write(before_raw=before)
        self.assertIn("before window: s1 is not owned by the source", report["errors"])

    def test_validator_reads_the_directory_again(self) -> None:
        path = write_attempt(self.root, **fx.attempt_inputs("committed")).path
        self.assertTrue(validate_attempt(path)["valid"])


class NotExecutedTest(unittest.TestCase):
    """SKIPPED MIGRATE and NO_OP never acted: pre-window problems are warnings."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def test_skipped_with_short_pre_window(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 3)
        report = write_attempt(self.root, **with_before("skipped", before)).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])
        self.assertIn("before window has 3 samples, expected 5", report["warnings"])

    def test_skipped_because_source_is_not_owner(self) -> None:
        owners = fx.default_ownership()
        owners["s1"] = "c2"
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5, ownership=owners)
        report = write_attempt(self.root, **with_before("skipped", before)).validation
        self.assertTrue(report["valid"], report["errors"])

    def test_no_op_with_short_pre_window(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 2)
        report = write_attempt(self.root, **with_before("no_op", before)).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])

    def test_executed_migrate_still_needs_the_full_window(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 3)
        report = write_attempt(self.root, **with_before("committed", before)).validation
        self.assertFalse(report["valid"])
        self.assertIn("before window has 3 samples, expected 5", report["errors"])


if __name__ == "__main__":
    unittest.main()
