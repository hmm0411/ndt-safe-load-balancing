"""Pre window rules: complete, Hard-Gate valid, time ordered, switch owned by the source."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.validator import validate_attempt
from src.experiments.migration.writer import write_attempt


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


if __name__ == "__main__":
    unittest.main()
