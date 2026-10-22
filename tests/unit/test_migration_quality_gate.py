"""Quality fields: Quality Score present, scores in [0, 1], capacity present, no NaN."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.writer import write_attempt


class QualityGateTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def report_for(self, before: list) -> dict:
        inputs = fx.attempt_inputs("committed")
        inputs["before_raw"] = before
        return write_attempt(self.root, **inputs).validation

    def test_missing_quality_score(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5, quality_score=None)
        report = self.report_for(before)
        self.assertFalse(report["valid"])
        self.assertIn("before[0]: quality_score missing", report["errors"])

    def test_consistency_out_of_range(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5)
        before[2]["quality"]["consistency_score"] = 1.2
        report = self.report_for(before)
        self.assertIn("before[2]: consistency_score out of [0, 1]", report["errors"])

    def test_missing_capacity(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5)
        before[0]["controllers"][1]["safe_capacity_pps"] = 0.0
        report = self.report_for(before)
        self.assertIn("before[0]: safe capacity missing for c2", report["errors"])

    def test_nan_utilization(self) -> None:
        before = fx.make_window([{"c1": 0.80, "c2": 0.45}] * 5)
        before[3]["controllers"][0]["utilization"] = float("nan")
        report = self.report_for(before)
        self.assertFalse(report["valid"])
        self.assertTrue(any(e.startswith("before[3]") for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
