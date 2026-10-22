"""Post window: a short or invalid after window is data (valid), never training data."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import fixtures as fx
from src.experiments.migration.writer import write_attempt


class PostWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        self.addCleanup(fx.unlock_tree, self.root)

    def test_post_window_timeout(self) -> None:
        report = write_attempt(self.root, **fx.attempt_inputs("committed", after_count=3)).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])
        self.assertIn("after window has 3 of 5 samples", report["warnings"])

    def test_skipped_attempt_without_after_window(self) -> None:
        report = write_attempt(self.root, **fx.attempt_inputs("skipped")).validation
        self.assertTrue(report["valid"], report["errors"])
        self.assertFalse(report["training_eligible"])
        self.assertEqual(report["transaction_status"], "SKIPPED")

    def test_after_window_before_pre_window_is_invalid(self) -> None:
        inputs = fx.attempt_inputs("committed")
        moved = fx.default_ownership()
        moved["s1"] = "c2"
        inputs["after_raw"] = fx.make_window([{"c1": 0.72, "c2": 0.53}] * 5, 0,
                                             ownership=moved, ownership_version=2)
        report = write_attempt(self.root, **inputs).validation
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
