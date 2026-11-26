"""One-sided coverage report with companions, on data whose answer is known."""
from __future__ import annotations

import unittest

import numpy as np

from scripts.analysis import margin_coverage_report as mc


class CoverageReportTest(unittest.TestCase):
    def test_known_values(self) -> None:
        y = np.array([1.0, 0.5, 0.9, 0.2, 1.1, 0.85])
        pred = np.array([0.8, 0.5, 1.2, 0.2, 1.05, 0.9])   # e: .2 0 -.3 0 .05 -.05
        m = np.full(6, 0.1)
        r = mc.report(y, pred, m, ["a", "a", "b", "b", "c", "c"], n_boot=200)
        self.assertAlmostEqual(r["coverage_one_sided"], 5 / 6)
        self.assertAlmostEqual(r["coverage_two_sided_reference"], 4 / 6)
        # decision region U >= 0.8: 1.0 (miss), 0.9, 1.1, 0.85 -> 3/4
        self.assertAlmostEqual(r["coverage_one_sided_given_U_ge_0.8"], 0.75)
        self.assertEqual(r["samples_U_ge_0.8"], 4)
        ci = r["coverage_one_sided_ci95_by_run"]
        assert ci is not None
        self.assertLessEqual(ci[0], r["coverage_one_sided"])
        self.assertGreaterEqual(ci[1], r["coverage_one_sided"])
        self.assertEqual(r["overload"]["actual_overload"], 2.0)

    def test_one_run_has_no_interval_and_bad_input_raises(self) -> None:
        self.assertIsNone(mc.bootstrap_by_run(np.zeros(3), np.ones(3), ["a", "a", "a"]))
        with self.assertRaises(ValueError):
            mc.report(np.zeros(2), np.zeros(3), np.zeros(2), ["a", "b"])

    def test_grouped_per_workload(self) -> None:
        rows = [{"future_utilization": u, "controller_id": "c1", "run_id": f"r{i % 2}",
                 "workload_type": w} for i, (u, w) in
                enumerate([(0.5, "stable"), (0.6, "stable"), (1.2, "burst"), (1.1, "burst")])]
        out = mc.grouped(rows, np.array([0.5, 0.6, 1.0, 1.1]), {"global": 0.1, "controllers": {}},
                         "workload_type", n_boot=100)
        self.assertEqual(set(out), {"all", "stable", "burst"})
        self.assertEqual(out["burst"]["coverage_one_sided"], 0.5)


if __name__ == "__main__":
    unittest.main()
