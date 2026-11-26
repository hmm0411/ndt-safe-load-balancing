"""campaign_report on a synthetic campaign whose answer is known."""
from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import campaign_report as cr

METHODS = ["full", "static", "threshold", "forecast_only", "ndt_no_feedback"]
SEEDS = range(9801, 9811)


def synthetic(path: Path, drop: int = 0) -> None:
    rng = random.Random(1)
    base = {"full": 2.0, "static": 30.0, "threshold": 12.0, "forecast_only": 8.0,
            "ndt_no_feedback": 2.5}
    rows = []
    for m in METHODS:
        for w in cr.WORKLOADS:
            for s in SEEDS:
                stable = w == "stable"
                m2 = 0.0 if stable else max(0.0, base[m] + rng.gauss(0, 1.0))
                rows.append({"method": m, "workload": w, "seed": s, "overload_time_s": m2,
                             "overload_exposure_us": m2 / 10, "max_u": 1 + m2 / 100,
                             "mean_imbalance": 0.1, "migrations": 0 if m == "static" else 3,
                             "unsafe_migrations": 0, "ping_pongs": 0,
                             "decision_p95_ms": "" if m == "static" else 40.0})
    rows = rows[drop:]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CampaignReportTest(unittest.TestCase):
    def test_full_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "ep.csv", Path(tmp) / "B"
            synthetic(src)
            self.assertEqual(cr.main(["--scenario", "B", "--episodes", str(src),
                                      "--out", str(out)]), 0)
            stats = json.loads((out / "stats_overload_time_s.json").read_text())
            self.assertTrue(stats["confirmatory"])
            static = stats["comparisons"]["static"]
            # stable: 10 tied pairs dropped by wilcox; 40 pairs left, all won by Full
            t = static["test"]
            self.assertEqual((t["n_pairs"], t["n_zero"], t["n_used"]), (50, 10, 40))
            self.assertLess(static["p_holm"], 1e-6)
            self.assertEqual(static["per_workload"]["stable"]["seeds_tied"], 10)
            self.assertEqual(static["per_workload"]["stable"]["seeds_won"], 0)
            self.assertFalse(json.loads((out / "stats_max_u.json").read_text())["confirmatory"])
            report = (out / "report.md").read_text()
            self.assertIn("Holm over the 4 comparisons", report)
            self.assertNotIn("Incomplete", report)
            for name in ("fig_B_overload_time", "fig_B_paired_improvement"):
                self.assertTrue((out / f"{name}.png").is_file())
                self.assertTrue((out / f"{name}.pdf").is_file())
            rows = list(csv.DictReader((out / "summary.csv").open()))
            self.assertEqual(len(rows), len(cr.METRICS) * 5 * 5)

    def test_partial_campaign_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "ep.csv", Path(tmp) / "B"
            synthetic(src, drop=3)
            self.assertEqual(cr.main(["--scenario", "B", "--episodes", str(src),
                                      "--out", str(out)]), 1)
            self.assertIn("Incomplete campaign: 3 episodes missing",
                          (out / "report.md").read_text())

    def test_c_and_b2_are_descriptive(self) -> None:
        self.assertFalse(cr.SCENARIOS["C"]["confirmatory"])
        self.assertFalse(cr.SCENARIOS["B2"]["confirmatory"])
        self.assertEqual(cr.SCENARIOS["B2"]["others"], ["static", "threshold"])
        self.assertEqual(cr._value(" nan "), None)
        self.assertEqual(cr._value("1.5"), 1.5)


if __name__ == "__main__":
    unittest.main()
