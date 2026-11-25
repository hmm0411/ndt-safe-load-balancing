from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analysis import plot_4c_models as p


class Plot4CModelsTest(unittest.TestCase):
    def test_numbers_come_from_the_committed_files(self) -> None:
        skill = p.skill_table()
        self.assertAlmostEqual(skill["gradual"]["xgboost"], 0.6490736445038088)
        self.assertGreater(skill["hot-switch"]["xgboost"], 1.0)      # the negative result
        rmse = p.outcome_rmse()
        self.assertAlmostEqual(rmse["direct_rf"]["u_dst_post"], 0.08267047878731165)
        deployed, margin, rows = p.mu_coverage()
        self.assertEqual(deployed, "direct_rf")
        self.assertAlmostEqual(margin, 0.13860323509560712)
        self.assertEqual([r["value"] for r in rows], ["0.45", "0.75", "0.85", "0.92", "0.98"])
        self.assertEqual(sum(r["n"] for r in rows), 150)

    def test_writes_both_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(p.main(["--out", tmp]), 0)
            names = sorted(f.name for f in Path(tmp).iterdir())
        self.assertEqual(names, ["fig_4c_forecast_skill.pdf", "fig_4c_forecast_skill.png",
                                 "fig_4c_outcome.pdf", "fig_4c_outcome.png"])


if __name__ == "__main__":
    unittest.main()
