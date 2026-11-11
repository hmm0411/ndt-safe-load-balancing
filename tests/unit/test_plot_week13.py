from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import plot_week13 as pw
from tests.unit.test_analyze_closed_loop import ALL_PASS
from scripts.decision import analyze_closed_loop as acl


class PlotWeek13Test(unittest.TestCase):
    def test_draws_what_exists_and_skips_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(pw.main(["--root", str(root)]), 0)  # nothing there: skips
            sens = {"policies": {
                "base": {"top1_agreement": 1.0, "kendall_tau_mean": 1.0, "epochs_compared": 4},
                "relief_heavy": {"top1_agreement": 0.75, "kendall_tau_mean": 0.8,
                                 "epochs_compared": 4}}}
            (root / "evidence/week13").mkdir(parents=True)
            (root / "evidence/week13/ranking_sensitivity.json").write_text(json.dumps(sens))
            (root / "evidence/week14").mkdir(parents=True)
            (root / "evidence/week14/closed_loop_2c_analysis.json").write_text(
                json.dumps(acl.analyze(ALL_PASS)))
            self.assertEqual(pw.main(["--root", str(root)]), 0)
            self.assertTrue((root / "evidence/week13/fig_w13_ranking_top1.png").exists())
            self.assertTrue((root / "evidence/week14/fig_w14_decision_latency.png").exists())


if __name__ == "__main__":
    unittest.main()
