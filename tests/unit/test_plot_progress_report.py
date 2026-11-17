from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis import plot_progress_report as ppr

CV_TEXT = """# round 1: candidates v1B, b1, b2 (margin only)
rows=120 folds=5 m_u=0.0366 band sizes=[44, 48, 28]
cand | unsafe_accepted | unsafe_rejected | false_rejections | safe_accepted | qualifies
v1B | 6 | 30 | 6 | 78 | True
b1 | 4 | 32 | 29 | 55 | False

# round 1: candidate a (retrained)
cand | unsafe_accepted | unsafe_rejected | false_rejections | safe_accepted | qualifies
v1B | 6 | - | 6 | - | True
a | 4 | 32 | 13 | 71 | False
"""

CSV_HEADER = ("run_id,controller,target_new_flow_rate,response_p95_ms_mean,"
              "qos_success_ratio,flow_setup_latency_p95_ms,valid\n")


class LoaderTest(unittest.TestCase):
    def test_cv_table_keeps_first_complete_row_and_skips_dashes(self) -> None:
        rows = ppr.parse_cv_table(CV_TEXT)
        self.assertEqual([r["name"] for r in rows], ["v1B", "b1", "a"])
        self.assertEqual(rows[0]["unsafe_accepted"] + rows[0]["unsafe_rejected"], 36)
        self.assertEqual(rows[2]["false_rejections"], 13)

    def test_capacity_means_and_failed_probe_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.csv"
            path.write_text(CSV_HEADER
                            + "r1,c1,30,0.70,1.0,20.0,True\n"
                            + "r2,c1,30,0.80,1.0,30.0,True\n"
                            + "r3,c1,160,0.59,0.0,0.0,True\n"
                            + "r4,c1,160,0.99,1.0,9.0,False\n")
            rows = ppr.capacity_curves(path)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["setup_p95_ms"], 25.0)
        self.assertAlmostEqual(rows[0]["response_p95_ms"], 0.75)
        self.assertIsNone(rows[1]["setup_p95_ms"])  # 0.0 with no successful probe
        self.assertEqual(rows[1]["repeats"], 1)  # the invalid run is not counted

    def test_workload_names_are_normalised(self) -> None:
        forecast = {"splits": {
            "test": {"with_gate": {"by_workload": {"burst": {"coverage": 0.9, "samples": 3}}}},
            "unseen": {"with_gate": {"by_workload": {"hot-switch": {"coverage": 0.8,
                                                                    "samples": 5}}}}}}
        outcome = {"by_model": {"hybrid_xgb": {"m_U": {"test": [
            {"group": "workload", "value": "hot_switch", "n": 4, "covered": 3},
            {"group": "all", "value": "all", "n": 4, "covered": 3}]}}}}
        rows = ppr.margin_coverage(forecast, outcome)
        self.assertEqual({r["workload"] for r in rows}, {"burst", "hot-switch"})
        self.assertAlmostEqual(rows[-1]["coverage"], 0.75)


class OutcomeLoaderTest(unittest.TestCase):
    def test_validation_rows_dedupe_and_add_selection_score(self) -> None:
        row = {"model": "analytical", "split": "validation", "target": "u_dst_post",
               "rmse": "0.07", "n": "60.0"}
        other = dict(row, model="hybrid_xgb", rmse="0.02")
        test_row = dict(row, split="test", rmse="9")
        selection = {"frozen": {"validation_scores": {"analytical": {"S": 0.23},
                                                      "hybrid_xgb": {"S": 0.1}}}}
        rows = ppr.outcome_validation([row, test_row], [row, other], selection)
        u = [r for r in rows if r["target"] == "u_dst_post"]
        self.assertEqual([(r["model"], r["rmse"]) for r in u],
                         [("analytical", 0.07), ("hybrid_xgb", 0.02)])
        self.assertEqual(sum(r["target"] == "S" for r in rows), 2)

    def test_bootstrap_keeps_rmse_rows_only(self) -> None:
        rows = ppr.outcome_test_ci([
            {"kind": "rmse", "model": "hybrid_xgb", "reference": "", "target": "u_dst_post",
             "estimate": "0.027", "ci_low": "0.021", "ci_high": "0.034", "share_better": ""},
            {"kind": "improvement", "model": "hybrid_xgb", "reference": "analytical",
             "target": "u_dst_post", "estimate": "59", "ci_low": "52", "ci_high": "67",
             "share_better": "1.0"}])
        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]["ci_low"], rows[0]["rmse"])


class MainTest(unittest.TestCase):
    def test_missing_inputs_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ppr.main(["--root", tmp]), 0)
            manifest = json.loads(
                (Path(tmp) / ppr.OUT / "figures_data.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["figures"], {})

    def test_draws_every_figure_from_the_repo_files(self) -> None:
        if not all((ppr.ROOT / p).exists() for p in ppr.SOURCES.values()):
            self.skipTest("repository evidence files not present")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ppr.main(["--out", tmp]), 0)
            manifest = json.loads((Path(tmp) / "figures_data.json").read_text(encoding="utf-8"))
            names = sorted(manifest["figures"])
            self.assertEqual(len(names), 11)
            for name in names:
                self.assertTrue((Path(tmp) / f"{name}.png").exists())
                self.assertTrue((Path(tmp) / f"{name}.pdf").exists())
        loop = manifest["figures"]["fig_10_closed_loop_migrations"]["data"]
        self.assertEqual(len(loop), 6)
        val = manifest["figures"]["fig_05_outcome_validation_all_models"]["data"]
        self.assertEqual({r["model"] for r in val}, set(ppr.OUTCOME_MODELS))
        fc = manifest["figures"]["fig_02_forecast_rmse_all_models"]["data"]
        self.assertEqual({r["model"] for r in fc}, set(ppr.FORECAST_MODELS))


if __name__ == "__main__":
    unittest.main()
