"""Final evaluation helpers (Day 4): groups, saturation, overload, bootstrap, ablation base."""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.twin.outcome import evaluation
from src.twin.outcome.targets import ALL_TARGETS, OutcomeValues

REPO_CONFIG = Path("configs/outcome/evaluation.yaml")


def make_rows(n: int = 12) -> list[dict[str, Any]]:
    rows = []
    for i in range(n):
        truth = OutcomeValues(0.5 + 0.02 * i, 0.7 + 0.03 * i, 0.77, 0.78)
        rows.append({
            "attempt_id": f"mig-{i:02d}", "workload": ("stable", "burst")[i % 2],
            "phase": "middle", "direction": ("c1->c2", "c2->c1")[i % 2],
            "destination_load_level": (0.45, 0.75)[(i // 2) % 2],
            "switch_share_group": ("low", "medium", "high")[i % 3],
            "saturated_before": i in (0, 1), "saturated_after": i == 11,
            "overload_after": truth.u_dst_post >= 1.0, "target_hot": False,
            "src_u_mean": 0.8, "dst_u_mean": 0.6,
            **truth.to_dict(), **OutcomeValues(0.55, 0.75, 0.76, 0.79).to_dict("analytical_"),
        })
    return rows


def truth_of(rows: list[dict[str, Any]], shift: float = 0.0) -> dict[str, list[float]]:
    return {t: [float(r[t]) + shift for r in rows] for t in ALL_TARGETS}


class EvaluationConfigTest(unittest.TestCase):
    def test_repo_config(self) -> None:
        config = evaluation.load_evaluation_config(REPO_CONFIG)
        self.assertEqual(config.direct_candidates, ("direct_xgb", "direct_rf"))
        self.assertEqual(config.hybrid_candidates, ("hybrid_xgb", "hybrid_rf"))
        self.assertIn("workload", config.group_keys)
        self.assertEqual(set(config.ablation), {"hybrid_no_temporal", "hybrid_no_switch_features",
                                                "hybrid_no_switch"})
        self.assertTrue(config.ablation["hybrid_no_switch"].persistence_base)
        self.assertIn("switch_control_load_share", config.ablation["hybrid_no_switch"].drop)
        self.assertIn("src_u_slope", config.ablation["hybrid_no_temporal"].drop)

    def test_config_is_checked(self) -> None:
        data = yaml.safe_load(REPO_CONFIG.read_text())
        bad_cases = [
            {**data, "candidates": {"direct": ["hybrid_xgb"], "hybrid": ["hybrid_rf"]}},
            {**data, "group_keys": ["attempt_id"]},
            {**data, "bootstrap": {**data["bootstrap"], "resamples": 10}},
            {**data, "ablation": {"x": {"drop": ["latency"]}}},
            {**data, "ablation": {"x": {"drop": [], "persistence_base": False}}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "e.yaml"
            for bad in bad_cases:
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):
                    evaluation.load_evaluation_config(path)


class EvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = make_rows()
        self.preds = {"perfect": truth_of(self.rows), "off": truth_of(self.rows, 0.1)}

    def test_choose_lowest_and_ties_keep_order(self) -> None:
        scores = {"direct_xgb": 0.113, "direct_rf": 0.132, "hybrid_xgb": 0.2, "hybrid_rf": 0.2}
        self.assertEqual(evaluation.choose(scores, ("direct_xgb", "direct_rf")), "direct_xgb")
        self.assertEqual(evaluation.choose(scores, ("hybrid_rf", "hybrid_xgb")), "hybrid_rf")
        with self.assertRaises(ValueError):
            evaluation.choose(scores, ("direct_xgb", "analytical"))

    def test_improvement(self) -> None:
        self.assertAlmostEqual(evaluation.improvement(0.113, 0.105), (0.008 / 0.113) * 100)
        self.assertLess(evaluation.improvement(0.0192, 0.0240), 0)
        self.assertTrue(math.isnan(evaluation.improvement(0.0, 0.1)))

    def test_group_metrics(self) -> None:
        table = evaluation.group_metrics(self.rows, self.preds, "test",
                                         ("workload", "switch_share_group"))
        groups = {(r["group"], r["value"]) for r in table}
        self.assertEqual(groups, {("all", "all"), ("workload", "burst"), ("workload", "stable"),
                                  ("switch_share_group", "high"), ("switch_share_group", "low"),
                                  ("switch_share_group", "medium")})
        burst = [r for r in table if r["value"] == "burst" and r["model"] == "off"]
        self.assertEqual(len(burst), len(ALL_TARGETS))
        self.assertEqual(burst[0]["n"], 6)
        self.assertAlmostEqual(burst[0]["rmse"], 0.1)
        self.assertNotIn("r2", burst[0])

    def test_saturation_subsets(self) -> None:
        table = evaluation.saturation_sensitivity(self.rows, self.preds, "test")
        n = {r["value"]: r["n"] for r in table if r["model"] == "perfect"}
        self.assertEqual(n, {"all": 12, "without_saturated": 9, "saturated_only": 3})
        rows = [r for r in self.rows if not r["saturated_before"] and not r["saturated_after"]]
        preds = {"perfect": truth_of(rows)}
        table = evaluation.saturation_sensitivity(rows, preds, "test")
        self.assertEqual({r["value"] for r in table}, {"all", "without_saturated"})

    def test_overload_cases(self) -> None:
        cases = evaluation.overload_cases(self.rows, self.preds, "test", 1.0)
        expected = [r["attempt_id"] for r in self.rows if r["u_dst_post"] >= 1.0]
        self.assertTrue(expected)
        self.assertEqual([c["attempt_id"] for c in cases], expected)
        first = cases[0]
        self.assertAlmostEqual(first["error_off"], 0.1)
        self.assertTrue(first["flags_overload_perfect"])
        self.assertEqual(first["u_dst_post"], first["pred_perfect"])

    def test_bootstrap_is_paired_and_reproducible(self) -> None:
        weights = {"u_dst_post": 0.5, "u_src_post": 0.3, "i_post": 0.2}
        args = (self.rows, self.preds, [("off", "perfect")], weights, 1e-9, 500, 7, 0.95)
        first = evaluation.paired_bootstrap(*args)
        self.assertEqual(first, evaluation.paired_bootstrap(*args))
        imp = {r["target"]: r for r in first if r["kind"] == "improvement"}
        self.assertAlmostEqual(imp["u_dst_post"]["estimate"], 100.0)
        self.assertAlmostEqual(imp["u_dst_post"]["ci_low"], 100.0)
        self.assertEqual(imp["u_dst_post"]["share_better"], 1.0)
        self.assertIn("S", imp)
        rmse = {(r["model"], r["target"]): r for r in first if r["kind"] == "rmse"}
        self.assertAlmostEqual(rmse[("off", "u_src_post")]["estimate"], 0.1)
        self.assertLessEqual(rmse[("off", "u_src_post")]["ci_low"], 0.1 + 1e-12)

    def test_bootstrap_same_model_has_zero_improvement(self) -> None:
        preds = {"a": truth_of(self.rows, 0.05), "b": truth_of(self.rows, 0.05)}
        out = evaluation.paired_bootstrap(self.rows, preds, [("a", "b")],
                                          {"u_dst_post": 1.0}, 1e-9, 200, 1, 0.9)
        for r in out:
            if r["kind"] == "improvement":
                self.assertAlmostEqual(r["estimate"], 0.0)
                self.assertAlmostEqual(r["ci_low"], 0.0)
                self.assertAlmostEqual(r["ci_high"], 0.0)
                self.assertEqual(r["share_better"], 0.0)

    def test_persistence_base(self) -> None:
        base = evaluation.persistence_base(self.rows)
        self.assertEqual(base["u_src_post"][0], 0.8)
        self.assertEqual(base["u_dst_post"][0], 0.6)
        self.assertEqual(base["r_dst95_post"][0], 0.79)
        copies = evaluation.with_base(self.rows, base)
        self.assertEqual(copies[0]["analytical_u_src_post"], 0.8)
        self.assertEqual(copies[0]["u_src_post"], self.rows[0]["u_src_post"])
        self.assertEqual(self.rows[0]["analytical_u_src_post"], 0.55)  # originals untouched


if __name__ == "__main__":
    unittest.main()
