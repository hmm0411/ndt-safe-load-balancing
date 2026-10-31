"""Hybrid outcome model (Day 3): y_H = y_A + r_hat, residual search, save/load, CLI."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import test_direct_outcome_models as direct_data
import yaml

from src.twin.outcome import hybrid, residual, trainer
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS, OutcomeValues

FEATURES = residual.hybrid_features(direct_data.FEATURES)
SCRIPT = "scripts/experiments/train_outcome_models.py"


class Constant:
    """Stand-in residual model that predicts the same value for every row."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)


def zero_residual_rows(n: int, split: str, seed: int) -> list[dict[str, Any]]:
    rows = direct_data.synthetic_rows(n, split, seed)
    for r in rows:
        r.update({t: r[f"analytical_{t}"] for t in ALL_TARGETS})
    return rows


class HybridOutcomeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.config_path = self.tmp / "models.yaml"
        self.config_path.write_text(yaml.safe_dump(direct_data.SMALL))
        self.config = trainer.load_models_config(self.config_path)
        self.train = direct_data.synthetic_rows(60, "train", 1)
        self.validation = direct_data.synthetic_rows(20, "validation", 2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def analytical(self, rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
        return {t: [float(r[f"analytical_{t}"]) for r in rows] for t in ALL_TARGETS}

    def test_zero_residual_gives_analytical(self) -> None:
        zero = {t: Constant(0.0) for t in BASE_TARGETS}
        self.assertEqual(hybrid.predict(zero, self.validation, FEATURES),
                         self.analytical(self.validation))

    def test_learned_zero_residual_gives_analytical(self) -> None:
        train = zero_residual_rows(40, "train", 5)
        validation = zero_residual_rows(10, "validation", 6)
        best, _ = trainer.search(train, validation, FEATURES, self.config,
                                 target_values=residual.residual_values)
        for family in trainer.FAMILIES:
            models = {t: best[(family, t)].model for t in BASE_TARGETS}
            pred = hybrid.predict(models, validation, FEATURES)
            for target in ALL_TARGETS:
                for got, want in zip(pred[target], self.analytical(validation)[target]):
                    self.assertAlmostEqual(got, want, places=6)

    def test_combine_derives_and_floors_only_at_zero(self) -> None:
        analytical = {"u_src_post": [0.05, 0.50], "u_dst_post": [1.20, 0.90],
                      "r_src95_post": [0.77, 0.77], "r_dst95_post": [0.78, 0.60]}
        residuals = {"u_src_post": [-0.08, 0.02], "u_dst_post": [0.10, -0.05],
                     "r_src95_post": [0.0, 0.0], "r_dst95_post": [0.0, 0.30]}
        pred = hybrid.combine(analytical, residuals)
        self.assertEqual(pred["u_src_post"][0], 0.0)  # floored
        self.assertAlmostEqual(pred["u_dst_post"][0], 1.30)  # no upper clamp
        self.assertAlmostEqual(pred["r95_post"][1], 0.90)
        self.assertAlmostEqual(pred["i_post"][0], abs(0.0 - 1.30) / 2)
        values = OutcomeValues(**{t: pred[t][1] for t in BASE_TARGETS})
        self.assertAlmostEqual(values.i_post, pred["i_post"][1])
        self.assertEqual(hybrid.floored(analytical, residuals),
                         {"u_src_post": 1, "u_dst_post": 0, "r_src95_post": 0,
                          "r_dst95_post": 0})
        with self.assertRaises(ValueError):
            hybrid.combine(analytical, {**residuals, "u_dst_post": [0.1]})

    def test_residual_search_scores_the_hybrid_error(self) -> None:
        best, records = trainer.search(self.train, self.validation, FEATURES, self.config,
                                       target_values=residual.residual_values)
        self.assertEqual(len(records), 4 * (2 + 2))
        for family in trainer.FAMILIES:
            models = {t: best[(family, t)].model for t in BASE_TARGETS}
            pred = hybrid.predict(models, self.validation, FEATURES)
            floors = hybrid.floored(hybrid.analytical_base(self.validation),
                                    hybrid.predict_residuals(models, self.validation, FEATURES))
            table = trainer.metrics_table("hybrid", "validation", self.validation, pred)
            for target in BASE_TARGETS:
                if floors[target]:
                    continue
                rmse = next(r["rmse"] for r in table if r["target"] == target)
                self.assertAlmostEqual(rmse, best[(family, target)].validation_rmse, places=9)
        test = direct_data.synthetic_rows(5, "test", 3)
        with self.assertRaises(ValueError):
            trainer.search(self.train, test, FEATURES, self.config,
                           target_values=residual.residual_values)

    def test_save_and_load_give_the_same_predictions(self) -> None:
        best, _ = trainer.search(self.train, self.validation, FEATURES, self.config,
                                 target_values=residual.residual_values)
        files = hybrid.save(best, self.tmp / "models")
        self.assertIn("residual_rf_u_src_post.joblib", files.values())
        self.assertIn("residual_xgb_u_src_post.json", files.values())
        for family in trainer.FAMILIES:
            fitted = {t: best[(family, t)].model for t in BASE_TARGETS}
            self.assertEqual(hybrid.predict(hybrid.load(self.tmp / "models", family),
                                            self.validation, FEATURES),
                             hybrid.predict(fitted, self.validation, FEATURES))
        with self.assertRaises(ValueError):
            hybrid.predict_residuals({"u_src_post": Constant(0.0)}, self.validation, FEATURES)

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, SCRIPT, *args, "--models-config",
                               str(self.config_path)],
                              capture_output=True, text=True, check=False)

    def test_cli_hybrid_stage(self) -> None:
        test_rows = direct_data.synthetic_rows(10, "test", 4)
        for r in test_rows:  # poisoned: any use of test rows would break the fit
            r["src_u_mean"] = float("nan")
            r["analytical_u_src_post"] = float("nan")
        dataset = direct_data.write_dataset(self.tmp / "dataset",
                                            self.train + self.validation + test_rows)
        direct_dir, out = self.tmp / "direct", self.tmp / "hybrid"
        common = ("--dataset", str(dataset), "--direct-dir", str(direct_dir))

        missing = self.run_cli("--stage", "hybrid", *common, "--out", str(out))
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("run --stage direct first", missing.stderr)

        direct = self.run_cli("--stage", "direct", "--dataset", str(dataset),
                              "--out", str(direct_dir))
        self.assertEqual(direct.returncode, 0, direct.stderr)
        result = self.run_cli("--stage", "hybrid", *common, "--out", str(out))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test rows dropped 10", result.stdout)
        self.assertIn("4 model features + 4 analytical = 8 hybrid features", result.stdout)
        self.assertIn("lowest validation S: Direct", result.stdout)
        self.assertIn("save/load identical: True", result.stdout)

        hyper = json.loads((out / "hyperparameters.json").read_text())
        self.assertEqual(hyper["stage"], "hybrid")
        self.assertEqual(hyper["hybrid_features"], FEATURES)
        self.assertEqual(hyper["rows"], {"train": 60, "validation": 20, "test_dropped": 10})
        self.assertEqual(len(hyper["selected"]), 8)
        self.assertEqual(set(hyper["floored_at_zero"]), {"hybrid_rf", "hybrid_xgb"})
        self.assertTrue((out / "models" / "residual_xgb_r_dst95_post.json").is_file())
        scores = json.loads((out / "selection_scores.json").read_text())
        self.assertEqual(set(scores), {"analytical", "direct_rf", "direct_xgb", "hybrid_rf",
                                       "hybrid_xgb"})
        self.assertNotIn("n_r95_post", scores["hybrid_xgb"]["selection"])
        direct_scores = json.loads((direct_dir / "selection_scores.json").read_text())
        self.assertAlmostEqual(scores["direct_xgb"]["selection"]["score"],
                               direct_scores["direct_xgb"]["selection"]["score"])
        with (out / "hybrid_metrics.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5 * 2 * 6)
        with (out / "residual_analysis.csv").open() as handle:
            analysis = list(csv.DictReader(handle))
        self.assertIn("rmse_hybrid_xgb", analysis[0])
        self.assertEqual({r["split"] for r in analysis}, {"train", "validation"})
        self.assertEqual({r["group"] for r in analysis},
                         {"all", "workload", "switch_share_group", "destination_load_level"})

    def test_cli_refuses_direct_models_on_other_features(self) -> None:
        dataset = direct_data.write_dataset(self.tmp / "dataset", self.train + self.validation)
        direct_dir = self.tmp / "direct"
        direct = self.run_cli("--stage", "direct", "--dataset", str(dataset),
                              "--out", str(direct_dir))
        self.assertEqual(direct.returncode, 0, direct.stderr)
        schema_path = dataset / "feature_schema.json"
        schema = json.loads(schema_path.read_text())
        schema["model_features"] = schema["model_features"][:3]
        schema_path.write_text(json.dumps(schema))
        result = self.run_cli("--stage", "hybrid", "--dataset", str(dataset),
                              "--direct-dir", str(direct_dir), "--out", str(self.tmp / "h"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rerun --stage direct", result.stderr)


if __name__ == "__main__":
    unittest.main()
