"""Direct ML (Day 2): grid on train, choice on validation, test never used, save/load."""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from src.twin.outcome import trainer
from src.twin.outcome.dataset import DATASET_COLUMNS
from src.twin.outcome.features import CANDIDATE_FEATURES
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS, OutcomeValues

FEATURES = ["src_u_mean", "dst_u_mean", "switch_control_load_share", "src_u_slope"]
SMALL = {
    "targets": list(BASE_TARGETS),
    "random_forest": {"fixed": {"n_jobs": 1, "random_state": 0},
                      "grid": {"n_estimators": [20], "max_depth": [3, None]}},
    "xgboost": {"fixed": {"n_jobs": 1, "random_state": 0, "tree_method": "hist"},
                "grid": {"n_estimators": [30], "max_depth": [2, 3]}},
}


def synthetic_rows(n: int, split: str, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        f = {name: rng.uniform(0.0, 1.0) for name in CANDIDATE_FEATURES}
        f["src_u_mean"], f["dst_u_mean"] = rng.uniform(0.6, 1.0), rng.uniform(0.4, 0.8)
        f["switch_control_load_share"] = rng.uniform(0.05, 0.5)
        move = f["src_u_mean"] * f["switch_control_load_share"]
        observed = OutcomeValues(f["src_u_mean"] - 1.2 * move + 0.5 * f["src_u_slope"] * 0.1,
                                 f["dst_u_mean"] + move, 0.77, 0.78 + 0.01 * f["dst_u_mean"])
        analytic = OutcomeValues(f["src_u_mean"] - move, f["dst_u_mean"] + move, 0.77, 0.78)
        rows.append({"attempt_id": f"mig-{split}-{i:03d}", "workload": "stable", "phase": "middle",
                     "destination_load_level": 0.45, "direction": "c1->c2", "repeat": "r01",
                     "split": split, "switch_id": "s1", "source_controller": "c1",
                     "target_controller": "c2", "target_hot": False, "saturated_before": False,
                     "saturated_after": False, "overload_after": False,
                     "switch_share_group": "medium", **f, **analytic.to_dict("analytical_"),
                     **observed.to_dict()})
    return rows


def write_dataset(out: Path, rows: list[dict[str, Any]]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "outcome_dataset.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DATASET_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows({c: r[c] for c in DATASET_COLUMNS} for r in rows)
    (out / "feature_schema.json").write_text(json.dumps(
        {"feature_schema": "outcome-features-v1", "model_features": FEATURES}))
    return out


class DirectModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.config_path = self.tmp / "models.yaml"
        self.config_path.write_text(yaml.safe_dump(SMALL))
        self.config = trainer.load_models_config(self.config_path)
        self.train = synthetic_rows(60, "train", 1)
        self.validation = synthetic_rows(20, "validation", 2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_repo_grid_sizes(self) -> None:
        config = trainer.load_models_config(Path("configs/outcome/models.yaml"))
        self.assertEqual(len(config.families["random_forest"].candidates()), 24)
        self.assertEqual(len(config.families["xgboost"].candidates()), 32)
        self.assertEqual(config.targets, BASE_TARGETS)

    def test_config_is_checked(self) -> None:
        bad = dict(SMALL, targets=["i_post"])
        self.config_path.write_text(yaml.safe_dump(bad))
        with self.assertRaises(ValueError):
            trainer.load_models_config(self.config_path)
        bad = dict(SMALL, xgboost={"fixed": {}, "grid": {"max_depth": []}})
        self.config_path.write_text(yaml.safe_dump(bad))
        with self.assertRaises(ValueError):
            trainer.load_models_config(self.config_path)

    def test_search_uses_validation_and_refuses_test(self) -> None:
        best, records = trainer.search(self.train, self.validation, FEATURES, self.config)
        self.assertEqual(len(records), 4 * (2 + 2))
        self.assertEqual(set(best), {(f, t) for f in trainer.FAMILIES for t in BASE_TARGETS})
        for (family, target), fitted in best.items():
            rmses = [r["validation_rmse"] for r in records
                     if r["family"] == family and r["target"] == target]
            self.assertEqual(fitted.validation_rmse, min(rmses))
        test = synthetic_rows(5, "test", 3)
        with self.assertRaises(ValueError):
            trainer.search(self.train, test, FEATURES, self.config)
        with self.assertRaises(ValueError):
            trainer.search(self.train + test, self.validation, FEATURES, self.config)

    def test_derived_targets_and_metrics(self) -> None:
        best, _ = trainer.search(self.train, self.validation, FEATURES, self.config,
                                 families=("random_forest",))
        models = {t: best[("random_forest", t)].model for t in BASE_TARGETS}
        pred = trainer.predict_targets(models, self.validation, FEATURES)
        self.assertEqual(set(pred), set(ALL_TARGETS))
        self.assertEqual(pred["r95_post"][0], max(pred["r_src95_post"][0], pred["r_dst95_post"][0]))
        self.assertAlmostEqual(pred["i_post"][0],
                               abs(pred["u_src_post"][0] - pred["u_dst_post"][0]) / 2)
        table = trainer.metrics_table("direct_rf", "validation", self.validation, pred)
        self.assertEqual([r["target"] for r in table], list(ALL_TARGETS))

    def test_save_and_load_give_the_same_predictions(self) -> None:
        best, _ = trainer.search(self.train, self.validation, FEATURES, self.config)
        files = trainer.save_models(best, self.tmp / "models")
        self.assertIn("direct_rf_u_dst_post.joblib", files.values())
        self.assertIn("direct_xgb_u_dst_post.json", files.values())
        for family in trainer.FAMILIES:
            loaded = trainer.load_models(self.tmp / "models", family, BASE_TARGETS)
            fitted = {t: best[(family, t)].model for t in BASE_TARGETS}
            self.assertEqual(trainer.predict_targets(loaded, self.validation, FEATURES),
                             trainer.predict_targets(fitted, self.validation, FEATURES))

    def test_cli_direct_stage(self) -> None:
        test_rows = synthetic_rows(10, "test", 4)
        for row in test_rows:  # poisoned: any use of test rows would break the fit
            row["src_u_mean"] = float("nan")
        dataset = write_dataset(self.tmp / "dataset", self.train + self.validation + test_rows)
        out = self.tmp / "out"
        result = subprocess.run(
            [sys.executable, "scripts/experiments/train_outcome_models.py", "--stage", "direct",
             "--dataset", str(dataset), "--models-config", str(self.config_path),
             "--out", str(out)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("test rows dropped 10", result.stdout)
        self.assertIn("save/load identical: True", result.stdout)
        hyper = json.loads((out / "hyperparameters.json").read_text())
        self.assertEqual(hyper["rows"], {"train": 60, "validation": 20, "test_dropped": 10})
        self.assertEqual(len(hyper["selected"]), 8)
        scores = json.loads((out / "selection_scores.json").read_text())
        self.assertEqual(set(scores), {"analytical", "direct_rf", "direct_xgb"})
        with (out / "direct_ml_metrics.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 3 * 2 * 6)


if __name__ == "__main__":
    unittest.main()
