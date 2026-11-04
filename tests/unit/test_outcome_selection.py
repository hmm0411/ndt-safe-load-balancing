"""Model selection and the one-time test (Day 4): lock, frozen choice, CLI end to end."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import test_direct_outcome_models as direct_data
import yaml

from src.twin.outcome import selection

SCRIPT = "scripts/experiments/train_outcome_models.py"
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot_switch")


def varied_rows(n: int, split: str, seed: int) -> list[dict[str, Any]]:
    rows = direct_data.synthetic_rows(n, split, seed)
    for i, r in enumerate(rows):
        r["workload"] = WORKLOADS[i % 5]
        r["destination_load_level"] = (0.45, 0.75)[i % 2]
        r["switch_share_group"] = ("low", "medium", "high")[i % 3]
        r["saturated_before"] = i % 7 == 0
        r["overload_after"] = r["u_dst_post"] >= 1.0
    return rows


class TestLockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.csv").write_text("x\n1\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_open_once(self) -> None:
        self.assertEqual(selection.check_can_open(self.tmp, "abc"), 1)
        lock = selection.open_test(self.tmp, "abc")
        self.assertEqual((lock["status"], lock["attempt"]), ("opened", 1))
        done = selection.complete_test(self.tmp, {"a.csv": self.tmp / "a.csv"}, {"n_test": 1})
        self.assertEqual(done["status"], "completed")
        self.assertEqual(done["outputs_sha256"]["a.csv"],
                         selection.file_sha256(self.tmp / "a.csv"))
        for sha in ("abc", "other"):
            with self.assertRaises(selection.LockedTestError):
                selection.open_test(self.tmp, sha)

    def test_crashed_run_may_retry_only_the_same_selection(self) -> None:
        selection.open_test(self.tmp, "abc")  # no complete_test: the run crashed
        with self.assertRaises(selection.LockedTestError):
            selection.check_can_open(self.tmp, "changed")
        lock = selection.open_test(self.tmp, "abc")
        self.assertEqual(lock["attempt"], 2)
        self.assertIn("previous_opened_at", lock)

    def test_complete_needs_an_opened_lock(self) -> None:
        with self.assertRaises(selection.LockedTestError):
            selection.complete_test(self.tmp, {}, {})

    def test_canonical_hash_ignores_key_order(self) -> None:
        self.assertEqual(selection.canonical_sha256({"a": 1, "b": [1, 2]}),
                         selection.canonical_sha256({"b": [1, 2], "a": 1}))
        with self.assertRaises(FileNotFoundError):
            selection.tree_sha256({"x": self.tmp / "missing"})


class SelectAndTestCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.models = self.tmp / "models.yaml"
        self.models.write_text(yaml.safe_dump(direct_data.SMALL))
        evaluation_config = yaml.safe_load(Path("configs/outcome/evaluation.yaml").read_text())
        evaluation_config["bootstrap"]["resamples"] = 200
        evaluation_config["ablation"]["hybrid_no_temporal"]["drop"] = ["src_u_slope"]
        evaluation_config["ablation"]["hybrid_no_switch_features"]["drop"] = [
            "switch_control_load_share"]
        evaluation_config["ablation"]["hybrid_no_switch"]["drop"] = ["switch_control_load_share"]
        self.eval = self.tmp / "evaluation.yaml"
        self.eval.write_text(yaml.safe_dump(evaluation_config))
        self.train = varied_rows(60, "train", 1)
        self.validation = varied_rows(20, "validation", 2)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, SCRIPT, *args, "--models-config", str(self.models)],
                              capture_output=True, text=True, check=False)

    def prepare(self, name: str, test_seed: int) -> dict[str, Path]:
        rows = self.train + self.validation + varied_rows(15, "test", test_seed)
        dataset = direct_data.write_dataset(self.tmp / name / "dataset", rows)
        dirs = {"dataset": dataset, "direct": self.tmp / name / "direct",
                "hybrid": self.tmp / name / "hybrid", "final": self.tmp / name / "final"}
        result = self.run_cli("--stage", "direct", "--dataset", str(dataset),
                              "--out", str(dirs["direct"]))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_cli("--stage", "hybrid", "--dataset", str(dataset), "--direct-dir",
                              str(dirs["direct"]), "--out", str(dirs["hybrid"]))
        self.assertEqual(result.returncode, 0, result.stderr)
        return dirs

    def final(self, dirs: dict[str, Path]) -> subprocess.CompletedProcess[str]:
        return self.run_cli("--stage", "select-and-test", "--dataset", str(dirs["dataset"]),
                            "--direct-dir", str(dirs["direct"]), "--hybrid-dir",
                            str(dirs["hybrid"]), "--eval-config", str(self.eval),
                            "--out", str(dirs["final"]))

    def test_select_and_test_once(self) -> None:
        dirs = self.prepare("a", 3)
        result = self.final(dirs)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = dirs["final"]
        chosen = json.loads((out / "model_selection.json").read_text())
        frozen = chosen["frozen"]
        scores = {n: v["S"] for n, v in frozen["validation_scores"].items()}
        self.assertEqual(frozen["best_direct"],
                         min(("direct_xgb", "direct_rf"), key=lambda n: scores[n]))
        self.assertEqual(frozen["best_hybrid"],
                         min(("hybrid_xgb", "hybrid_rf"), key=lambda n: scores[n]))
        self.assertEqual(frozen["selection_exclude"], ["r95_post"])
        self.assertEqual(chosen["frozen_sha256"], selection.canonical_sha256(frozen))
        lock = json.loads((out / "test_lock.json").read_text())
        self.assertEqual(lock["status"], "completed")
        self.assertEqual(lock["selection_sha256"], chosen["frozen_sha256"])
        self.assertEqual(lock["test_summary"]["n_test"], 15)
        self.assertEqual(lock["test_summary"]["refit_max_abs_diff"], 0.0)
        for name in ("validation_metrics.csv", "test_metrics.csv", "per_workload_metrics.csv",
                     "model_comparison.csv", "bootstrap_ci.csv", "saturation_sensitivity.csv",
                     "overload_cases.csv", "ablation.csv"):
            self.assertIn(name, lock["outputs_sha256"])
        with (out / "test_metrics.csv").open() as handle:
            models = {r["model"] for r in csv.DictReader(handle)}
        self.assertEqual(models, {"analytical", frozen["best_direct"], frozen["best_hybrid"]})
        with (out / "ablation.csv").open() as handle:
            variants = {r["variant"] for r in csv.DictReader(handle)}
        self.assertEqual(variants, {"analytical", frozen["best_direct"], "hybrid_full",
                                    "hybrid_no_temporal", "hybrid_no_switch_features",
                                    "hybrid_no_switch"})
        with (out / "per_workload_metrics.csv").open() as handle:
            groups = {(r["split"], r["group"]) for r in csv.DictReader(handle)}
        self.assertIn(("test", "workload"), groups)
        self.assertIn(("validation", "saturated_before"), groups)

        again = self.final(dirs)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("test already evaluated", again.stderr)
        self.assertEqual(json.loads((out / "model_selection.json").read_text()), chosen)

    def test_selection_does_not_depend_on_test_rows(self) -> None:
        frozen = []
        for name, seed in (("a", 3), ("b", 4)):
            dirs = self.prepare(name, seed)
            self.assertEqual(self.final(dirs).returncode, 0)
            chosen = json.loads((dirs["final"] / "model_selection.json").read_text())["frozen"]
            for key in ("sha256",):  # the dataset file differs (other test rows)
                chosen.pop(key)
            frozen.append(chosen)
        self.assertEqual(frozen[0], frozen[1])

    def test_changed_validation_scores_are_refused(self) -> None:
        dirs = self.prepare("a", 3)
        path = dirs["hybrid"] / "selection_scores.json"
        scores = json.loads(path.read_text())
        scores["hybrid_xgb"]["selection"]["score"] += 0.01
        path.write_text(json.dumps(scores))
        result = self.final(dirs)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("models or dataset changed", result.stderr)
        self.assertFalse((dirs["final"] / "test_lock.json").exists())


if __name__ == "__main__":
    unittest.main()
