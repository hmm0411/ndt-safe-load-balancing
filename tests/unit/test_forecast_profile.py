from __future__ import annotations

import json
import os
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from scripts.experiments.build_forecast_splits import check_split_counts
from src.experiments.workloads import forecast_4c as f4c
from src.twin.forecasting import splitter
from src.twin.forecasting.loader import load_official_forecast_runs
from src.twin.forecasting.versions import PROFILES, active_profile

ROOT = Path(__file__).resolve().parents[2]
CONFIG_4C = ROOT / "configs/experiments/forecast_collection_4c_v1.yaml"


def run_with_profile(code: str, profile: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "NDT_FORECAST_PROFILE"}
    env["PYTHONPATH"] = str(ROOT)
    if profile is not None:
        env["NDT_FORECAST_PROFILE"] = profile
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


VERSIONS = ("from src.twin.forecasting import versions as v, splitter as s; "
            "print(v.MODEL_VERSION, v.dataset_version(), len(s.planned_run_ids()), "
            "s.planned_run_ids()[0])")


class ForecastProfileTest(unittest.TestCase):
    def test_default_profile_is_the_frozen_2c_v3(self) -> None:
        out = run_with_profile(VERSIONS, None)
        self.assertEqual(out, "forecast-models-v3 forecast-dataset-v3-h3-provisional 25 "
                              "stable-r01-s101")

    def test_4c_profile_switches_every_version(self) -> None:
        out = run_with_profile(VERSIONS, "4c40s")
        self.assertEqual(out, "forecast-models-v4 forecast-dataset-v4-h3-provisional 25 "
                              "forecast4c-stable-r01-s9921")

    def test_unknown_profile_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            active_profile({"NDT_FORECAST_PROFILE": "3c30s"})

    def test_4c_plan_matches_the_collected_run_directories(self) -> None:
        config = yaml.safe_load(CONFIG_4C.read_text(encoding="utf-8"))
        collected = {run.run_id for run in f4c.transfer_plan(config)}
        planned = set(splitter.planned_run_ids(PROFILES["4c40s"].experiment_config))
        self.assertEqual(planned, collected)

    def test_4c_split_follows_the_config(self) -> None:
        config = yaml.safe_load(CONFIG_4C.read_text(encoding="utf-8"))
        split = config["split"]
        policy = splitter.DEFAULT_SPLIT_POLICY
        self.assertEqual(tuple(split["train"]), policy.train_repeats)
        self.assertEqual(tuple(split["validation"]), policy.validation_repeats)
        self.assertEqual(tuple(split["test"]), policy.test_repeats)
        self.assertEqual([w.replace("_", "-") for w in split["unseen_workloads"]],
                         list(policy.unseen_workloads))
        ids = splitter.planned_run_ids(PROFILES["4c40s"].experiment_config)
        parts = splitter.split_run_ids(ids).as_dict()
        self.assertEqual({k: len(v) for k, v in parts.items()},
                         {"train": 12, "validation": 4, "test": 4, "unseen": 5})
        self.assertTrue(all("-hot-switch-" in r for r in parts["unseen"]))
        self.assertTrue(all("-r04-" in r for r in parts["validation"]))

    def test_prefixed_run_id_parses_to_the_workload(self) -> None:
        parsed = splitter.parse_run_id("forecast4c-hot-switch-r05-s9925")
        self.assertEqual((parsed.workload, parsed.repeat, parsed.seed),
                         ("hot-switch", 5, 9925))
        with self.assertRaises(ValueError):
            splitter.assign_split("transfer4c-stable-r01-s9701")

    def test_split_count_check(self) -> None:
        runs = {"train": 12, "validation": 4, "test": 4, "unseen": 5}
        rows = {"train": 8000, "validation": 2600, "test": 2600, "unseen": 3300}
        check_split_counts(rows, runs, 16500, profile="4c40s")
        with self.assertRaises(RuntimeError):
            check_split_counts(rows, runs, 16000, profile="4c40s")
        with self.assertRaises(RuntimeError):
            check_split_counts({**rows, "test": 0}, runs, 13900, profile="4c40s")
        with self.assertRaises(RuntimeError):
            check_split_counts(rows, {**runs, "unseen": 4}, 16500, profile="4c40s")
        with self.assertRaises(RuntimeError):   # 2C keeps its frozen row counts
            check_split_counts(rows, runs, 16500, profile="2c20s")

    def test_loader_reads_only_the_planned_4c_runs(self) -> None:
        """forecast_4c/ also holds the queue logs; transfer4c-* runs live elsewhere but a
        stray one must not enter the 4C training data."""
        planned = splitter.planned_run_ids(PROFILES["4c40s"].experiment_config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for run_id in (planned[0], planned[-1], "transfer4c-stable-r01-s9701"):
                (root / run_id).mkdir()
                (root / run_id / "forecast_samples.jsonl").write_text(
                    json.dumps({"run_id": run_id, "snapshot_valid": True}) + "\n",
                    encoding="utf-8")
            (root / "day5_vm_20261124T152001Z.log").write_text("x", encoding="utf-8")
            rows = load_official_forecast_runs(root, expected_run_ids=planned)
        self.assertEqual(sorted(r["run_id"] for r in rows), sorted([planned[0], planned[-1]]))


if __name__ == "__main__":
    unittest.main()
