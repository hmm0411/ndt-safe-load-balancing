"""M4 tier 2 on the real test runs (VM only: needs data/ outside git).

    python -m unittest tests.integration.test_feature_parity -v
"""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

from scripts.analysis.feature_parity import compare, key_of, replay_run
from src.twin.forecasting.features import feature_names
from src.twin.forecasting.loader import load_jsonl
from src.twin.forecasting.splitter import assign_split
from src.twin.forecasting.versions import PROCESSED_ROOT, dataset_version

MODEL = Path("data/models/forecasting/forecast-v1")
DATASET = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
RAW_ROOT = Path("data/experiment_runs")


@unittest.skipUnless(DATASET.is_file() and RAW_ROOT.is_dir(), "dataset is only on the VM")
class FeatureParityTest(unittest.TestCase):
    def test_test_split_runtime_features_equal_dataset(self) -> None:
        with DATASET.open(encoding="utf-8", newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if assign_split(r["run_id"]) == "test"]
        expected = {key_of(r["run_id"], r["controller_id"], r["observed_at"]): r for r in rows}
        runtime: dict[tuple[str, str, int], list[float]] = {}
        for run_id in sorted({r["run_id"] for r in rows}):
            runtime.update(replay_run(load_jsonl(RAW_ROOT / run_id / "forecast_samples.jsonl"), MODEL))
        result = compare(expected, runtime, list(feature_names()))
        self.assertEqual(result["mismatched_rows"], 0, result["first_mismatches"])
        self.assertGreaterEqual(result["compared_rows"], 0.9 * result["expected_rows"])


if __name__ == "__main__":
    unittest.main()
