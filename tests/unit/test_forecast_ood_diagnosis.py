from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from scripts.analysis.forecast_ood_diagnosis import diagnose
from src.twin.forecasting.ood import FeatureRanges

RANGES = FeatureRanges(("utilization", "process_memory_rss_mb", "response_mean_5"),
                       (0.0, 50.0, 1.0), (1.0, 51.0, 2.0), 0.10, 3)
MARGINS = {"global": 0.1, "controllers": {"c1": 0.1}}


def rows(run: str, n: int, memory: float = 50.5, response: float = 1.5,
         error: float = 0.0) -> list[dict[str, Any]]:
    return [{"run_id": run, "sequence_id": f"{run}:c1", "controller_id": "c1",
             "observed_at": f"2026-11-24T00:00:{i:02d}Z", "utilization": 0.5,
             "future_utilization": 0.5 + error, "process_memory_rss_mb": memory,
             "response_mean_5": response} for i in range(n)]


def model(records: list[dict[str, Any]]) -> np.ndarray:
    return np.full(len(records), 0.5)


class OodDiagnosisTest(unittest.TestCase):
    def test_runtime_gate_ignores_the_ignored_feature(self) -> None:
        data = rows("a-r05", 10, memory=60.0)          # memory out of range, nothing else
        out = diagnose(data, model, MARGINS, RANGES, ["process_memory_rss_mb"])["overall"]
        self.assertAlmostEqual(out["offline_gate"]["gate_on"], 0.8)   # on from the 3rd step
        self.assertEqual(out["runtime_gate"]["gate_on"], 0.0)
        self.assertEqual(set(out["out_of_range_share"]), {"process_memory_rss_mb"})
        self.assertEqual(out["out_of_range_values"]["process_memory_rss_mb"]["run"]["max"], 60.0)

    def test_upper_coverage_counts_over_prediction_as_covered(self) -> None:
        data = rows("a-r05", 6, error=-0.3)            # U well below U_hat: not a missed overload
        out = diagnose(data, model, MARGINS, RANGES, [])["overall"]["no_gate"]
        self.assertEqual((out["two_sided"], out["upper"]), (0.0, 1.0))

    def test_per_run_blocks_and_gate_restarts_per_sequence(self) -> None:
        data = rows("a-r04", 4, response=9.0) + rows("b-r05", 2, response=9.0)
        out = diagnose(data, model, MARGINS, RANGES, ["process_memory_rss_mb"])
        self.assertEqual(sorted(out["by_run"]), ["a-r04", "b-r05"])
        self.assertAlmostEqual(out["by_run"]["a-r04"]["runtime_gate"]["gate_on"], 0.5)
        self.assertEqual(out["by_run"]["b-r05"]["runtime_gate"]["gate_on"], 0.0)

    def test_empty_split_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            diagnose([], model, MARGINS, RANGES, [])


if __name__ == "__main__":
    unittest.main()
