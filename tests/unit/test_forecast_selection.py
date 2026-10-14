from __future__ import annotations

import unittest

import numpy as np

from src.twin.forecasting.features import feature_names
from src.twin.forecasting.selection import (
    DEFAULT_WEIGHTS,
    measure_latency_ms,
    select_model,
    selection_score,
    single_sample_predictors,
)

BASELINE = {"rmse": 0.10, "p95_absolute_error": 0.20, "mae": 0.05}


class _Constant:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full(len(x), self.value)


class _Identity:
    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


class SelectionScoreTest(unittest.TestCase):
    def test_persistence_scores_090_plus_latency(self) -> None:
        score = selection_score({**BASELINE, "inference_p95_ms": 0.0}, BASELINE)
        self.assertAlmostEqual(score, 0.90)

    def test_errors_are_normalized_by_persistence(self) -> None:
        metrics = {"rmse": 0.05, "p95_absolute_error": 0.10, "mae": 0.025,
                   "inference_p95_ms": 100.0}
        expected = 0.45 * 0.5 + 0.30 * 0.5 + 0.15 * 0.5 + 0.10 * 0.1
        self.assertAlmostEqual(selection_score(metrics, BASELINE), expected)

    def test_latency_term_is_capped_at_one(self) -> None:
        slow = selection_score({**BASELINE, "inference_p95_ms": 5000.0}, BASELINE)
        self.assertAlmostEqual(slow, 0.90 + 0.10)

    def test_weights_must_sum_to_one(self) -> None:
        weights = {**DEFAULT_WEIGHTS, "rmse": 0.5}
        with self.assertRaises(ValueError):
            selection_score({**BASELINE, "inference_p95_ms": 1.0}, BASELINE, weights)

    def test_zero_baseline_raises(self) -> None:
        with self.assertRaises(ValueError):
            selection_score({**BASELINE, "inference_p95_ms": 1.0}, {**BASELINE, "rmse": 0.0})


class SelectModelTest(unittest.TestCase):
    def test_lowest_score_wins(self) -> None:
        result = select_model({"persistence": 0.9, "random_forest": 0.42, "xgboost": 0.40})
        self.assertEqual(result.selected, "xgboost")
        self.assertEqual(result.ranking, ["xgboost", "random_forest", "persistence"])

    def test_near_tie_between_trees_is_flagged(self) -> None:
        result = select_model({"random_forest": 0.405, "xgboost": 0.40}, tie_tolerance=0.01)
        self.assertTrue(result.tree_near_tie)
        self.assertAlmostEqual(result.tree_gap or 0.0, 0.005)

    def test_clear_gap_is_not_a_tie(self) -> None:
        result = select_model({"random_forest": 0.45, "xgboost": 0.40})
        self.assertFalse(result.tree_near_tie)

    def test_empty_or_nan_scores_raise(self) -> None:
        with self.assertRaises(ValueError):
            select_model({})
        with self.assertRaises(ValueError):
            select_model({"xgboost": float("nan")})


class LatencyTest(unittest.TestCase):
    def test_counts_warmup_and_repeats(self) -> None:
        calls: list[int] = []
        stats = measure_latency_ms(lambda: calls.append(1), warmup=5, repeats=20)
        self.assertEqual(len(calls), 25)
        self.assertLessEqual(stats["p50_ms"], stats["p95_ms"])
        self.assertLessEqual(stats["p95_ms"], stats["p99_ms"])

    def test_invalid_repeats_raise(self) -> None:
        with self.assertRaises(ValueError):
            measure_latency_ms(lambda: None, repeats=0)


class SingleSamplePredictorsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.names = list(feature_names())
        row = np.zeros((1, len(self.names)))
        row[0, self.names.index("utilization")] = 0.6
        row[0, self.names.index("utilization_lag_1")] = 0.3
        row[0, self.names.index("utilization_lag_2")] = 0.0
        self.row = row

    def test_baselines_read_the_row(self) -> None:
        predictors = single_sample_predictors(self.names, self.row)
        self.assertAlmostEqual(predictors["persistence"](), 0.6)
        self.assertAlmostEqual(predictors["moving_average_w3"](), 0.3)

    def test_tree_models_add_delta_to_current_utilization(self) -> None:
        tree = _Constant(0.1)
        predictors = single_sample_predictors(self.names, self.row, xgboost=tree)
        self.assertAlmostEqual(predictors["xgboost"](), 0.7)
        self.assertEqual(tree.calls, 1)

    def test_linear_regression_predicts_level_after_scaling(self) -> None:
        predictors = single_sample_predictors(
            self.names, self.row, linear_model=_Constant(0.8), scaler=_Identity()
        )
        self.assertAlmostEqual(predictors["linear_regression"](), 0.8)

    def test_row_shape_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            single_sample_predictors(self.names, np.zeros((2, len(self.names))))


if __name__ == "__main__":
    unittest.main()
