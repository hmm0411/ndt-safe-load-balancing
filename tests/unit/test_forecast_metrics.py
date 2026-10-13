import math
import unittest

from src.twin.forecasting.metrics import (
    evaluate_regression,
    compute_metrics,
    mae,
    mape_percent,
    p95_abs_error,
    percentile,
    r2_score,
    rmse,
)

Y_TRUE = [0.5, 0.6, 0.7]
Y_PRED = [0.4, 0.6, 0.8]  # abs errors: 0.1, 0.0, 0.1


class ForecastMetricsTest(unittest.TestCase):
    def test_mae(self) -> None:
        self.assertAlmostEqual(mae(Y_TRUE, Y_PRED), 0.2 / 3)  # 0.066667

    def test_rmse(self) -> None:
        self.assertAlmostEqual(rmse(Y_TRUE, Y_PRED), math.sqrt(0.02 / 3))  # 0.081650

    def test_mape_percent(self) -> None:
        expected = 100 * (0.1 / 0.5 + 0.0 + 0.1 / 0.7) / 3  # 11.42857
        self.assertAlmostEqual(mape_percent(Y_TRUE, Y_PRED), expected)

    def test_r2_is_zero_when_residual_equals_total_variance(self) -> None:
        # SS_res = 0.02, SS_tot = 0.02 -> R2 = 0
        self.assertAlmostEqual(r2_score(Y_TRUE, Y_PRED), 0.0)

    def test_r2_perfect_prediction(self) -> None:
        self.assertAlmostEqual(r2_score(Y_TRUE, Y_TRUE), 1.0)

    def test_r2_known_value(self) -> None:
        # y=[1,2,3,4], pred=[1,2,3,5]: SS_res=1, SS_tot=5 -> 0.8
        self.assertAlmostEqual(r2_score([1, 2, 3, 4], [1, 2, 3, 5]), 0.8)

    def test_r2_constant_target(self) -> None:
        self.assertEqual(r2_score([1, 1, 1], [1, 1, 1]), 1.0)
        self.assertTrue(math.isnan(r2_score([1, 1, 1], [1, 1, 2])))

    def test_p95_abs_error_interpolates(self) -> None:
        # errors 0.0..0.4 -> position 0.95*4=3.8 -> 0.3 + 0.8*0.1 = 0.38
        y_true = [0.0] * 5
        y_pred = [0.0, 0.1, 0.2, 0.3, 0.4]
        self.assertAlmostEqual(p95_abs_error(y_true, y_pred), 0.38)

    def test_p95_small_sample(self) -> None:
        self.assertAlmostEqual(p95_abs_error(Y_TRUE, Y_PRED), 0.1)

    def test_percentile_edges(self) -> None:
        self.assertEqual(percentile([3, 1, 2], 0), 1)
        self.assertEqual(percentile([3, 1, 2], 100), 3)

    def test_mape_skips_zero_targets(self) -> None:
        self.assertAlmostEqual(mape_percent([0.0, 0.5], [0.3, 0.4]), 20.0)
        with self.assertRaises(ValueError):
            mape_percent([0.0, 0.0], [0.1, 0.2])

    def test_compute_metrics_bundle(self) -> None:
        result = compute_metrics(Y_TRUE, Y_PRED)
        self.assertEqual(result.sample_count, 3)
        self.assertAlmostEqual(result.mae, 0.2 / 3)
        self.assertIn("p95_abs_error", result.to_dict())

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            mae([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            mae([], [])
        with self.assertRaises(ValueError):
            mae([float("nan")], [1.0])


class EvaluateRegressionTests(unittest.TestCase):
    def test_perfect_prediction(self):
        result = evaluate_regression(
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        )

        self.assertAlmostEqual(
            result["mae"],
            0.0,
        )
        self.assertAlmostEqual(
            result["rmse"],
            0.0,
        )
        self.assertAlmostEqual(
            result["mape"],
            0.0,
        )
        self.assertAlmostEqual(
            result["r2"],
            1.0,
        )
        self.assertAlmostEqual(
            result["p95_absolute_error"],
            0.0,
        )

    def test_zero_target_is_safe_for_mape(self):
        result = evaluate_regression(
            [0.0, 1.0],
            [0.0, 1.0],
        )

        self.assertEqual(
            result["mape"],
            0.0,
        )

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_regression(
                [1.0],
                [1.0, 2.0],
            )


if __name__ == "__main__":
    unittest.main()
