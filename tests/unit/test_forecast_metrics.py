import unittest

from src.twin.forecasting.metrics import (
    evaluate_regression,
)


class ForecastMetricTests(unittest.TestCase):
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
