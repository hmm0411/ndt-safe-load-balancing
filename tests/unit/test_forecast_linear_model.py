import unittest

from src.twin.forecasting.linear_model import (
    fit_linear_regression,
)


class ForecastLinearModelTests(unittest.TestCase):
    def test_fit_and_predict(self):
        rows = [
            {
                "x": 1.0,
                "future_utilization": 2.0,
            },
            {
                "x": 2.0,
                "future_utilization": 4.0,
            },
            {
                "x": 3.0,
                "future_utilization": 6.0,
            },
        ]

        fitted = fit_linear_regression(
            rows,
            ["x"],
        )

        predictions = fitted.predict(
            rows
        )

        self.assertEqual(
            len(predictions),
            3,
        )

        self.assertAlmostEqual(
            predictions[0],
            2.0,
            places=6,
        )

    def test_scaler_fitted_from_train_only(self):
        train = [
            {
                "x": 1.0,
                "future_utilization": 1.0,
            },
            {
                "x": 3.0,
                "future_utilization": 3.0,
            },
        ]

        fitted = fit_linear_regression(
            train,
            ["x"],
        )

        self.assertAlmostEqual(
            float(fitted.scaler.mean_[0]),
            2.0,
        )

        validation = [
            {
                "x": 1000.0,
                "future_utilization": 0.0,
            }
        ]

        fitted.predict(validation)

        self.assertAlmostEqual(
            float(fitted.scaler.mean_[0]),
            2.0,
        )

    def test_empty_training_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            fit_linear_regression(
                [],
                ["x"],
            )


if __name__ == "__main__":
    unittest.main()
