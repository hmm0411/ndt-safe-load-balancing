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


class FeatureShiftDiagnosticsTests(unittest.TestCase):
    """Regression tests for the baseline-v1 hot-switch failure."""

    def _records(self, share: float, count: int = 40):
        return [
            {
                "utilization": 0.4 + 0.01 * index,
                "load_share": share + 0.0001 * (index % 3),
                "future_utilization": 0.4 + 0.01 * index,
            }
            for index in range(count)
        ]

    def test_near_constant_feature_is_reported(self):
        from src.twin.forecasting.linear_model import (
            near_constant_features,
        )

        self.assertEqual(
            near_constant_features(
                self._records(0.10),
                ["utilization", "load_share"],
            ),
            ["load_share"],
        )

    def test_shift_report_flags_out_of_distribution_split(self):
        from src.twin.forecasting.linear_model import (
            feature_shift_report,
        )

        report = feature_shift_report(
            self._records(0.10),
            self._records(0.51),
            ["utilization", "load_share"],
        )

        self.assertGreater(
            report["load_share"]["max_abs_z"],
            100.0,
        )
        self.assertEqual(
            report["load_share"][
                "fraction_outside_train_range"
            ],
            1.0,
        )
        self.assertLess(
            report["utilization"]["max_abs_z"],
            3.0,
        )
