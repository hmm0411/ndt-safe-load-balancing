import unittest

from src.twin.forecasting.baselines import (
    moving_average_predict,
    persistence_predict,
)


class ForecastBaselineTests(unittest.TestCase):
    def test_persistence_uses_current_utilization(
        self,
    ):
        rows = [
            {
                "sequence_id": "a",
                "utilization": 0.1,
            },
            {
                "sequence_id": "a",
                "utilization": 0.2,
            },
        ]

        prediction = persistence_predict(
            rows
        )

        self.assertEqual(
            prediction.tolist(),
            [0.1, 0.2],
        )

    def test_moving_average_uses_precomputed_feature(
        self,
    ):
        rows = [
            {
                "sequence_id": "a",
                "utilization": 100.0,
                "util_mean_3": 1.5,
            },
            {
                "sequence_id": "a",
                "utilization": 200.0,
                "util_mean_3": 2.5,
            },
            {
                "sequence_id": "b",
                "utilization": 300.0,
                "util_mean_3": 9.5,
            },
        ]

        prediction = moving_average_predict(
            rows,
            window=3,
        )

        self.assertEqual(
            prediction.tolist(),
            [1.5, 2.5, 9.5],
        )

    def test_moving_average_does_not_reconstruct_history(
        self,
    ):
        rows = [
            {
                "sequence_id": "a",
                "utilization": 999.0,
                "util_mean_5": 0.25,
            },
            {
                "sequence_id": "a",
                "utilization": 888.0,
                "util_mean_5": 0.30,
            },
        ]

        prediction = moving_average_predict(
            rows,
            window=5,
        )

        self.assertEqual(
            prediction.tolist(),
            [0.25, 0.30],
        )

    def test_supported_windows(
        self,
    ):
        for window in (3, 5, 10):
            field = f"util_mean_{window}"

            prediction = moving_average_predict(
                [
                    {
                        "sequence_id": "a",
                        "utilization": 99.0,
                        field: 0.42,
                    }
                ],
                window=window,
            )

            self.assertEqual(
                prediction.tolist(),
                [0.42],
            )

    def test_invalid_window_rejected(self):
        for window in (0, -1, 2, 4, 20):
            with self.assertRaises(
                ValueError
            ):
                moving_average_predict(
                    [],
                    window=window,
                )

    def test_missing_moving_average_feature_rejected(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "missing moving-average feature",
        ):
            moving_average_predict(
                [
                    {
                        "sequence_id": "a",
                        "utilization": 1.0,
                    }
                ],
                window=3,
            )

    def test_null_moving_average_feature_rejected(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "must not be null",
        ):
            moving_average_predict(
                [
                    {
                        "sequence_id": "a",
                        "utilization": 1.0,
                        "util_mean_3": None,
                    }
                ],
                window=3,
            )


if __name__ == "__main__":
    unittest.main()
