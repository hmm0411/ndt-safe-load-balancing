import unittest

import numpy as np

from src.twin.forecasting.evaluator import (
    evaluate_by_workload,
    evaluate_predictions,
)


class ForecastEvaluatorTests(unittest.TestCase):
    def test_evaluate_predictions(self):
        rows = [
            {
                "future_utilization": 0.1,
            },
            {
                "future_utilization": 0.2,
            },
        ]

        metrics = evaluate_predictions(
            rows,
            np.asarray(
                [0.1, 0.2]
            ),
        )

        self.assertAlmostEqual(
            metrics["rmse"],
            0.0,
        )

    def test_length_mismatch_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            evaluate_predictions(
                [
                    {
                        "future_utilization":
                            0.1,
                    }
                ],
                np.asarray(
                    [0.1, 0.2]
                ),
            )

    def test_workload_metrics_are_separate(self):
        rows = [
            {
                "workload_type": "stable",
                "future_utilization": 0.1,
            },
            {
                "workload_type": "stable",
                "future_utilization": 0.2,
            },
            {
                "workload_type": "burst",
                "future_utilization": 0.5,
            },
            {
                "workload_type": "burst",
                "future_utilization": 0.6,
            },
        ]

        result = evaluate_by_workload(
            rows,
            np.asarray(
                [0.1, 0.2, 0.5, 0.6]
            ),
        )

        self.assertEqual(
            set(result),
            {
                "stable",
                "burst",
            },
        )


if __name__ == "__main__":
    unittest.main()
