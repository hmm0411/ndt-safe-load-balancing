import unittest

from src.twin.forecasting.labels import (
    attach_future_utilization,
)


def row(
    sequence_id: str,
    index: int,
):
    return {
        "run_id": sequence_id.split(":")[0],
        "controller_id":
            sequence_id.split(":")[1],
        "sequence_id": sequence_id,
        "observed_at":
            f"2026-01-01T00:00:{index:02d}+00:00",
        "utilization": float(index),
    }


class ForecastLabelTests(unittest.TestCase):
    def test_horizon_three_preserves_rows(self):
        rows = [
            row("run-a:c1:0", index)
            for index in range(10)
        ]

        result = attach_future_utilization(
            rows,
            horizon_steps=3,
        )

        self.assertEqual(
            len(result),
            10,
        )

        self.assertEqual(
            result[0]["future_utilization"],
            3.0,
        )

        self.assertEqual(
            result[6]["future_utilization"],
            9.0,
        )

        for item in result[-3:]:
            self.assertIsNone(
                item["future_utilization"]
            )

    def test_label_never_crosses_sequence(self):
        rows = [
            row("run-a:c1:0", index)
            for index in range(3)
        ]

        rows += [
            row("run-b:c1:0", index)
            for index in range(3, 6)
        ]

        result = attach_future_utilization(
            rows,
            horizon_steps=3,
        )

        self.assertEqual(
            len(result),
            6,
        )

        self.assertTrue(
            all(
                item["future_utilization"]
                is None
                for item in result
            )
        )

    def test_last_row_of_run_does_not_use_next_run(self):
        rows = [
            row("run-a:c1:0", 0),
            row("run-a:c1:0", 1),
            row("run-b:c1:0", 2),
            row("run-b:c1:0", 3),
        ]

        result = attach_future_utilization(
            rows,
            horizon_steps=1,
        )

        run_a = [
            item
            for item in result
            if item["run_id"] == "run-a"
        ]

        self.assertEqual(
            run_a[0]["future_utilization"],
            1.0,
        )

        self.assertIsNone(
            run_a[-1]["future_utilization"]
        )

        self.assertNotEqual(
            run_a[-1]["future_utilization"],
            2.0,
        )

    def test_invalid_horizon_rejected(self):
        with self.assertRaises(ValueError):
            attach_future_utilization(
                [],
                horizon_steps=0,
            )


if __name__ == "__main__":
    unittest.main()
