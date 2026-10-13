import unittest

from src.twin.forecasting.dataset import (
    assign_sequence_ids,
)


def sample(
    run_id: str,
    controller_id: str,
    observed_at: str,
):
    return {
        "run_id": run_id,
        "controller_id": controller_id,
        "observed_at": observed_at,
    }


class ForecastDatasetTests(unittest.TestCase):
    def test_continuous_records_share_sequence(self):
        rows = [
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:01+00:00",
            ),
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:02+00:00",
            ),
        ]

        result = assign_sequence_ids(
            rows,
            max_gap_seconds=2.5,
        )

        self.assertEqual(
            result[0]["sequence_id"],
            "run-a:c1:0",
        )

        self.assertEqual(
            result[1]["sequence_id"],
            "run-a:c1:0",
        )

    def test_gap_starts_new_sequence(self):
        rows = [
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:01+00:00",
            ),
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:05+00:00",
            ),
        ]

        result = assign_sequence_ids(
            rows,
            max_gap_seconds=2.5,
        )

        self.assertEqual(
            result[0]["sequence_id"],
            "run-a:c1:0",
        )

        self.assertEqual(
            result[1]["sequence_id"],
            "run-a:c1:1",
        )

    def test_sequences_never_cross_controller(self):
        rows = [
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:01+00:00",
            ),
            sample(
                "run-a",
                "c2",
                "2026-01-01T00:00:01+00:00",
            ),
        ]

        result = assign_sequence_ids(
            rows,
            max_gap_seconds=2.5,
        )

        ids = {
            row["sequence_id"]
            for row in result
        }

        self.assertEqual(
            ids,
            {
                "run-a:c1:0",
                "run-a:c2:0",
            },
        )

    def test_sequences_never_cross_run(self):
        rows = [
            sample(
                "run-a",
                "c1",
                "2026-01-01T00:00:01+00:00",
            ),
            sample(
                "run-b",
                "c1",
                "2026-01-01T00:00:02+00:00",
            ),
        ]

        result = assign_sequence_ids(
            rows,
            max_gap_seconds=2.5,
        )

        ids = {
            row["sequence_id"]
            for row in result
        }

        self.assertEqual(
            ids,
            {
                "run-a:c1:0",
                "run-b:c1:0",
            },
        )


if __name__ == "__main__":
    unittest.main()
