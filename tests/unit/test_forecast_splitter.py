import unittest

from src.twin.forecasting.splitter import (
    build_split_run_ids,
    split_records,
    validate_split_run_ids,
)


class ForecastSplitterTests(unittest.TestCase):
    def test_expected_run_counts(self):
        splits = build_split_run_ids()

        self.assertEqual(
            len(splits["train"]),
            12,
        )
        self.assertEqual(
            len(splits["validation"]),
            4,
        )
        self.assertEqual(
            len(splits["test"]),
            4,
        )
        self.assertEqual(
            len(splits["unseen"]),
            5,
        )

    def test_splits_do_not_overlap(self):
        splits = build_split_run_ids()

        validate_split_run_ids(splits)

        names = sorted(splits)

        for index, left in enumerate(names):
            for right in names[index + 1:]:
                self.assertFalse(
                    splits[left]
                    & splits[right]
                )

    def test_hot_switch_is_unseen_only(self):
        splits = build_split_run_ids()

        for name in (
            "train",
            "validation",
            "test",
        ):
            self.assertFalse(
                any(
                    "hot-switch" in run_id
                    for run_id
                    in splits[name]
                )
            )

        self.assertTrue(
            all(
                "hot-switch" in run_id
                for run_id
                in splits["unseen"]
            )
        )

    def test_split_records(self):
        splits = build_split_run_ids()

        all_runs = sorted(
            set().union(
                *splits.values()
            )
        )

        rows = [
            {
                "run_id": run_id,
                "value": index,
            }
            for index, run_id in enumerate(
                all_runs
            )
        ]

        result = split_records(rows)

        self.assertEqual(
            {
                name: len(part)
                for name, part
                in result.items()
            },
            {
                "train": 12,
                "validation": 4,
                "test": 4,
                "unseen": 5,
            },
        )


if __name__ == "__main__":
    unittest.main()
