from __future__ import annotations

import unittest

import pandas as pd

from src.twin.forecasting.splitter import split_forecast_dataset


WORKLOADS = (
    "stable",
    "gradual",
    "burst",
    "oscillating",
    "hot-switch",
)


SEEDS = (
    101,
    102,
    103,
    104,
    105,
)


def build_dummy_dataframe(
    rows_per_segment: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for workload in WORKLOADS:
        for repeat in range(1, 6):
            seed = SEEDS[repeat - 1]
            run_id = (
                f"{workload}-r{repeat:02d}-s{seed}"
            )

            for controller_id in ("c1", "c2"):
                for sequence_id in ("segment-0", "segment-1"):
                    for step in range(rows_per_segment):
                        rows.append(
                            {
                                "run_id": run_id,
                                "workload_type": workload,
                                "repeat": repeat,
                                "seed": seed,
                                "controller_id": controller_id,
                                "sequence_id": (
                                    f"{run_id}-{sequence_id}"
                                ),
                                "observed_at": (
                                    f"2026-10-05T10:00:{step:02d}+00:00"
                                ),
                                "target": float(step),
                            }
                        )

    return pd.DataFrame(rows)


class ForecastSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = build_dummy_dataframe()

    def test_expected_run_counts(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            len(split.train_run_ids),
            12,
        )
        self.assertEqual(
            len(split.validation_run_ids),
            4,
        )
        self.assertEqual(
            len(split.test_run_ids),
            4,
        )
        self.assertEqual(
            len(split.unseen_run_ids),
            5,
        )

    def test_train_validation_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.train_run_ids & split.validation_run_ids,
            set(),
        )

    def test_train_test_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.train_run_ids & split.test_run_ids,
            set(),
        )

    def test_validation_test_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.validation_run_ids & split.test_run_ids,
            set(),
        )

    def test_unseen_train_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.unseen_run_ids & split.train_run_ids,
            set(),
        )

    def test_unseen_validation_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.unseen_run_ids & split.validation_run_ids,
            set(),
        )

    def test_unseen_test_are_disjoint(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertEqual(
            split.unseen_run_ids & split.test_run_ids,
            set(),
        )

    def test_hot_switch_not_in_train_validation_test(self) -> None:
        split = split_forecast_dataset(self.frame)

        self.assertTrue(
            all(
                run_id.startswith("hot-switch-")
                for run_id in split.unseen_run_ids
            )
        )
        self.assertFalse(
            any(
                run_id.startswith("hot-switch-")
                for run_id in (
                    split.train_run_ids
                    | split.validation_run_ids
                    | split.test_run_ids
                )
            )
        )

    def test_same_run_id_cannot_appear_in_two_splits(self) -> None:
        split = split_forecast_dataset(self.frame)

        split_sets = (
            split.train_run_ids,
            split.validation_run_ids,
            split.test_run_ids,
            split.unseen_run_ids,
        )

        for index, left in enumerate(split_sets):
            for right in split_sets[index + 1:]:
                self.assertEqual(
                    left & right,
                    set(),
                )

    def test_all_rows_of_a_run_stay_in_one_split(self) -> None:
        split = split_forecast_dataset(self.frame)

        split_sets = (
            split.train_run_ids,
            split.validation_run_ids,
            split.test_run_ids,
            split.unseen_run_ids,
        )

        for run_id in self.frame["run_id"].unique():
            membership = sum(
                run_id in run_ids
                for run_ids in split_sets
            )
            self.assertEqual(
                membership,
                1,
                msg=f"{run_id} was not assigned atomically",
            )

    def test_cross_sequence_rows_of_same_run_stay_together(self) -> None:
        split = split_forecast_dataset(self.frame)

        for run_id in self.frame["run_id"].unique():
            original_rows = self.frame[
                self.frame["run_id"] == run_id
            ]
            split_name = None

            for name, subset in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
                ("unseen", split.unseen),
            ):
                if run_id in set(subset["run_id"]):
                    split_name = name
                    break

            self.assertIsNotNone(split_name)
            assigned = {
                "train": split.train,
                "validation": split.validation,
                "test": split.test,
                "unseen": split.unseen,
            }[split_name]

            self.assertEqual(
                len(
                    assigned[
                        assigned["run_id"] == run_id
                    ]
                ),
                len(original_rows),
            )

    def test_random_shuffle_is_disabled(self) -> None:
        with self.assertRaises(ValueError):
            split_forecast_dataset(
                self.frame,
                random_shuffle=True,
            )

    def test_missing_required_column(self) -> None:
        frame = self.frame.drop(
            columns=["run_id"]
        )

        with self.assertRaises(ValueError):
            split_forecast_dataset(frame)

    def test_conflicting_repeat_metadata_is_rejected(self) -> None:
        frame = self.frame.copy()

        mask = (
            (frame["run_id"] == "stable-r01-s101")
            & (frame["controller_id"] == "c1")
        )
        frame.loc[mask, "repeat"] = 5

        with self.assertRaises(ValueError):
            split_forecast_dataset(frame)

    def test_conflicting_seed_metadata_is_rejected(self) -> None:
        frame = self.frame.copy()

        mask = frame["run_id"] == "stable-r01-s101"
        frame.loc[mask, "seed"] = 105

        with self.assertRaises(ValueError):
            split_forecast_dataset(frame)

    def test_official_run_id_pattern_is_supported(self) -> None:
        frame = self.frame[
            self.frame["run_id"] == "stable-r01-s101"
        ]

        split = split_forecast_dataset(frame)

        self.assertEqual(
            split.train_run_ids,
            {"stable-r01-s101"},
        )

    def test_hot_switch_all_repeats_are_unseen(self) -> None:
        frame = self.frame[
            self.frame["workload_type"] == "hot-switch"
        ]

        split = split_forecast_dataset(frame)

        self.assertEqual(
            len(split.unseen_run_ids),
            5,
        )
        self.assertEqual(
            split.train_run_ids,
            set(),
        )
        self.assertEqual(
            split.validation_run_ids,
            set(),
        )
        self.assertEqual(
            split.test_run_ids,
            set(),
        )


if __name__ == "__main__":
    unittest.main()
