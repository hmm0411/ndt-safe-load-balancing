from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


OFFICIAL_WORKLOADS = (
    "stable",
    "gradual",
    "burst",
    "oscillating",
    "hot-switch",
)

OFFICIAL_SEEDS = (
    101,
    102,
    103,
    104,
    105,
)

TRAIN_REPEATS = (1, 2, 3)
VALIDATION_REPEATS = (4,)
TEST_REPEATS = (5,)


@dataclass(frozen=True)
class ForecastSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    unseen: pd.DataFrame

    @property
    def train_run_ids(self) -> set[str]:
        return set(self.train["run_id"].astype(str).unique())

    @property
    def validation_run_ids(self) -> set[str]:
        return set(self.validation["run_id"].astype(str).unique())

    @property
    def test_run_ids(self) -> set[str]:
        return set(self.test["run_id"].astype(str).unique())

    @property
    def unseen_run_ids(self) -> set[str]:
        return set(self.unseen["run_id"].astype(str).unique())


def _normalize_workload(value: Any) -> str:
    workload = str(value).strip().lower().replace("_", "-")
    if workload not in OFFICIAL_WORKLOADS:
        raise ValueError(
            f"unsupported workload_type: {value!r}; "
            f"expected one of {OFFICIAL_WORKLOADS}"
        )
    return workload


def _repeat_from_run_id(run_id: str) -> int | None:
    match = re.search(
        r"(?:^|-)r(\d+)(?:-s\d+)?$",
        run_id.strip(),
    )
    if match is None:
        return None
    return int(match.group(1))


def _seed_from_run_id(run_id: str) -> int | None:
    match = re.search(
        r"(?:^|-)s(\d+)$",
        run_id.strip(),
    )
    if match is None:
        return None
    return int(match.group(1))


def _resolve_run_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_id, group in frame.groupby("run_id", sort=False):
        run_id_text = str(run_id).strip()
        if not run_id_text:
            raise ValueError("run_id must not be empty")

        workload_values = (
            group["workload_type"]
            .dropna()
            .map(_normalize_workload)
            .unique()
        )

        if len(workload_values) != 1:
            raise ValueError(
                f"run_id {run_id_text!r} must have exactly one "
                "workload_type"
            )

        workload = str(workload_values[0])

        repeat_candidates: list[int] = []

        # Prefer explicit metadata when present, but cross-check every source.
        for column in ("repeat", "repeat_index"):
            if column not in group.columns:
                continue
            values = group[column].dropna().unique()
            repeat_candidates.extend(
                int(value)
                for value in values
            )

        parsed_repeat = _repeat_from_run_id(run_id_text)
        if parsed_repeat is not None:
            repeat_candidates.append(parsed_repeat)

        repeat_values = set(repeat_candidates)
        if len(repeat_values) != 1:
            raise ValueError(
                f"run_id {run_id_text!r} must resolve to one repeat index"
            )

        repeat = next(iter(repeat_values))
        if repeat not in range(1, 6):
            raise ValueError(
                f"run_id {run_id_text!r} has unsupported repeat index "
                f"{repeat}; official runs use r01-r05"
            )

        seed_values: set[int] = set()
        if "seed" in group.columns:
            seed_values = {
                int(value)
                for value in group["seed"].dropna().unique()
            }

        if len(seed_values) > 1:
            raise ValueError(
                f"run_id {run_id_text!r} has inconsistent seed values"
            )

        seed = next(iter(seed_values)) if seed_values else None
        parsed_seed = _seed_from_run_id(run_id_text)

        if (
            seed is not None
            and parsed_seed is not None
            and seed != parsed_seed
        ):
            raise ValueError(
                f"run_id {run_id_text!r} conflicts with seed metadata"
            )

        if seed is None:
            seed = parsed_seed

        expected_seed = OFFICIAL_SEEDS[repeat - 1]
        if seed is not None and seed != expected_seed:
            raise ValueError(
                f"run_id {run_id_text!r} has seed {seed}; "
                f"repeat {repeat} requires seed {expected_seed}"
            )

        rows.append(
            {
                "run_id": run_id_text,
                "workload_type": workload,
                "repeat_index": repeat,
                "seed": seed,
            }
        )

    return pd.DataFrame(rows)


def _assign_split(
    workload: str,
    repeat_index: int,
) -> str:
    # hot-switch is reserved as an unseen workload and never enters train/val/test.
    if workload == "hot-switch":
        return "unseen"

    if repeat_index in TRAIN_REPEATS:
        return "train"
    if repeat_index in VALIDATION_REPEATS:
        return "validation"
    if repeat_index in TEST_REPEATS:
        return "test"

    raise ValueError(
        f"unsupported repeat index {repeat_index} for workload {workload!r}; "
        "official runs use r01-r05"
    )


def split_forecast_dataset(
    frame: pd.DataFrame,
    *,
    random_shuffle: bool = False,
) -> ForecastSplit:
    """
    Split forecast rows by whole run_id.

    Official strategy:
      stable/gradual/burst/oscillating:
        r01-r03 -> train
        r04     -> validation
        r05     -> test

      hot-switch:
        r01-r05 -> unseen

    A run_id is atomic: all rows belonging to the same run_id stay in
    exactly one split. random_shuffle is intentionally disabled.
    """
    if random_shuffle:
        raise ValueError(
            "random_shuffle must be False for the official forecast split"
        )

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    required_columns = {
        "run_id",
        "workload_type",
    }
    missing = sorted(
        required_columns.difference(frame.columns)
    )
    if missing:
        raise ValueError(
            f"missing required columns: {', '.join(missing)}"
        )

    if frame["run_id"].isna().any():
        raise ValueError("run_id must not contain null values")

    # Resolve metadata once per run so C1/C2 rows and all segments cannot leak.
    run_metadata = _resolve_run_metadata(frame)

    assignments: dict[str, str] = {}
    for row in run_metadata.itertuples(index=False):
        run_id = str(row.run_id)
        split = _assign_split(
            workload=str(row.workload_type),
            repeat_index=int(row.repeat_index),
        )
        assignments[run_id] = split

    normalized_run_ids = frame["run_id"].astype(str).str.strip()
    split_series = normalized_run_ids.map(assignments)

    if split_series.isna().any():
        raise ValueError(
            "unable to assign one or more run_id values"
        )

    train = frame.loc[
        split_series == "train"
    ].copy()
    validation = frame.loc[
        split_series == "validation"
    ].copy()
    test = frame.loc[
        split_series == "test"
    ].copy()
    unseen = frame.loc[
        split_series == "unseen"
    ].copy()

    for result in (
        train,
        validation,
        test,
        unseen,
    ):
        if not result.empty:
            result.reset_index(
                drop=True,
                inplace=True,
            )

    split = ForecastSplit(
        train=train,
        validation=validation,
        test=test,
        unseen=unseen,
    )

    run_sets = (
        split.train_run_ids,
        split.validation_run_ids,
        split.test_run_ids,
        split.unseen_run_ids,
    )

    # A run can contain multiple controllers and continuity segments, but must
    # remain atomic to prevent cross-run or cross-sequence leakage.
    for index, left in enumerate(run_sets):
        for right in run_sets[index + 1:]:
            if left.intersection(right):
                raise ValueError(
                    "the same run_id cannot appear in multiple splits"
                )

    return split


def split_run_ids(
    run_ids: list[str] | tuple[str, ...],
    *,
    random_shuffle: bool = False,
) -> dict[str, set[str]]:
    """
    Convenience API for validating the official run-level split.
    """
    frame = pd.DataFrame(
        {
            "run_id": list(run_ids),
            "workload_type": [
                _workload_from_run_id(run_id)
                for run_id in run_ids
            ],
        }
    )

    split = split_forecast_dataset(
        frame,
        random_shuffle=random_shuffle,
    )

    return {
        "train": split.train_run_ids,
        "validation": split.validation_run_ids,
        "test": split.test_run_ids,
        "unseen": split.unseen_run_ids,
    }


def _workload_from_run_id(run_id: str) -> str:
    normalized = str(run_id).strip().lower().replace("_", "-")

    matches = [
        workload
        for workload in OFFICIAL_WORKLOADS
        if re.search(
            rf"(?:^|-){re.escape(workload)}(?:-|$)",
            normalized,
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            f"cannot resolve workload_type from run_id: {run_id!r}"
        )

    return matches[0]
