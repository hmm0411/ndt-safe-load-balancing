from __future__ import annotations

from typing import Any, Iterable


TRAINABLE_WORKLOADS = (
    "stable",
    "gradual",
    "burst",
    "oscillating",
)


def build_split_run_ids() -> dict[str, set[str]]:
    train = {
        f"forecast-{workload}-2c20s-r{repeat:02d}"
        for workload in TRAINABLE_WORKLOADS
        for repeat in (1, 2, 3)
    }

    validation = {
        f"forecast-{workload}-2c20s-r04"
        for workload in TRAINABLE_WORKLOADS
    }

    test = {
        f"forecast-{workload}-2c20s-r05"
        for workload in TRAINABLE_WORKLOADS
    }

    unseen = {
        f"forecast-hot-switch-2c20s-r{repeat:02d}"
        for repeat in range(1, 6)
    }

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "unseen": unseen,
    }


def validate_split_run_ids(
    splits: dict[str, set[str]],
) -> None:
    required = {
        "train",
        "validation",
        "test",
        "unseen",
    }

    if set(splits) != required:
        raise ValueError(
            "split names must be exactly "
            "train, validation, test, unseen"
        )

    names = sorted(splits)

    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = (
                splits[left]
                & splits[right]
            )

            if overlap:
                raise ValueError(
                    "run overlap detected between "
                    f"{left} and {right}: "
                    f"{sorted(overlap)}"
                )


def split_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    splits = build_split_run_ids()
    validate_split_run_ids(splits)

    run_to_split = {
        run_id: split_name
        for split_name, run_ids in splits.items()
        for run_id in run_ids
    }

    output: dict[
        str,
        list[dict[str, Any]],
    ] = {
        "train": [],
        "validation": [],
        "test": [],
        "unseen": [],
    }

    seen_runs: set[str] = set()

    for record in records:
        run_id = str(
            record.get("run_id", "")
        )

        if run_id not in run_to_split:
            raise ValueError(
                f"unassigned run_id: {run_id!r}"
            )

        split_name = run_to_split[run_id]

        output[split_name].append(
            dict(record)
        )

        seen_runs.add(run_id)

    expected_runs = set(run_to_split)

    if seen_runs != expected_runs:
        missing = sorted(
            expected_runs - seen_runs
        )

        raise ValueError(
            f"expected runs missing: {missing}"
        )

    return output
