#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.twin.forecasting.versions import dataset_version
from src.twin.forecasting.splitter import (
    build_split_run_ids,
    split_records,
)


DATASET_VERSION = dataset_version()

PROCESSED_ROOT = Path(
    "data/forecasting/processed"
) / DATASET_VERSION

DATASET_PATH = (
    PROCESSED_ROOT / "forecast_dataset.csv"
)

QUALITY_PATH = (
    PROCESSED_ROOT / "dataset_quality_report.json"
)

SPLIT_MANIFEST_PATH = (
    PROCESSED_ROOT / "split_manifest.json"
)


def load_csv(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    rows = load_csv(DATASET_PATH)

    parts = split_records(rows)
    run_ids = build_split_run_ids()

    split_counts = {
        name: len(parts[name])
        for name in (
            "train",
            "validation",
            "test",
            "unseen",
        )
    }

    expected_counts = {
        "train": 4008,
        "validation": 1336,
        "test": 1336,
        "unseen": 1670,
    }

    if split_counts != expected_counts:
        raise RuntimeError(
            "unexpected split row counts: "
            f"{split_counts}"
        )

    names = list(run_ids)

    overlap_detected = any(
        bool(run_ids[left] & run_ids[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    )

    if overlap_detected:
        raise RuntimeError(
            "run overlap detected"
        )

    manifest = {
        "schema_version": "1.0",
        "dataset_version": DATASET_VERSION,
        "split_strategy":
            "explicit_run_level_temporal_holdout",
        "random_shuffle": False,
        "train": {
            "run_ids":
                sorted(run_ids["train"]),
            "run_count":
                len(run_ids["train"]),
            "row_count":
                split_counts["train"],
        },
        "validation": {
            "run_ids":
                sorted(run_ids["validation"]),
            "run_count":
                len(run_ids["validation"]),
            "row_count":
                split_counts["validation"],
        },
        "test": {
            "run_ids":
                sorted(run_ids["test"]),
            "run_count":
                len(run_ids["test"]),
            "row_count":
                split_counts["test"],
        },
        "unseen": {
            "run_ids":
                sorted(run_ids["unseen"]),
            "run_count":
                len(run_ids["unseen"]),
            "row_count":
                split_counts["unseen"],
            "purpose":
                "unseen_hot_switch_generalization",
        },
        "run_overlap_detected":
            overlap_detected,
    }

    SPLIT_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    quality = json.loads(
        QUALITY_PATH.read_text(
            encoding="utf-8"
        )
    )

    quality.update({
        "train_rows":
            split_counts["train"],
        "validation_rows":
            split_counts["validation"],
        "test_rows":
            split_counts["test"],
        "unseen_rows":
            split_counts["unseen"],
        "run_overlap_detected":
            overlap_detected,
        "random_shuffle":
            False,
    })

    QUALITY_PATH.write_text(
        json.dumps(
            quality,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("WEEK 9 SPLIT ARTIFACT BUILD")
    print("=" * 72)

    for name in (
        "train",
        "validation",
        "test",
        "unseen",
    ):
        print(
            f"{name:12}: "
            f"{len(run_ids[name]):2} runs, "
            f"{split_counts[name]:4} rows"
        )

    print()
    print(
        "Run overlap detected :",
        overlap_detected,
    )
    print("Random shuffle        :", False)
    print()
    print(
        "Split manifest        :",
        SPLIT_MANIFEST_PATH,
    )
    print(
        "QA report updated     :",
        QUALITY_PATH,
    )


if __name__ == "__main__":
    main()
