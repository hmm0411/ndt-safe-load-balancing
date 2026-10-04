#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.twin.forecasting.dataset import assign_sequence_ids  # noqa: E402
from src.twin.forecasting.features import (  # noqa: E402
    build_features,
    feature_names,
)
from src.twin.forecasting.labels import (  # noqa: E402
    attach_future_utilization,
)
from src.twin.forecasting.loader import (  # noqa: E402
    load_official_forecast_runs,
)


WORKLOADS = (
    "stable",
    "gradual",
    "burst",
    "oscillating",
    "hot-switch",
)

REPEATS = range(1, 6)

MAX_GAP_SECONDS = 2.5
HISTORY_STEPS = 10

FORECAST_CONFIG = Path(
    "configs/forecasting/forecast.yaml"
)


def load_provisional_horizon(
    config_path: Path,
) -> tuple[int, float]:
    import yaml

    config = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    horizon = config["horizon"]
    provisional = horizon["provisional"]

    if horizon.get("finalized") is not False:
        raise ValueError(
            "Week 9 provisional dataset requires "
            "horizon.finalized=false"
        )

    if provisional.get("enabled") is not True:
        raise ValueError(
            "provisional horizon must be enabled"
        )

    steps = int(
        provisional["horizon_steps"]
    )
    seconds = float(
        provisional[
            "effective_horizon_seconds"
        ]
    )

    if steps <= 0:
        raise ValueError(
            "horizon_steps must be positive"
        )

    if seconds <= 0:
        raise ValueError(
            "effective_horizon_seconds "
            "must be positive"
        )

    return steps, seconds


HORIZON_STEPS, HORIZON_SECONDS = (
    load_provisional_horizon(
        FORECAST_CONFIG
    )
)

DATASET_VERSION = (
    f"forecast-dataset-v1-"
    f"h{HORIZON_STEPS}-provisional"
)


def expected_run_ids() -> set[str]:
    return {
        (
            f"forecast-{workload}-"
            f"2c20s-r{repeat:02d}"
        )
        for workload in WORKLOADS
        for repeat in REPEATS
    }


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            "cannot write empty processed dataset"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    preferred = [
        "run_id",
        "controller_id",
        "sequence_id",
        "sequence_index",
        "observed_at",
        "workload_type",
        "workload_phase",
        "snapshot_id",
        *feature_names(),
        "future_utilization",
        "forecast_horizon_steps",
    ]

    all_fields: set[str] = set()

    for row in rows:
        all_fields.update(row)

    fields = [
        field
        for field in preferred
        if field in all_fields
    ]

    remaining = sorted(
        all_fields - set(fields)
    )

    fields.extend(remaining)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )



def has_complete_features(
    row: dict[str, Any],
) -> bool:
    return all(
        row.get(name) is not None
        for name in feature_names()
    )


def finalize_ml_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep only rows that have both:
    - complete production predictor features;
    - a future U(t+h) label.

    Filtering intentionally occurs only after feature engineering
    and future-label attachment.
    """
    return [
        row
        for row in rows
        if has_complete_features(row)
        and row.get("future_utilization")
        is not None
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/experiment_runs"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/forecasting/processed/"
            + DATASET_VERSION
        ),
    )

    args = parser.parse_args()

    raw_all = load_official_forecast_runs(
        args.raw_root,
        require_valid_snapshot=False,
    )

    raw_rows = len(raw_all)

    valid_raw = [
        row
        for row in raw_all
        if bool(row.get("snapshot_valid"))
    ]

    valid_raw_rows = len(valid_raw)

    actual_runs = {
        str(row["run_id"])
        for row in raw_all
    }

    expected_runs = expected_run_ids()

    if actual_runs != expected_runs:
        missing = sorted(
            expected_runs - actual_runs
        )
        unexpected = sorted(
            actual_runs - expected_runs
        )

        raise RuntimeError(
            "official run set mismatch; "
            f"missing={missing}, "
            f"unexpected={unexpected}"
        )

    segmented = assign_sequence_ids(
        valid_raw,
        max_gap_seconds=MAX_GAP_SECONDS,
    )

    sequence_ids = {
        str(row["sequence_id"])
        for row in segmented
    }

    controller_series = {
        (
            str(row["run_id"]),
            str(row["controller_id"]),
        )
        for row in segmented
    }

    sequence_count = len(sequence_ids)

    gap_break_count = (
        sequence_count
        - len(controller_series)
    )

    featured = build_features(
        segmented
    )

    labeled = attach_future_utilization(
        featured,
        horizon_steps=HORIZON_STEPS,
    )

    feature_rows = sum(
        1
        for row in featured
        if has_complete_features(row)
    )

    dropped_for_history = (
        valid_raw_rows - feature_rows
    )

    label_ready_rows = sum(
        1
        for row in labeled
        if row.get("future_utilization")
        is not None
    )

    dropped_for_future_label = (
        len(labeled) - label_ready_rows
    )

    processed = finalize_ml_rows(
        labeled
    )

    output_dir = args.output_dir

    dataset_path = (
        output_dir / "forecast_dataset.csv"
    )

    quality_path = (
        output_dir
        / "dataset_quality_report.json"
    )

    manifest_path = (
        output_dir
        / "forecast_dataset_manifest.json"
    )

    write_csv(
        dataset_path,
        processed,
    )

    workload_counts = Counter(
        str(row["workload_type"])
        for row in processed
    )

    controller_counts = Counter(
        str(row["controller_id"])
        for row in processed
    )

    quality_report = {
        "dataset_version": DATASET_VERSION,
        "raw_rows": raw_rows,
        "valid_raw_rows": valid_raw_rows,
        "sequence_count": sequence_count,
        "gap_break_count": gap_break_count,
        "feature_rows": feature_rows,
        "processed_rows": len(processed),
        "dropped_for_history":
            dropped_for_history,
        "dropped_for_future_label":
            dropped_for_future_label,
        "official_run_count":
            len(actual_runs),
        "controller_series_count":
            len(controller_series),
        "workload_row_counts":
            dict(sorted(workload_counts.items())),
        "controller_row_counts":
            dict(sorted(controller_counts.items())),
        "future_label_cross_run": False,
        "raw_data_imputed": False,
    }

    manifest = {
        "schema_version": "1.0",
        "dataset_version": DATASET_VERSION,
        "source_dataset":
            "forecast-raw-v1",
        "source_root": str(args.raw_root),
        "processed_dataset":
            str(dataset_path),
        "git_commit_at_build": git_head(),
        "sampling_interval_seconds": 1.0,
        "max_gap_seconds":
            MAX_GAP_SECONDS,
        "history_steps":
            HISTORY_STEPS,
        "horizon": {
            "status": "provisional",
            "steps": HORIZON_STEPS,
            "effective_seconds":
                HORIZON_SECONDS,
            "method":
                "measured_runtime_p95_lower_bound",
            "reason":
                "decision_p95_ms_not_yet_available",
            "runtime_inputs_path":
                "data/benchmarks/runtime/"
                "forecast_horizon_inputs.json",
        },
        "require_valid_snapshot": True,
        "raw_data_imputed": False,
        "feature_schema_version": "1.0",
        "feature_names":
            list(feature_names()),
        "target":
            "future_utilization",
        "official_run_ids":
            sorted(actual_runs),
        "row_count":
            len(processed),
    }

    write_json(
        quality_path,
        quality_report,
    )

    write_json(
        manifest_path,
        manifest,
    )

    print("=" * 72)
    print("WEEK 9 PROCESSED DATASET BUILD")
    print("=" * 72)

    print(
        f"Dataset version           : "
        f"{DATASET_VERSION}"
    )
    print(
        f"Raw rows                  : "
        f"{raw_rows}"
    )
    print(
        f"Valid raw rows            : "
        f"{valid_raw_rows}"
    )
    print(
        f"Sequences                 : "
        f"{sequence_count}"
    )
    print(
        f"Gap breaks                : "
        f"{gap_break_count}"
    )
    print(
        f"Feature rows              : "
        f"{feature_rows}"
    )
    print(
        f"Dropped for history       : "
        f"{dropped_for_history}"
    )
    print(
        f"Dropped for future label  : "
        f"{dropped_for_future_label}"
    )
    print(
        f"Processed rows            : "
        f"{len(processed)}"
    )

    print()
    print("Output:")
    print(f"  {dataset_path}")
    print(f"  {quality_path}")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
