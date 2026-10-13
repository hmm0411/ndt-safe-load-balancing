from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from src.twin.forecasting.splitter import parse_run_id


class ForecastDatasetLoadError(ValueError):
    """Raised when frozen Week 8 forecast data cannot be loaded safely."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file without modifying or imputing its contents."""

    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ForecastDatasetLoadError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(record, dict):
                raise ForecastDatasetLoadError(
                    f"{path}:{line_number}: expected JSON object"
                )

            records.append(record)

    return records


def _is_official_run_id(name: str) -> bool:
    try:
        parse_run_id(name)
    except ValueError:
        return False
    return True


def load_official_forecast_runs(
    root: Path,
    *,
    require_valid_snapshot: bool = True,
    expected_run_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Load the official forecast runs (directories named {workload}-rNN-sSEED).

    No interpolation, filling, resampling, or imputation is performed.
    """

    if not root.exists():
        raise FileNotFoundError(root)

    # Official runs are named {workload}-rNN-sSEED (collect_forecast_data.sh).
    # Other experiment folders (capacity runs, _archive, smoke runs) under the
    # same root never match this pattern.
    wanted = (
        None if expected_run_ids is None else set(expected_run_ids)
    )

    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and _is_official_run_id(path.name)
        and (wanted is None or path.name in wanted)
    )

    if not run_dirs:
        raise ForecastDatasetLoadError(
            f"no official forecast runs found under {root}"
        )

    records: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        sample_path = run_dir / "forecast_samples.jsonl"

        if not sample_path.exists():
            raise ForecastDatasetLoadError(
                f"missing forecast samples: {sample_path}"
            )

        run_records = load_jsonl(sample_path)

        for record in run_records:
            if record.get("run_id") != run_dir.name:
                raise ForecastDatasetLoadError(
                    "run_id mismatch: "
                    f"directory={run_dir.name!r}, "
                    f"sample={record.get('run_id')!r}"
                )

            if (
                require_valid_snapshot
                and not bool(record.get("snapshot_valid"))
            ):
                continue

            records.append(record)

    return records
