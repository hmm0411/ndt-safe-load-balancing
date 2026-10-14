#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.forecasting.raw_validator import validate_raw_samples  # noqa: E402


REQUIRED_FIELDS = {
    "run_id",
    "controller_id",
    "observed_at",
    "workload_type",
    "workload_phase",
    "processed_packet_in_rate",
    "flow_mod_rate",
    "process_cpu_percent",
    "process_memory_rss_mb",
    "response_p95_ms",
    "managed_switch_count",
    "safe_capacity_pps",
    "utilization",
    "max_switch_control_load_share",
    "age_of_twin_ms",
    "twinning_rate",
    "completeness_ratio",
    "synchronization_jitter_ms",
    "snapshot_valid",
    "snapshot_id",
}


# Deterministic run id: {workload}-r{NN}-s{seed}. The workload may contain "-"
# (hot-switch), so the pattern is anchored on the trailing "-rNN-sSEED".
RUN_ID_PATTERN = re.compile(
    r"^(?P<workload>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)"
    r"-r(?P<repeat>\d{2})-s(?P<seed>\d+)$"
)

# Metadata contract of an official run (config: metadata.required_fields).
# The runner may write an older/alternative name, so aliases are accepted.
METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "controller_ids": ("controller_ids", "controllers"),
    "repeat": ("repeat", "repeat_index"),
    "finished_at": ("finished_at", "ended_at"),
}


@dataclass(frozen=True)
class CollectionIssue:
    code: str
    message: str
    controller_id: str | None = None


@dataclass(frozen=True)
class CollectionValidationResult:
    valid: bool
    run_id: str
    total_samples: int
    samples_per_controller: dict[str, int]
    coverage_per_controller: dict[str, float]
    issues: tuple[CollectionIssue, ...]


@dataclass(frozen=True)
class OfficialValidationResult:
    valid: bool
    expected_runs: int
    found_runs: int
    expected_samples: int
    total_samples: int
    runs: tuple[CollectionValidationResult, ...]
    issues: tuple[CollectionIssue, ...]


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or value == []


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc

            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: row must be a JSON object"
                )

            rows.append(value)

    return rows


def validate_collection(
    *,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    min_coverage: float = 0.95,
    utilization_tolerance: float = 0.02,
    max_gap_seconds: float = 2.5,
    require_fixed_ownership: bool = True,
    metadata_required_fields: tuple[str, ...] = (),
) -> CollectionValidationResult:
    issues: list[CollectionIssue] = []

    run_id = str(metadata.get("run_id", "")).strip()
    workload_type = str(metadata.get("workload_type", "")).strip()

    try:
        duration_seconds = float(metadata["duration_seconds"])
        sampling_interval_seconds = float(
            metadata["sampling_interval_seconds"]
        )
    except (KeyError, TypeError, ValueError):
        duration_seconds = 0.0
        sampling_interval_seconds = 0.0
        issues.append(
            CollectionIssue(
                code="metadata_invalid",
                message=(
                    "duration_seconds and sampling_interval_seconds "
                    "must be present and positive"
                ),
            )
        )

    if duration_seconds <= 0 or sampling_interval_seconds <= 0:
        if not any(issue.code == "metadata_invalid" for issue in issues):
            issues.append(
                CollectionIssue(
                    code="metadata_invalid",
                    message=(
                        "duration_seconds and sampling_interval_seconds "
                        "must be positive"
                    ),
                )
            )

    controllers_raw = metadata.get("controllers", [])
    if not isinstance(controllers_raw, list):
        controllers_raw = []

    expected_controllers = tuple(
        str(value).strip()
        for value in controllers_raw
        if str(value).strip()
    )

    if not run_id:
        issues.append(
            CollectionIssue(
                code="run_id_missing",
                message="metadata.run_id must not be empty",
            )
        )

    if not workload_type:
        issues.append(
            CollectionIssue(
                code="workload_type_missing",
                message="metadata.workload_type must not be empty",
            )
        )

    if not expected_controllers:
        issues.append(
            CollectionIssue(
                code="controllers_missing",
                message="metadata.controllers must not be empty",
            )
        )

    if metadata.get("migration_enabled") is not False:
        issues.append(
            CollectionIssue(
                code="migration_enabled",
                message="official forecast collection requires migration_enabled=false",
            )
        )

    # Run identity must agree with the deterministic run_id contract.
    run_id_match = RUN_ID_PATTERN.match(run_id)
    if run_id and run_id_match is None:
        issues.append(
            CollectionIssue(
                code="run_id_format_invalid",
                message=(
                    f"run_id={run_id!r} does not match "
                    "{workload}-rNN-sSEED"
                ),
            )
        )
    elif run_id_match is not None:
        if workload_type and run_id_match.group("workload") != workload_type:
            issues.append(
                CollectionIssue(
                    code="run_id_workload_mismatch",
                    message=(
                        f"run_id workload={run_id_match.group('workload')!r} "
                        f"!= metadata workload_type={workload_type!r}"
                    ),
                )
            )

        for field_name, run_id_part in (
            ("seed", run_id_match.group("seed")),
            ("repeat", run_id_match.group("repeat")),
        ):
            for alias in METADATA_ALIASES.get(field_name, (field_name,)):
                value = metadata.get(alias)
                if value is None:
                    continue
                if str(value).lstrip("0") != run_id_part.lstrip("0"):
                    issues.append(
                        CollectionIssue(
                            code=f"run_id_{field_name}_mismatch",
                            message=(
                                f"metadata.{alias}={value!r} does not match "
                                f"run_id part {run_id_part!r}"
                            ),
                        )
                    )

    # Fields every official run must record (freeze / traceability).
    for field_name in metadata_required_fields:
        aliases = METADATA_ALIASES.get(field_name, (field_name,))
        if all(_is_blank(metadata.get(alias)) for alias in aliases):
            issues.append(
                CollectionIssue(
                    code="metadata_field_missing",
                    message=f"metadata.{field_name} is missing or empty",
                )
            )

    if not samples:
        issues.append(
            CollectionIssue(
                code="samples_empty",
                message="forecast_samples.jsonl contains no samples",
            )
        )

    counts: Counter[str] = Counter()
    capacities: dict[str, list[float]] = defaultdict(list)
    managed_switch_counts: dict[str, set[int]] = defaultdict(set)

    for index, sample in enumerate(samples):
        # A field that is present but null is as unusable as an absent one.
        missing = sorted(
            name for name in REQUIRED_FIELDS if sample.get(name) is None
        )
        if missing:
            issues.append(
                CollectionIssue(
                    code="missing_fields",
                    message=(
                        f"sample {index} missing fields: "
                        + ", ".join(missing)
                    ),
                    controller_id=str(
                        sample.get("controller_id", "")
                    ).strip()
                    or None,
                )
            )

        sample_run_id = str(sample.get("run_id", "")).strip()
        controller_id = str(sample.get("controller_id", "")).strip()
        sample_workload = str(
            sample.get("workload_type", "")
        ).strip()
        workload_phase = str(
            sample.get("workload_phase", "")
        ).strip()

        if sample_run_id != run_id:
            issues.append(
                CollectionIssue(
                    code="run_id_mismatch",
                    message=(
                        f"sample {index} run_id={sample_run_id!r} "
                        f"does not match metadata run_id={run_id!r}"
                    ),
                    controller_id=controller_id or None,
                )
            )

        if controller_id not in expected_controllers:
            issues.append(
                CollectionIssue(
                    code="controller_unexpected",
                    message=(
                        f"sample {index} controller_id={controller_id!r} "
                        "is not listed in metadata.controllers"
                    ),
                    controller_id=controller_id or None,
                )
            )

        if controller_id:
            counts[controller_id] += 1

        if sample_workload != workload_type:
            issues.append(
                CollectionIssue(
                    code="workload_type_mismatch",
                    message=(
                        f"sample {index} workload_type={sample_workload!r} "
                        f"does not match metadata workload_type="
                        f"{workload_type!r}"
                    ),
                    controller_id=controller_id or None,
                )
            )

        if not workload_phase:
            issues.append(
                CollectionIssue(
                    code="workload_phase_missing",
                    message=f"sample {index} workload_phase is empty",
                    controller_id=controller_id or None,
                )
            )

        # Only valid snapshots may feed forecasting; every other row is bad data.
        if sample.get("snapshot_valid") is not True:
            issues.append(
                CollectionIssue(
                    code="snapshot_invalid",
                    message=(
                        f"sample {index} snapshot_valid="
                        f"{sample.get('snapshot_valid')!r} "
                        f"(snapshot_id={sample.get('snapshot_id')!r})"
                    ),
                    controller_id=controller_id or None,
                )
            )

        # Ownership evidence: switches per controller must stay constant
        # while migration is disabled.
        switch_count = sample.get("managed_switch_count")
        if (
            controller_id
            and isinstance(switch_count, (int, float))
            and not isinstance(switch_count, bool)
        ):
            managed_switch_counts[controller_id].add(int(switch_count))

        try:
            safe_capacity = float(sample["safe_capacity_pps"])
            processed_rate = float(sample["processed_packet_in_rate"])
            utilization = float(sample["utilization"])
        except (KeyError, TypeError, ValueError):
            continue

        if controller_id and math.isfinite(safe_capacity):
            capacities[controller_id].append(safe_capacity)

        # U = L / C_safe is undefined for C_safe <= 0; the utilization check
        # below is skipped in that case, so report it explicitly here.
        if not math.isfinite(safe_capacity) or safe_capacity <= 0:
            issues.append(
                CollectionIssue(
                    code="safe_capacity_non_positive",
                    message=(
                        f"sample {index}: safe_capacity_pps="
                        f"{safe_capacity:g} must be finite and > 0"
                    ),
                    controller_id=controller_id or None,
                )
            )

        if (
            math.isfinite(safe_capacity)
            and safe_capacity > 0
            and math.isfinite(processed_rate)
            and math.isfinite(utilization)
        ):
            expected_utilization = processed_rate / safe_capacity
            if not math.isclose(
                utilization,
                expected_utilization,
                rel_tol=utilization_tolerance,
                abs_tol=utilization_tolerance,
            ):
                issues.append(
                    CollectionIssue(
                        code="utilization_mismatch",
                        message=(
                            f"sample {index}: utilization={utilization:g}, "
                            f"processed/safe={expected_utilization:g}"
                        ),
                        controller_id=controller_id or None,
                    )
                )

    for controller_id, values in capacities.items():
        if not values:
            continue

        reference = values[0]
        for value in values[1:]:
            if not math.isclose(
                value,
                reference,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                issues.append(
                    CollectionIssue(
                        code="safe_capacity_inconsistent",
                        message=(
                            f"safe_capacity_pps changes within run for "
                            f"{controller_id}"
                        ),
                        controller_id=controller_id,
                    )
                )
                break

    # The capacity artifact is frozen: samples must use the value that the
    # run metadata recorded for this controller (safe_capacity_c1/c2).
    for controller_id, values in capacities.items():
        recorded = metadata.get(f"safe_capacity_{controller_id}")
        if (
            values
            and isinstance(recorded, (int, float))
            and not isinstance(recorded, bool)
            and not math.isclose(
                values[0],
                float(recorded),
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            issues.append(
                CollectionIssue(
                    code="safe_capacity_metadata_mismatch",
                    message=(
                        f"{controller_id}: samples use {values[0]:g} but "
                        f"metadata.safe_capacity_{controller_id}={recorded:g}"
                    ),
                    controller_id=controller_id,
                )
            )

    if require_fixed_ownership:
        for controller_id, counts_seen in managed_switch_counts.items():
            if len(counts_seen) > 1:
                issues.append(
                    CollectionIssue(
                        code="ownership_changed",
                        message=(
                            f"{controller_id}: managed_switch_count took "
                            f"values {sorted(counts_seen)}; ownership must be "
                            "fixed while migration is disabled"
                        ),
                        controller_id=controller_id,
                    )
                )

    expected_per_controller = 0
    if duration_seconds > 0 and sampling_interval_seconds > 0:
        expected_per_controller = int(
            round(duration_seconds / sampling_interval_seconds)
        )

    coverage: dict[str, float] = {}

    for controller_id in expected_controllers:
        count = counts.get(controller_id, 0)

        if expected_per_controller > 0:
            controller_coverage = count / expected_per_controller
        else:
            controller_coverage = 0.0

        coverage[controller_id] = controller_coverage

        if controller_coverage < min_coverage:
            issues.append(
                CollectionIssue(
                    code="coverage_below_threshold",
                    message=(
                        f"{controller_id}: {count}/"
                        f"{expected_per_controller} samples "
                        f"({controller_coverage:.3f}) < "
                        f"{min_coverage:.3f}"
                    ),
                    controller_id=controller_id,
                )
            )
        elif controller_coverage > 1.0 + (1.0 - min_coverage):
            # Far more samples than duration/interval: duplicated or appended data.
            issues.append(
                CollectionIssue(
                    code="sample_count_excess",
                    message=(
                        f"{controller_id}: {count}/"
                        f"{expected_per_controller} samples "
                        f"({controller_coverage:.3f})"
                    ),
                    controller_id=controller_id,
                )
            )

    # Week 8 official continuity contract.
    #
    # Raw validation intentionally keeps the stricter Week 7 cadence
    # diagnostics. Official collection acceptance instead allows gaps
    # up to max_gap_seconds and handles them here.
    from datetime import datetime

    timestamps_by_controller: dict[str, list[datetime]] = defaultdict(list)

    for sample in samples:
        controller_id = str(
            sample.get("controller_id", "")
        ).strip()
        observed_at = sample.get("observed_at")

        if not controller_id or not isinstance(observed_at, str):
            if controller_id and observed_at is not None:
                issues.append(
                    CollectionIssue(
                        code="observed_at_invalid",
                        message=f"observed_at must be ISO-8601 text: {observed_at!r}",
                        controller_id=controller_id,
                    )
                )
            continue

        try:
            timestamp = datetime.fromisoformat(
                observed_at.replace("Z", "+00:00")
            )
        except ValueError:
            issues.append(
                CollectionIssue(
                    code="observed_at_invalid",
                    message=f"cannot parse observed_at={observed_at!r}",
                    controller_id=controller_id,
                )
            )
            continue

        timestamps_by_controller[controller_id].append(timestamp)

    for controller_id, timestamps in timestamps_by_controller.items():
        for previous, current in zip(
            timestamps,
            timestamps[1:],
        ):
            gap_seconds = (
                current - previous
            ).total_seconds()

            # Per-controller time must strictly increase in file order.
            if gap_seconds == 0:
                issues.append(
                    CollectionIssue(
                        code="timestamp_duplicate",
                        message=f"{controller_id}: duplicate timestamp {current.isoformat()}",
                        controller_id=controller_id,
                    )
                )
            elif gap_seconds < 0:
                issues.append(
                    CollectionIssue(
                        code="timestamp_out_of_order",
                        message=(
                            f"{controller_id}: timestamp goes backwards by "
                            f"{-gap_seconds:.3f}s at {current.isoformat()}"
                        ),
                        controller_id=controller_id,
                    )
                )
            elif gap_seconds > max_gap_seconds:
                issues.append(
                    CollectionIssue(
                        code="max_gap_exceeded",
                        message=(
                            f"{controller_id}: sampling gap "
                            f"{gap_seconds:.3f}s > "
                            f"{max_gap_seconds:.3f}s"
                        ),
                        controller_id=controller_id,
                    )
                )

    raw_result = validate_raw_samples(
        samples,
        expected_sampling_interval_seconds=(
            sampling_interval_seconds
            if sampling_interval_seconds > 0
            else 1.0
        ),
    )

    for issue in raw_result.issues:
        if issue.code in {
            "sampling_gap",
            "sampling_interval",
        }:
            continue
        issues.append(
            CollectionIssue(
                code=f"raw_{issue.code}",
                message=issue.message,
                controller_id=issue.controller_id,
            )
        )

    return CollectionValidationResult(
        valid=not issues,
        run_id=run_id,
        total_samples=len(samples),
        samples_per_controller=dict(counts),
        coverage_per_controller=coverage,
        issues=tuple(issues),
    )


def validate_run_directory(
    run_dir: Path,
    *,
    min_coverage: float = 0.95,
    max_gap_seconds: float = 2.5,
    utilization_tolerance: float = 0.02,
    require_fixed_ownership: bool = True,
    metadata_required_fields: tuple[str, ...] = (),
) -> CollectionValidationResult:
    metadata = _load_json(run_dir / "metadata.json")
    samples = _load_jsonl(run_dir / "forecast_samples.jsonl")

    return validate_collection(
        metadata=metadata,
        samples=samples,
        min_coverage=min_coverage,
        utilization_tolerance=utilization_tolerance,
        max_gap_seconds=max_gap_seconds,
        require_fixed_ownership=require_fixed_ownership,
        metadata_required_fields=metadata_required_fields,
    )


def validate_official_collection(
    output_root: Path,
    plan: Any,
    *,
    expected_samples: int | None = None,
    **run_options: Any,
) -> OfficialValidationResult:
    """Accept the whole official dataset: every planned run must exist and pass.

    ``plan`` is any iterable of objects with a ``run_id`` attribute
    (scripts.experiments.forecast_collection_plan.build_plan).
    """
    issues: list[CollectionIssue] = []
    results: list[CollectionValidationResult] = []
    plan_ids = [run.run_id for run in plan]

    if len(set(plan_ids)) != len(plan_ids):
        issues.append(
            CollectionIssue(
                code="official_plan_duplicate",
                message="plan contains duplicate run_id values",
            )
        )

    for run_id in plan_ids:
        run_dir = output_root / run_id
        if not run_dir.is_dir():
            issues.append(
                CollectionIssue(
                    code="official_run_missing",
                    message=f"planned run directory not found: {run_dir}",
                )
            )
            continue

        try:
            result = validate_run_directory(run_dir, **run_options)
        except (FileNotFoundError, ValueError) as exc:
            issues.append(
                CollectionIssue(
                    code="official_run_unreadable",
                    message=f"{run_id}: {exc}",
                )
            )
            continue

        results.append(result)
        if not result.valid:
            issues.append(
                CollectionIssue(
                    code="official_run_invalid",
                    message=f"{run_id}: {len(result.issues)} issue(s)",
                )
            )
        if result.run_id != run_id:
            issues.append(
                CollectionIssue(
                    code="official_run_id_mismatch",
                    message=f"directory {run_id} holds run_id={result.run_id!r}",
                )
            )

    return OfficialValidationResult(
        valid=not issues,
        expected_runs=len(plan_ids),
        found_runs=len(results),
        expected_samples=expected_samples or 0,
        total_samples=sum(result.total_samples for result in results),
        runs=tuple(results),
        issues=tuple(issues),
    )


def _load_validation_settings(config_path: Path) -> dict[str, Any]:
    """Read thresholds and the metadata contract from the experiment YAML."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    validation = config.get("validation", {})
    return {
        "config": config,
        "min_coverage": validation.get("min_sample_coverage_ratio"),
        "max_gap_seconds": validation.get("max_gap_seconds"),
        "utilization_tolerance": validation.get("utilization_tolerance"),
        "require_fixed_ownership": validation.get("require_fixed_ownership"),
        "metadata_required_fields": tuple(
            config.get("metadata", {}).get("required_fields", ())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate one Week 8 forecast collection run, "
        "or the whole official dataset with --official."
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML; supplies thresholds and the metadata contract.",
    )
    parser.add_argument(
        "--official",
        action="store_true",
        help="Validate every planned run under experiment.output_root.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help="Maximum allowed per-controller sampling gap in seconds.",
    )
    parser.add_argument(
        "--utilization-tolerance",
        type=float,
        default=None,
        help="Allowed |utilization - processed/safe_capacity| (default 0.02).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.official and args.config is None:
        parser.error("--official requires --config")
    if not args.official and args.run_dir is None:
        parser.error("run_dir is required unless --official is used")

    # Precedence: command line > YAML > built-in default.
    settings: dict[str, Any] = {}
    if args.config is not None:
        settings = _load_validation_settings(args.config)

    def pick(cli_value: Any, key: str, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if settings.get(key) is not None:
            return settings[key]
        return default

    options: dict[str, Any] = {
        "min_coverage": pick(args.min_coverage, "min_coverage", 0.95),
        "max_gap_seconds": pick(args.max_gap_seconds, "max_gap_seconds", 2.5),
        "utilization_tolerance": pick(
            args.utilization_tolerance, "utilization_tolerance", 0.02
        ),
        "require_fixed_ownership": pick(None, "require_fixed_ownership", True),
        "metadata_required_fields": settings.get("metadata_required_fields", ()),
    }

    if args.official:
        from scripts.experiments.forecast_collection_plan import build_plan

        config = settings["config"]
        plan = build_plan(config)
        interval = float(config["sampling_interval_seconds"])
        duration = float(config["timing"]["duration_seconds"])
        # nominal = runs x (duration / interval) x controllers  (25 x 180 x 2 = 9000)
        nominal = int(
            len(plan) * round(duration / interval) * len(config["controllers"])
        )
        official = validate_official_collection(
            ROOT / config["experiment"]["output_root"],
            plan,
            expected_samples=nominal,
            **options,
        )
        valid = official.valid
        payload = {
            **asdict(official),
            "runs": [
                {**asdict(run), "issues": [asdict(i) for i in run.issues]}
                for run in official.runs
            ],
            "issues": [asdict(issue) for issue in official.issues],
        }
    else:
        result = validate_run_directory(args.run_dir, **options)
        valid = result.valid
        payload = {
            **asdict(result),
            "issues": [asdict(issue) for issue in result.issues],
        }

    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
