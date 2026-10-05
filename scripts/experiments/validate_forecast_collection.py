#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.forecasting.raw_validator import validate_raw_samples


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

RUN_ID_PATTERN = re.compile(
    r"^(?P<workload>[a-z][a-z0-9]*(?:-[a-z0-9]+)*)"
    r"-r(?P<repeat>\d{2})-s(?P<seed>\d+)$"
)

METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "controller_ids": ("controller_ids", "controllers"),
    "repeat": ("repeat", "repeat_index"),
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


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _capacity_mapping(
    capacity_model: Mapping[str, Any] | str | Path | None,
) -> dict[str, float]:
    if capacity_model is None:
        return {}

    value: Any = capacity_model
    if isinstance(capacity_model, (str, Path)):
        path = Path(capacity_model)
        value = _load_json(path)

    result: dict[str, float] = {}

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        for controller_id, raw in node.items():
            if controller_id not in {"c1", "c2"}:
                if isinstance(raw, Mapping):
                    visit(raw)
                continue

            candidate = raw
            if isinstance(raw, Mapping):
                for key in (
                    "safe_capacity_pps",
                    "capacity_pps",
                    "safe_capacity",
                    "capacity",
                ):
                    if key in raw:
                        candidate = raw[key]
                        break

            try:
                number = float(candidate)
            except (TypeError, ValueError):
                continue

            if math.isfinite(number) and number > 0:
                result[str(controller_id)] = number

    visit(value)

    if not result and isinstance(value, Mapping):
        controllers = value.get("controllers")
        if isinstance(controllers, Mapping):
            visit(controllers)

    return result


def _ownership_signature(sample: Mapping[str, Any]) -> str | None:
    ownership = sample.get("ownership")
    if ownership is None:
        return None
    try:
        return json.dumps(
            ownership,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError:
        return repr(ownership)


def validate_collection(
    *,
    metadata: dict[str, Any],
    samples: list[dict[str, Any]],
    min_coverage: float = 0.95,
    utilization_tolerance: float = 1.0e-6,
    max_gap_seconds: float = 2.5,
    require_fixed_ownership: bool = True,
    metadata_required_fields: tuple[str, ...] = (),
    expected_controller_ids: tuple[str, ...] | None = None,
    expected_workload_types: tuple[str, ...] | None = None,
    expected_migration_enabled: bool = False,
    capacity_model: Mapping[str, Any] | str | Path | None = None,
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
                    "must be present and numeric"
                ),
            )
        )

    if duration_seconds <= 0 or sampling_interval_seconds <= 0:
        issues.append(
            CollectionIssue(
                code="metadata_invalid",
                message=(
                    "duration_seconds and sampling_interval_seconds "
                    "must be positive"
                ),
            )
        )

    configured_controllers = tuple(
        expected_controller_ids
        or tuple(
            str(value).strip()
            for value in metadata.get("controller_ids", metadata.get("controllers", []))
            if str(value).strip()
        )
    )

    if not configured_controllers:
        issues.append(
            CollectionIssue(
                code="controllers_missing",
                message="expected controller IDs are missing",
            )
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

    if expected_workload_types and workload_type not in expected_workload_types:
        issues.append(
            CollectionIssue(
                code="workload_type_unexpected",
                message=(
                    f"metadata.workload_type={workload_type!r} is not an "
                    "official workload"
                ),
            )
        )

    if metadata.get("migration_enabled") is not expected_migration_enabled:
        issues.append(
            CollectionIssue(
                code="migration_enabled",
                message=(
                    "official forecast collection requires "
                    "migration_enabled=false"
                ),
            )
        )

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
        if run_id_match.group("workload") != workload_type:
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
            aliases = METADATA_ALIASES.get(field_name, (field_name,))
            present = False
            for alias in aliases:
                value = metadata.get(alias)
                if value is None:
                    continue
                present = True
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
            if not present:
                issues.append(
                    CollectionIssue(
                        code=f"metadata_{field_name}_missing",
                        message=f"metadata.{field_name} is missing",
                    )
                )

    for field_name in metadata_required_fields:
        aliases = METADATA_ALIASES.get(field_name, (field_name,))
        if all(_is_blank(metadata.get(alias)) for alias in aliases):
            issues.append(
                CollectionIssue(
                    code="metadata_field_missing",
                    message=f"metadata.{field_name} is missing or empty",
                )
            )

    for field_name in (
        "started_at",
        "finished_at",
    ):
        parsed = _timestamp(metadata.get(field_name))
        if parsed is None:
            issues.append(
                CollectionIssue(
                    code="metadata_timestamp_invalid",
                    message=f"metadata.{field_name} must be a valid timestamp",
                )
            )

    started_at = _timestamp(metadata.get("started_at"))
    finished_at = _timestamp(metadata.get("finished_at"))
    if started_at and finished_at and finished_at < started_at:
        issues.append(
            CollectionIssue(
                code="metadata_time_order_invalid",
                message="metadata.finished_at must not precede started_at",
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
    ownership_signatures: dict[str, set[str]] = defaultdict(set)
    timestamps_by_controller: dict[str, list[datetime]] = defaultdict(list)
    capacity_artifact = _capacity_mapping(capacity_model)

    for index, sample in enumerate(samples):
        missing = sorted(
            name for name in REQUIRED_FIELDS
            if name not in sample or sample.get(name) is None
        )
        if missing:
            issues.append(
                CollectionIssue(
                    code="missing_fields",
                    message=(
                        f"sample {index} missing fields: " + ", ".join(missing)
                    ),
                    controller_id=str(
                        sample.get("controller_id", "")
                    ).strip() or None,
                )
            )

        sample_run_id = str(sample.get("run_id", "")).strip()
        controller_id = str(sample.get("controller_id", "")).strip()
        sample_workload = str(sample.get("workload_type", "")).strip()
        workload_phase = str(sample.get("workload_phase", "")).strip()

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

        if controller_id not in configured_controllers:
            issues.append(
                CollectionIssue(
                    code="controller_unexpected",
                    message=(
                        f"sample {index} controller_id={controller_id!r} "
                        "is not an official controller"
                    ),
                    controller_id=controller_id or None,
                )
            )
        else:
            counts[controller_id] += 1

        if sample_workload != workload_type:
            issues.append(
                CollectionIssue(
                    code="workload_type_mismatch",
                    message=(
                        f"sample {index} workload_type={sample_workload!r} "
                        f"does not match metadata workload_type={workload_type!r}"
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

        sample_migration = sample.get("migration_enabled")
        if sample_migration is not None and sample_migration is not expected_migration_enabled:
            issues.append(
                CollectionIssue(
                    code="sample_migration_enabled",
                    message=(
                        f"sample {index}: migration_enabled must be false"
                    ),
                    controller_id=controller_id or None,
                )
            )

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

        observed_at = _timestamp(sample.get("observed_at"))
        if observed_at is None:
            issues.append(
                CollectionIssue(
                    code="observed_at_invalid",
                    message=(
                        f"sample {index} observed_at is not a valid ISO-8601 timestamp"
                    ),
                    controller_id=controller_id or None,
                )
            )
        elif controller_id in configured_controllers:
            timestamps_by_controller[controller_id].append(observed_at)

        switch_count = sample.get("managed_switch_count")
        if (
            controller_id
            and isinstance(switch_count, (int, float))
            and not isinstance(switch_count, bool)
        ):
            managed_switch_counts[controller_id].add(int(switch_count))

        ownership_signature = _ownership_signature(sample)
        if ownership_signature is not None and controller_id:
            ownership_signatures[controller_id].add(ownership_signature)

        try:
            safe_capacity = float(sample["safe_capacity_pps"])
            processed_rate = float(sample["processed_packet_in_rate"])
            utilization = float(sample["utilization"])
        except (KeyError, TypeError, ValueError):
            continue

        if controller_id:
            if math.isfinite(safe_capacity):
                capacities[controller_id].append(safe_capacity)

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
            continue

        if not math.isfinite(processed_rate) or processed_rate < 0:
            issues.append(
                CollectionIssue(
                    code="processed_rate_invalid",
                    message=(
                        f"sample {index}: processed_packet_in_rate must be finite and >= 0"
                    ),
                    controller_id=controller_id or None,
                )
            )
            continue

        if not math.isfinite(utilization) or utilization < 0:
            issues.append(
                CollectionIssue(
                    code="utilization_invalid",
                    message=(
                        f"sample {index}: utilization must be finite and >= 0"
                    ),
                    controller_id=controller_id or None,
                )
            )
            continue

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

        if capacity_artifact:
            expected_capacity = capacity_artifact.get(controller_id)
            if expected_capacity is not None and not math.isclose(
                safe_capacity,
                expected_capacity,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                issues.append(
                    CollectionIssue(
                        code="safe_capacity_artifact_mismatch",
                        message=(
                            f"{controller_id}: sample safe_capacity_pps="
                            f"{safe_capacity:g} != capacity artifact "
                            f"{expected_capacity:g}"
                        ),
                        controller_id=controller_id or None,
                    )
                )

    for controller_id, values in capacities.items():
        if not values:
            continue
        reference = values[0]
        if any(
            not math.isclose(
                value,
                reference,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            for value in values[1:]
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

        recorded = metadata.get(f"safe_capacity_{controller_id}")
        if recorded is not None:
            try:
                recorded_value = float(recorded)
            except (TypeError, ValueError):
                recorded_value = math.nan
            if not math.isfinite(recorded_value) or recorded_value <= 0:
                issues.append(
                    CollectionIssue(
                        code="metadata_safe_capacity_invalid",
                        message=(
                            f"metadata.safe_capacity_{controller_id} "
                            "must be finite and > 0"
                        ),
                        controller_id=controller_id,
                    )
                )
            elif not math.isclose(
                reference,
                recorded_value,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                issues.append(
                    CollectionIssue(
                        code="safe_capacity_metadata_mismatch",
                        message=(
                            f"{controller_id}: samples use {reference:g} but "
                            f"metadata.safe_capacity_{controller_id}={recorded_value:g}"
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
        for controller_id, signatures in ownership_signatures.items():
            if len(signatures) > 1:
                issues.append(
                    CollectionIssue(
                        code="ownership_changed",
                        message=(
                            f"{controller_id}: ownership signature changed "
                            "while migration is disabled"
                        ),
                        controller_id=controller_id,
                    )
                )

    expected_per_controller = (
        int(round(duration_seconds / sampling_interval_seconds))
        if duration_seconds > 0 and sampling_interval_seconds > 0
        else 0
    )

    coverage: dict[str, float] = {}
    for controller_id in configured_controllers:
        count = counts.get(controller_id, 0)
        controller_coverage = (
            count / expected_per_controller
            if expected_per_controller > 0
            else 0.0
        )
        coverage[controller_id] = controller_coverage

        if controller_coverage < min_coverage:
            issues.append(
                CollectionIssue(
                    code="coverage_below_threshold",
                    message=(
                        f"{controller_id}: {count}/{expected_per_controller} "
                        f"samples ({controller_coverage:.3f}) < {min_coverage:.3f}"
                    ),
                    controller_id=controller_id,
                )
            )

        if expected_per_controller > 0 and count > expected_per_controller + 1:
            issues.append(
                CollectionIssue(
                    code="sample_count_excess",
                    message=(
                        f"{controller_id}: {count} samples; expected "
                        f"{expected_per_controller} +/- 1"
                    ),
                    controller_id=controller_id,
                )
            )

    for controller_id, timestamps in timestamps_by_controller.items():
        for previous, current in zip(timestamps, timestamps[1:]):
            gap_seconds = (current - previous).total_seconds()
            if gap_seconds == 0:
                issues.append(
                    CollectionIssue(
                        code="timestamp_duplicate",
                        message=(
                            f"{controller_id}: duplicate timestamp "
                            f"{current.isoformat()}"
                        ),
                        controller_id=controller_id,
                    )
                )
            elif gap_seconds < 0:
                issues.append(
                    CollectionIssue(
                        code="timestamp_out_of_order",
                        message=(
                            f"{controller_id}: timestamp goes backwards by "
                            f"{-gap_seconds:.3f}s"
                        ),
                        controller_id=controller_id,
                    )
                )
            elif gap_seconds > max_gap_seconds:
                issues.append(
                    CollectionIssue(
                        code="max_gap_exceeded",
                        message=(
                            f"{controller_id}: sampling gap {gap_seconds:.3f}s > "
                            f"{max_gap_seconds:.3f}s"
                        ),
                        controller_id=controller_id,
                    )
                )

    # Reuse the frozen raw validator for structural checks, but do not let
    # its stricter ±10% interval policy conflict with this official max-gap
    # contract. Duplicate/order/invalid snapshot/safe-capacity checks remain.
    raw_result = validate_raw_samples(
        samples,
        expected_sampling_interval_seconds=(
            sampling_interval_seconds if sampling_interval_seconds > 0 else 1.0
        ),
    )
    for issue in raw_result.issues:
        if issue.code in {"sampling_gap", "sampling_interval"}:
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
    config: Mapping[str, Any] | None = None,
    min_coverage: float | None = None,
    max_gap_seconds: float | None = None,
    utilization_tolerance: float | None = None,
    require_fixed_ownership: bool | None = None,
    metadata_required_fields: tuple[str, ...] | None = None,
    capacity_model: Mapping[str, Any] | str | Path | None = None,
) -> CollectionValidationResult:
    metadata = _load_json(run_dir / "metadata.json")
    samples = _load_jsonl(run_dir / "forecast_samples.jsonl")

    config_data = config or {}
    validation = config_data.get("validation", {})
    controllers = tuple(config_data.get("controllers", ()))
    workloads = tuple(config_data.get("workloads", ()))

    return validate_collection(
        metadata=metadata,
        samples=samples,
        min_coverage=(
            float(min_coverage)
            if min_coverage is not None
            else float(validation.get("min_sample_coverage_ratio", 0.95))
        ),
        max_gap_seconds=(
            float(max_gap_seconds)
            if max_gap_seconds is not None
            else float(validation.get("max_gap_seconds", 2.5))
        ),
        utilization_tolerance=(
            float(utilization_tolerance)
            if utilization_tolerance is not None
            else float(validation.get("utilization_tolerance", 1.0e-6))
        ),
        require_fixed_ownership=(
            bool(require_fixed_ownership)
            if require_fixed_ownership is not None
            else bool(validation.get("require_fixed_ownership", True))
        ),
        metadata_required_fields=(
            tuple(metadata_required_fields)
            if metadata_required_fields is not None
            else tuple(config_data.get("metadata", {}).get("required_fields", ()))
        ),
        expected_controller_ids=controllers or None,
        expected_workload_types=workloads or None,
        expected_migration_enabled=False,
        capacity_model=capacity_model,
    )


def _load_config(config_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)
    import yaml
    return yaml.safe_load(Path(config_or_path).read_text(encoding="utf-8")) or {}


def _legacy_errors(result: CollectionValidationResult) -> list[str]:
    aliases = {
        "timestamp_duplicate": "duplicate_timestamp",
        "timestamp_out_of_order": "timestamp_not_monotonic",
        "max_gap_exceeded": "sampling_gap",
        "snapshot_invalid": "invalid_snapshot",
        "controller_unexpected": "unexpected_controller_id",
        "safe_capacity_non_positive": "invalid_safe_capacity",
        "utilization_mismatch": "utilization_inconsistent",
        "workload_phase_missing": "missing_workload_phase",
        "migration_enabled": "migration_must_be_disabled_for_official_dataset",
    }
    errors: list[str] = []
    for issue in result.issues:
        prefix = f"{issue.controller_id}:" if issue.controller_id else ""
        alias = aliases.get(issue.code, issue.code)

        if issue.code == "missing_fields":
            missing = issue.message.split(":", 1)[-1].strip()
            for field in [part.strip() for part in missing.split(",") if part.strip()]:
                errors.append(f"{issue.controller_id or 'run'}:missing_field:{field}")
            continue

        errors.append(f"{prefix}{alias}")
        errors.append(f"{prefix}{issue.code}")
    return errors


def validate_forecast_collection(
    run_dir: Path,
    config: Mapping[str, Any] | str | Path,
    capacity_model: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Compatibility API required by the Week 9 validator contract/tests."""
    config_data = _load_config(config)
    result = validate_run_directory(
        run_dir,
        config=config_data,
        capacity_model=capacity_model,
    )
    return {
        "valid": result.valid,
        "run_id": result.run_id,
        "total_samples": result.total_samples,
        "controllers": {
            controller_id: {
                "sample_count": result.samples_per_controller.get(controller_id, 0),
                "coverage_ratio": result.coverage_per_controller.get(controller_id, 0.0),
            }
            for controller_id in config_data.get("controllers", ())
        },
        "errors": _legacy_errors(result),
        "issues": [asdict(issue) for issue in result.issues],
    }


def validate_official_collection(
    output_root: Path,
    plan: Any,
    *,
    expected_samples: int | None = None,
    config: Mapping[str, Any] | None = None,
    capacity_model: Mapping[str, Any] | str | Path | None = None,
    **run_options: Any,
) -> OfficialValidationResult:
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
            result = validate_run_directory(
                run_dir,
                config=config,
                capacity_model=capacity_model,
                **run_options,
            )
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


def _build_fallback_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    workloads = [str(item) for item in config.get("workloads", ())]
    repeats = int(config.get("repeats", 0))
    seeds = [int(item) for item in config.get("seeds", ())]
    if len(seeds) != repeats:
        raise ValueError("seeds length must equal repeats")

    plan: list[dict[str, Any]] = []
    for workload in workloads:
        for repeat in range(1, repeats + 1):
            seed = seeds[repeat - 1]
            run_id = f"{workload}-r{repeat:02d}-s{seed}"
            plan.append(
                {
                    "run_id": run_id,
                    "workload": workload,
                    "repeat_index": repeat,
                    "seed": seed,
                }
            )
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one Week 9 forecast collection run, or the "
            "whole official dataset with --official."
        )
    )
    parser.add_argument("run_dir", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Experiment YAML; supplies thresholds and the metadata contract.",
    )
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--min-coverage", type=float, default=None)
    parser.add_argument("--max-gap-seconds", type=float, default=None)
    parser.add_argument("--utilization-tolerance", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.config is None:
        parser.error("--config is required")

    settings = _load_validation_settings(args.config)
    config = settings["config"]
    options = {
        "min_coverage": settings["min_coverage"],
        "max_gap_seconds": settings["max_gap_seconds"],
        "utilization_tolerance": settings["utilization_tolerance"],
        "require_fixed_ownership": settings["require_fixed_ownership"],
        "metadata_required_fields": settings["metadata_required_fields"],
    }

    if args.min_coverage is not None:
        options["min_coverage"] = args.min_coverage
    if args.max_gap_seconds is not None:
        options["max_gap_seconds"] = args.max_gap_seconds
    if args.utilization_tolerance is not None:
        options["utilization_tolerance"] = args.utilization_tolerance

    if args.official:
        try:
            from scripts.experiments.forecast_collection_plan import build_plan
            plan_objects = build_plan(config)
            plan_ids = [run.run_id for run in plan_objects]
            plan = plan_objects
        except (ImportError, ModuleNotFoundError):
            plan = _build_fallback_plan(config)
            plan_ids = [run["run_id"] for run in plan]

        duration = float(config["duration_seconds"])
        interval = float(config["sampling_interval_seconds"])
        expected_samples = (
            len(plan_ids)
            * round(duration / interval)
            * len(config["controllers"])
        )

        official = validate_official_collection(
            ROOT / config["experiment"]["output_root"],
            plan,
            expected_samples=expected_samples,
            config=config,
            **options,
        )
        payload = {
            **asdict(official),
            "runs": [
                {
                    **asdict(run),
                    "issues": [asdict(issue) for issue in run.issues],
                }
                for run in official.runs
            ],
            "issues": [asdict(issue) for issue in official.issues],
        }
        valid = official.valid
    else:
        if args.run_dir is None:
            parser.error("run_dir is required unless --official is used")
        result = validate_run_directory(
            args.run_dir,
            config=config,
            **options,
        )
        payload = {
            **asdict(result),
            "issues": [asdict(issue) for issue in result.issues],
        }
        valid = result.valid

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
