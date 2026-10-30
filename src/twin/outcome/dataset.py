"""Processed dataset of the Post-Migration Outcome Model (Week 12, Day 1).

One COMMITTED, training-eligible MIGRATE attempt = one row (plan section 4): the
five post snapshots of an attempt are never separate rows. NO_OP attempts go to a
separate reference table (no-action drift), never into the regression. Every other
attempt is listed with its exclusion reason; nothing is imputed.

Raw data is read as written by the Week 11 runner (``src.experiments.migration``),
the split is taken unchanged from the collection's split manifest, saturation flags
come from ``src.experiments.migration.saturation`` (CPU p95 >= 95 %).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.experiments.migration.saturation import attempt_saturation
from src.experiments.migration.schema import MigrationAction, StateSnapshotView
from src.twin.outcome import analytical
from src.twin.outcome.features import (
    CANDIDATE_FEATURES,
    ActionSpec,
    build_features,
    feature_schema,
    select_model_features,
)
from src.twin.outcome.targets import (
    ALL_TARGETS,
    OutcomeValues,
    target_schema,
    targets_from_window,
)

ID_COLUMNS = (
    "attempt_id", "workload", "phase", "destination_load_level", "direction", "repeat", "split",
    "switch_id", "source_controller", "target_controller", "target_hot",
)
FLAG_COLUMNS = ("saturated_before", "saturated_after", "overload_after", "switch_share_group")
ANALYTICAL_COLUMNS = tuple(f"analytical_{name}" for name in ALL_TARGETS)
DATASET_COLUMNS = ID_COLUMNS + FLAG_COLUMNS + CANDIDATE_FEATURES + ANALYTICAL_COLUMNS + ALL_TARGETS
NOOP_COLUMNS = (
    "attempt_id", "workload", "phase", "destination_load_level", "direction", "repeat", "split",
    "source_controller", "target_controller", "saturated_before", "saturated_after",
    "src_u_pre_mean", "dst_u_pre_mean",
) + ANALYTICAL_COLUMNS + ALL_TARGETS
BOOL_COLUMNS = ("target_hot", "saturated_before", "saturated_after", "overload_after")
TEXT_COLUMNS = ("attempt_id", "workload", "phase", "direction", "repeat", "split", "switch_id",
                "source_controller", "target_controller", "switch_share_group")
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class OutcomeDatasetConfig:
    dataset_version: str
    source_dataset: str
    required_window: int
    overload_threshold: float
    capacity_tolerance_pps: float
    switch_share_bins: tuple[float, float]
    selection_weights: dict[str, float]
    selection_epsilon: float
    selection_exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.required_window < 2:
            raise ValueError("required_window must be >= 2")
        low, high = self.switch_share_bins
        if not 0 < low < high < 1:
            raise ValueError("switch_share_bins must be 0 < low < high < 1")
        if abs(sum(self.selection_weights.values()) - 1.0) > 1e-9:
            raise ValueError("selection_weights must sum to 1")
        if self.capacity_tolerance_pps < 0 or self.selection_epsilon <= 0:
            raise ValueError("capacity_tolerance_pps >= 0 and selection_epsilon > 0 required")
        unknown = [t for t in self.selection_exclude if t not in self.selection_weights]
        if unknown:
            raise ValueError(f"selection_exclude names unknown targets: {unknown}")
        if set(self.selection_exclude) >= set(self.selection_weights):
            raise ValueError("selection_exclude cannot remove every target")

    @property
    def effective_selection_weights(self) -> dict[str, float]:
        """Weights used to choose models: plan weights without the excluded targets,
        renormalized to sum to 1."""
        kept = {t: w for t, w in self.selection_weights.items() if t not in self.selection_exclude}
        total = sum(kept.values())
        return {t: w / total for t, w in kept.items()}


def load_dataset_config(path: Path) -> OutcomeDatasetConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        bins = [float(x) for x in data["switch_share_bins"]]
        if len(bins) != 2:
            raise ValueError("switch_share_bins needs two values")
        return OutcomeDatasetConfig(
            dataset_version=str(data["dataset_version"]),
            source_dataset=str(data["source_dataset"]),
            required_window=int(data["required_window"]),
            overload_threshold=float(data["overload_threshold"]),
            capacity_tolerance_pps=float(data["capacity_tolerance_pps"]),
            switch_share_bins=(bins[0], bins[1]),
            selection_weights={str(k): float(v) for k, v in dict(data["selection_weights"]).items()},
            selection_epsilon=float(data["selection_epsilon"]),
            selection_exclude=tuple(str(t) for t in data.get("selection_exclude") or ()),
        )
    except KeyError as exc:
        raise ValueError(f"{path}: missing key {exc}") from exc


# ------------------------------------------------------------------ inputs
def capacity_info(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    capacities = {str(cid): float(item["c_safe_pps"]) for cid, item in data["controllers"].items()}
    digest = hashlib.sha256(raw).hexdigest()
    return {"path": str(path), "sha256": digest,
            "capacity_version": f"{path.name}@sha256:{digest[:12]}", "c_safe_pps": capacities}


def load_split_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for name, entry in data["splits"].items():
        for attempt_id in entry["attempts"]:
            if attempt_id in mapping:
                raise ValueError(f"{attempt_id} is in two splits")
            mapping[str(attempt_id)] = str(name)
    return mapping


def saturation_flags(raw_root: Path) -> dict[str, tuple[bool, bool]]:
    """(saturated_before, saturated_after) per attempt directory, CPU p95 >= 95 %."""
    return {str(r["run_id"]): (bool(r["sat_before"]), bool(r["sat_after"]))
            for r in attempt_saturation(raw_root)}


def load_saturation_csv(path: Path) -> dict[str, tuple[bool, bool]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["run_id"]: (row["sat_before"] == "True", row["sat_after"] == "True")
                for row in csv.DictReader(handle)}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_window(path: Path) -> list[StateSnapshotView]:
    views = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            views.append(StateSnapshotView.from_twin_state(json.loads(line)))
    return views


def attempt_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


# ------------------------------------------------------------------ rows
def share_group(share: float, bins: tuple[float, float]) -> str:
    if share < bins[0]:
        return "low"
    return "medium" if share < bins[1] else "high"


def _check_capacity(snap: StateSnapshotView, controller: str, capacity: Mapping[str, float],
                    tolerance: float, where: str) -> None:
    raw = snap.controller(controller).safe_capacity_pps
    if controller not in capacity:
        raise ValueError(f"{where}: {controller} is not in the capacity file")
    if abs(raw - capacity[controller]) > tolerance:
        raise ValueError(f"{where}: raw safe_capacity_pps {raw} of {controller} differs from the "
                         f"capacity file ({capacity[controller]}); wrong capacity version?")


def _analytical_columns(values: OutcomeValues) -> dict[str, float]:
    return values.to_dict(prefix="analytical_")


def build_rows(
    raw_root: Path,
    split_map: Mapping[str, str],
    capacity: Mapping[str, float],
    config: OutcomeDatasetConfig,
    saturation: Mapping[str, tuple[bool, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """(regression rows, NO_OP reference rows, excluded attempt -> reason)."""
    rows: list[dict[str, Any]] = []
    noop_rows: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    window = config.required_window
    for attempt_dir in attempt_dirs(raw_root):
        name = attempt_dir.name
        metadata = _read_json(attempt_dir / "metadata.json")
        action = MigrationAction.from_dict(_read_json(attempt_dir / "action.json"))
        outcome = _read_json(attempt_dir / "outcome.json")["outcome"]
        validation = _read_json(attempt_dir / "validation.json")
        attempt_id = str(metadata["attempt_id"])
        if not validation.get("valid", False):
            excluded[attempt_id] = "invalid"
            continue
        if not outcome.get("training_eligible", False):
            excluded[attempt_id] = "not_training_eligible:" + ";".join(
                outcome.get("ineligible_reasons", []))
            continue
        if attempt_id not in split_map:
            raise ValueError(f"{attempt_id} is in no split of the split manifest")
        if name not in saturation:
            raise ValueError(f"{name}: no saturation flags")
        src, dst = action.source_controller, action.target_controller
        before = _read_window(attempt_dir / "before_samples.jsonl")
        after = _read_window(attempt_dir / "after_samples.jsonl")
        for snap in before[-1:] + after[-1:]:
            _check_capacity(snap, src, capacity, config.capacity_tolerance_pps, attempt_id)
            _check_capacity(snap, dst, capacity, config.capacity_tolerance_pps, attempt_id)
        sat_before, sat_after = saturation[name]
        base = {
            "attempt_id": attempt_id,
            "workload": str(metadata["workload"]),
            "phase": str(metadata["phase"]),
            "destination_load_level": float(metadata["destination_load_level"]),
            "direction": f"{src}->{dst}",
            "repeat": str(metadata["repeat"]),
            "split": split_map[attempt_id],
            "source_controller": src,
            "target_controller": dst,
            "saturated_before": sat_before,
            "saturated_after": sat_after,
        }
        observed = targets_from_window(after, src, dst).to_dict()
        if action.action_type == "NO_OP":
            inputs = analytical.inputs_from_window(before, None, src, dst, window)
            row = dict(base)
            row["src_u_pre_mean"] = sum(s.controller(src).utilization for s in before) / len(before)
            row["dst_u_pre_mean"] = sum(s.controller(dst).utilization for s in before) / len(before)
            row.update(_analytical_columns(analytical.predict(inputs)))
            row.update(observed)
            noop_rows.append(row)
            continue
        if outcome.get("transaction_status") != "COMMITTED":
            excluded[attempt_id] = f"transaction_{outcome.get('transaction_status')}"
            continue
        if action.switch_id is None:  # MigrationAction already refuses this for MIGRATE
            raise ValueError(f"{attempt_id}: MIGRATE without switch_id")
        spec = ActionSpec(action.switch_id, src, dst)
        try:
            features = build_features(before, spec, window)
            inputs = analytical.inputs_from_window(before, spec, src, dst, window)
        except ValueError as exc:
            excluded[attempt_id] = f"features: {exc}"
            continue
        row = dict(base)
        row["switch_id"] = action.switch_id
        row["target_hot"] = bool(action.target_hot)
        row["overload_after"] = observed["u_dst_post"] >= config.overload_threshold
        row["switch_share_group"] = share_group(features["switch_share_mean"],
                                                config.switch_share_bins)
        row.update(features)
        row.update(_analytical_columns(analytical.predict(inputs)))
        row.update(observed)
        rows.append(row)
    return rows, noop_rows, excluded


# ------------------------------------------------------------------ reports
def _count(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


def quality_report(rows: Sequence[Mapping[str, Any]], noop_rows: Sequence[Mapping[str, Any]],
                   excluded: Mapping[str, str], model_features: Sequence[str],
                   audit_only: Mapping[str, str]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for reason in excluded.values():
        key = reason.split(":", 1)[0]
        reasons[key] = reasons.get(key, 0) + 1
    by_split = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    return {
        "attempts_scanned": len(rows) + len(noop_rows) + len(excluded),
        "rows": len(rows),
        "noop_reference_rows": len(noop_rows),
        "excluded": len(excluded),
        "excluded_by_reason": reasons,
        "excluded_attempts": dict(excluded),
        "rows_by_split": {s: len(v) for s, v in by_split.items()},
        "noop_by_split": _count(noop_rows, "split"),
        "rows_by_workload": _count(rows, "workload"),
        "rows_by_split_and_workload": {s: _count(v, "workload") for s, v in by_split.items()},
        "saturated_before": sum(1 for r in rows if r["saturated_before"]),
        "saturated_after": sum(1 for r in rows if r["saturated_after"]),
        "overload_after": sum(1 for r in rows if r["overload_after"]),
        "overload_after_by_split": {s: sum(1 for r in v if r["overload_after"])
                                    for s, v in by_split.items()},
        "switch_share_groups": _count(rows, "switch_share_group"),
        "candidate_features": len(CANDIDATE_FEATURES),
        "model_features": len(model_features),
        "audit_only": dict(audit_only),
        "one_row_per_attempt": len({r["attempt_id"] for r in rows}) == len(rows),
    }


def _cell(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value in the dataset: {value!r}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c, "")) for c in columns})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_dataset(
    out: Path,
    rows: Sequence[Mapping[str, Any]],
    noop_rows: Sequence[Mapping[str, Any]],
    excluded: Mapping[str, str],
    config: OutcomeDatasetConfig,
    provenance: Mapping[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """Write the processed dataset; refuse to overwrite a built one unless ``force``."""
    if (out / "dataset_manifest.json").exists() and not force:
        raise FileExistsError(f"{out} already holds a dataset; bump the version instead")
    if not rows:
        raise ValueError("no eligible rows")
    out.mkdir(parents=True, exist_ok=True)
    train = [dict(r) for r in rows if r["split"] == "train"]
    model_features, audit_only = select_model_features(train)
    if not model_features:
        raise ValueError("every candidate feature is audit_only on train; check the raw data")
    ordered = sorted(rows, key=lambda r: (SPLITS.index(r["split"]), r["attempt_id"]))
    dataset_sha = _write_csv(out / "outcome_dataset.csv", ordered, DATASET_COLUMNS)
    noop_sha = _write_csv(out / "noop_reference.csv",
                          sorted(noop_rows, key=lambda r: r["attempt_id"]), NOOP_COLUMNS)
    _write_json(out / "feature_schema.json", feature_schema(model_features, audit_only))
    _write_json(out / "target_schema.json", target_schema())
    _write_json(out / "split_manifest.json", {
        "rule": "unchanged from the raw collection's split manifest (by repeat, never by row)",
        "splits": {s: sorted(r["attempt_id"] for r in rows if r["split"] == s) for s in SPLITS},
    })
    report = quality_report(rows, noop_rows, excluded, model_features, audit_only)
    _write_json(out / "dataset_quality_report.json", report)
    manifest = {
        "dataset_version": config.dataset_version,
        "source_dataset": config.source_dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "required_window": config.required_window,
        "feature_schema": feature_schema(model_features, audit_only)["feature_schema"],
        "target_schema": target_schema()["target_schema"],
        "rows": len(rows),
        "rows_by_split": report["rows_by_split"],
        "noop_reference_rows": len(noop_rows),
        "files": {"outcome_dataset.csv": dataset_sha, "noop_reference.csv": noop_sha},
        **dict(provenance),
    }
    _write_json(out / "dataset_manifest.json", manifest)
    return {"manifest": manifest, "report": report}


# ------------------------------------------------------------------ reading back
def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Rows of outcome_dataset.csv / noop_reference.csv with numbers and booleans parsed."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if key in TEXT_COLUMNS:
                parsed[key] = value
            elif key in BOOL_COLUMNS:
                parsed[key] = value == "True"
            else:
                parsed[key] = float(value)
        out.append(parsed)
    return out
