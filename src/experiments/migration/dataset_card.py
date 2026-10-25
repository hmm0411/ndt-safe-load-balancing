"""Data Card of the migration-outcome dataset (Week 11, Day 7, plan section 37).

Builds ``evidence/week11/dataset_card.json`` from three sources only:

* the matrix config (``configs/experiments/migration_outcome.yaml``): what was planned;
* the evidence files written by ``migration_dataset_report.py``: what was collected
  and measured;
* the card config (``configs/experiments/migration_dataset_card.yaml``): the words
  people decide (purpose, limitations, open decisions, backup checksum).

No number is typed by hand: every count, range and ratio is read from a file, so
the card is regenerated, never edited.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.experiments.migration.config import OutcomeConfig, expand_attempts
from src.experiments.migration.schema import SCHEMA_VERSION
from src.experiments.workloads.hot_switch import HOT_SWITCH_LEVELS

REQUIRED_CARD_KEYS = (
    "dataset_version", "purpose", "topology", "action", "collection_mode", "collected",
    "switch_selection", "freeze", "backup", "known_limitations", "open_decisions",
)
# written after the card (the card itself, Day 7 evidence), so never listed in it
CARD_OUTPUTS = ("dataset_card.json", "ci.txt", "week11_dod.txt")
EVIDENCE_FILES = (
    "dataset_quality_report.json", "split_manifest.json", "migration_latency.json",
    "analytic_check.json", "dataset_statistics.csv", "committed_summary.csv",
    "noop_vs_migrate.csv",
)
RANGE_VARIABLES = (
    "src_u_before_mean", "src_u_after_mean", "dst_u_before_mean", "dst_u_after_mean",
    "switch_share_before_mean", "delta_src_u_mean", "delta_dst_u_mean",
    "transaction_s", "settle_s", "transition_hard_gate_blocked",
)


def load_card_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = [k for k in REQUIRED_CARD_KEYS if k not in data]
    if missing:
        raise ValueError(f"dataset card config: missing {missing}")
    if len(str(data["backup"].get("sha256", ""))) != 64:
        raise ValueError("dataset card config: backup.sha256 must be a 64-character sha256")
    return dict(data)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text)


def _round(value: Any, digits: int = 3) -> Any:
    return round(value, digits) if isinstance(value, float) else value


def measured_vs_planned(committed: list[dict[str, str]], config: OutcomeConfig) -> dict[str, Any]:
    """Median measured hot-switch share and destination U against the planned levels."""
    hot: dict[str, Any] = {}
    for phase, level in HOT_SWITCH_LEVELS.items():
        values = [_num(r["switch_share_before_mean"]) for r in committed
                  if r["workload"] == "hot_switch" and r["phase"] == phase
                  and r["target_hot"] == "true"]
        kept = [v for v in values if v is not None]
        hot[phase] = {"planned_share": level, "n": len(kept),
                      "measured_share_median": _round(statistics.median(kept)) if kept else None}
    destination: dict[str, Any] = {}
    for workload in config.workloads:
        for level in config.destination_load_levels:
            values = [_num(r["dst_u_before_mean"]) for r in committed
                      if r["workload"] == workload and _num(r["destination_load_level"]) == level]
            kept = [v for v in values if v is not None]
            destination[f"{workload}@{level:.2f}"] = {
                "n": len(kept),
                "median": _round(statistics.median(kept)) if kept else None,
                "min": _round(min(kept)) if kept else None,
            }
    return {"hot_switch_share": hot, "destination_u_before": destination}


def build_dataset_card(
    config: OutcomeConfig,
    card: Mapping[str, Any],
    evidence: Path,
    workload_duration_s: Mapping[str, float],
) -> dict[str, Any]:
    missing = [name for name in EVIDENCE_FILES if not (evidence / name).is_file()]
    if missing:
        raise FileNotFoundError(f"evidence files missing in {evidence}: {missing}")
    quality = _read_json(evidence / "dataset_quality_report.json")
    split = _read_json(evidence / "split_manifest.json")
    latency = _read_json(evidence / "migration_latency.json")
    analytic = _read_json(evidence / "analytic_check.json")
    stats = {r["variable"]: r for r in _read_csv(evidence / "dataset_statistics.csv")}
    committed = _read_csv(evidence / "committed_summary.csv")
    noop = [r for r in _read_csv(evidence / "noop_vs_migrate.csv") if r["no_op_n"] not in ("", "0")]

    plans = expand_attempts(config)
    planned_migrate = sum(1 for p in plans if p.action_type == "MIGRATE")
    stationary = analytic["stationary_workloads"]
    hot_ratios = [o / e for e, o in (
        (_num(r["expected_delta_src"]), _num(r["delta_src_u_mean"])) for r in committed
        if r["workload"] == "hot_switch" and r["target_hot"] == "true") if e and o is not None]

    return {
        "dataset_version": card["dataset_version"],
        "schema_version": SCHEMA_VERSION,
        "purpose": card["purpose"],
        "topology": card["topology"],
        "action": card["action"],
        "workloads": list(config.workloads),
        "action_phases": {w: list(p) for w, p in config.workloads.items()},
        "phases_per_workload": sorted({len(p) for p in config.workloads.values()}),
        "destination_load_levels": list(config.destination_load_levels),
        "directions": [f"{s}->{d}" for s, d in config.directions],
        "repeats": dict(config.repeats),
        "planned_attempts": {"migrate": planned_migrate, "no_op": config.no_op_controls,
                             "total": len(plans)},
        "actual_attempts": {key: quality[key] for key in (
            "total_attempts", "missing_attempts", "valid_count", "invalid_count",
            "committed_count", "restored_count", "failed_count", "skipped_count",
            "no_op_count", "training_eligible_committed_count", "training_eligible_no_op_count",
            "valid_pre_count", "valid_post_count", "destination_overload_after_count")},
        "pre_window": f"{config.pre_window_valid_snapshots} valid snapshots",
        "post_window": f"{config.post_window_valid_snapshots} valid snapshots",
        "post_window_timeout_s": config.post_window_timeout_s,
        "quality_gate": "required: every window snapshot passes the Hard Gate "
                        "(valid = fresh and complete and consistent)",
        "failed_transactions": "retained (RESTORED / FAILED / SKIPPED are kept, "
                               "training_eligible = false)",
        "raw_imputation": "none",
        "split": {
            "rule": split["rule"],
            **{name: {"repeats": s["repeats"], "attempts": s["count"], "migrate": s["migrate"],
                      "no_op": s["no_op"]} for name, s in split["splits"].items()},
            "attempts_in_more_than_one_split": len(split["attempts_in_more_than_one_split"]),
        },
        "collection": {
            "mode": card["collection_mode"],
            "collected": card["collected"],
            "workload_duration_s": sorted(set(workload_duration_s.values())),
            "attempt_duration_s_median": _round(latency["attempt_duration_s"]["median"], 1),
            "freeze_tag": card["freeze"]["tag"],
            "git_commit": quality["freeze"]["git_commit"],
            "config_sha256": quality["freeze"]["config_sha256"],
            "capacity_version": quality["freeze"]["capacity_version"],
            "quality_version": quality["freeze"]["quality_version"],
            "freeze_consistent": quality["freeze_consistent"],
            "dirty_tree_check": card["freeze"]["dirty_tree_check"],
        },
        "switch_selection": card["switch_selection"],
        "measured_ranges": {
            name: {k: _round(_num(stats[name][k])) for k in ("min", "median", "max")}
            for name in RANGE_VARIABLES if name in stats
        },
        "measured_vs_planned": measured_vs_planned(committed, config),
        "analytic_check": {
            "definition": analytic["definition"],
            "stable_and_hot_switch": {
                "n": stationary["n"],
                "share_vs_neg_delta_src_pearson_r": _round(
                    stationary["share_vs_neg_delta_src"]["pearson_r"]),
                "observed_over_expected_median": _round(
                    stationary["observed_over_expected"]["median"]),
            },
            "hot_switch_moves": {
                "n": len(hot_ratios),
                "observed_over_expected_median": (
                    _round(statistics.median(hot_ratios)) if hot_ratios else None),
            },
        },
        "noop_controls": [
            {"workload": r["workload"], "phase": r["phase"], "no_op_n": int(r["no_op_n"]),
             "migration_effect_src": _round(_num(r["migration_effect_src"])),
             "migration_effect_dst": _round(_num(r.get("migration_effect_dst")))}
            for r in noop
        ],
        "backup": dict(card["backup"]),
        "known_limitations": list(card["known_limitations"]),
        "notes": list(card.get("notes", [])),
        "open_decisions": list(card["open_decisions"]),
        "evidence": sorted(p.name for p in evidence.iterdir()
                           if p.is_file() and p.name not in CARD_OUTPUTS),
    }
