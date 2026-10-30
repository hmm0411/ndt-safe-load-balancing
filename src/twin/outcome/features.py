"""Features of the Post-Migration Outcome Model (``outcome-features-v1``).

``build_features`` is the ONLY place features are computed: the offline dataset
builder and the runtime what-if predictor both call it, so training and serving
cannot drift apart. It only reads the pre window (the snapshots before the
action) and the action itself; anything observed after the action is leakage.

"Current" values come from the last snapshot of the pre window, "temporal" values
from the whole window (slope = least-squares slope per second). Fail-closed: a
missing value raises ``ValueError``; nothing is filled with 0.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from src.experiments.migration.metrics import imbalance, max_utilization
from src.experiments.migration.schema import StateSnapshotView

FEATURE_SCHEMA_VERSION = "outcome-features-v1"

CONTROLLER_FIELDS = (
    "utilization", "packet_in_rate", "flow_mod_rate", "cpu_percent", "memory_rss_mb",
    "response_p95_ms", "switch_count", "safe_capacity_pps",
)
SOURCE_FEATURES = tuple(f"src_{name}" for name in CONTROLLER_FIELDS)
DESTINATION_FEATURES = tuple(f"dst_{name}" for name in CONTROLLER_FIELDS)
ACTION_FEATURES = ("switch_control_load_share", "switch_packet_in_rate", "capacity_ratio_dst_src")
TEMPORAL_FEATURES = (
    "src_u_mean", "src_u_std", "src_u_slope", "dst_u_mean", "dst_u_std", "dst_u_slope",
    "switch_share_mean", "switch_share_slope",
)
SYSTEM_FEATURES = ("imbalance_before", "max_u_before")
CANDIDATE_FEATURES = (
    SOURCE_FEATURES + DESTINATION_FEATURES + ACTION_FEATURES + TEMPORAL_FEATURES + SYSTEM_FEATURES
)
FEATURE_GROUPS = {
    "source": SOURCE_FEATURES,
    "destination": DESTINATION_FEATURES,
    "action": ACTION_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "system": SYSTEM_FEATURES,
}
# Never model inputs, whatever their variance: the two safe capacities differ by a
# fraction of a pps between c1 and c2, so they (and their ratio) encode the controller id.
AUDIT_BY_DESIGN = ("src_safe_capacity_pps", "dst_safe_capacity_pps", "capacity_ratio_dst_src")
# Columns that must never be model inputs (post-action information, ids, labels, quality).
FORBIDDEN_FEATURES = (
    "attempt_id", "switch_id", "source_controller", "target_controller", "transaction_id",
    "workload", "phase", "destination_load_level", "direction", "repeat", "split", "target_hot",
    "transaction_status", "transaction_s", "settle_s", "transition_hard_gate_blocked",
    "saturated_before", "saturated_after", "overload_after", "rollback",
    "quality_score", "consistency_score",
)
FORBIDDEN_PREFIXES = ("u_src_post", "u_dst_post", "r_src95_post", "r_dst95_post", "r95_post",
                      "i_post", "after_", "post_", "delta_")


@dataclass(frozen=True)
class ActionSpec:
    """The hypothetical action A = (switch, source, target)."""

    switch_id: str
    source_controller: str
    target_controller: str

    def __post_init__(self) -> None:
        if not self.switch_id:
            raise ValueError("switch_id is required")
        if not self.source_controller or not self.target_controller:
            raise ValueError("source_controller and target_controller are required")
        if self.source_controller == self.target_controller:
            raise ValueError("source_controller and target_controller must differ")


def _timestamp(value: str) -> float:
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).timestamp()


def slope(times: Sequence[float], values: Sequence[float]) -> float:
    """Least-squares slope of values over times (units per second)."""
    if len(times) != len(values) or len(values) < 2:
        raise ValueError("slope needs at least two (time, value) pairs")
    t0 = times[0]
    xs = [t - t0 for t in times]
    mx = sum(xs) / len(xs)
    my = sum(values) / len(values)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("slope needs distinct timestamps")
    return sum((x - mx) * (y - my) for x, y in zip(xs, values)) / sxx


def _controller_values(snap: StateSnapshotView, controller_id: str, prefix: str) -> dict[str, float]:
    c = snap.controller(controller_id)
    if c.process_memory_rss_mb is None:
        raise ValueError(f"snapshot {snap.snapshot_id}: {controller_id} has no process_memory_rss_mb")
    values = {
        "utilization": c.utilization,
        "packet_in_rate": c.processed_packet_in_rate,
        "flow_mod_rate": c.flow_mod_rate,
        "cpu_percent": c.process_cpu_percent,
        "memory_rss_mb": c.process_memory_rss_mb,
        "response_p95_ms": c.response_p95_ms,
        "switch_count": float(c.managed_switch_count),
        "safe_capacity_pps": c.safe_capacity_pps,
    }
    return {f"{prefix}{name}": float(values[name]) for name in CONTROLLER_FIELDS}


def check_window(
    pre_window: Sequence[StateSnapshotView], action: ActionSpec, required_window: int = 5
) -> None:
    """Raise ValueError unless the pre window can be used for this action."""
    if len(pre_window) < required_window:
        raise ValueError(f"pre window has {len(pre_window)} snapshots, needs {required_window}")
    for snap in pre_window:
        if not snap.quality.valid:
            raise ValueError(f"snapshot {snap.snapshot_id} failed the Hard Gate")
    last = pre_window[-1]
    owner = last.owner_of(action.switch_id)
    if owner != action.source_controller:
        raise ValueError(f"{action.switch_id} is owned by {owner}, not {action.source_controller}")
    last.controller(action.target_controller)


def build_features(
    pre_window: Sequence[StateSnapshotView], action: ActionSpec, required_window: int = 5
) -> dict[str, float]:
    """The CANDIDATE_FEATURES of one (state, action), in that order."""
    check_window(pre_window, action, required_window)
    src, dst = action.source_controller, action.target_controller
    last = pre_window[-1]
    times = [_timestamp(s.created_at) for s in pre_window]
    src_u = [s.controller(src).utilization for s in pre_window]
    dst_u = [s.controller(dst).utilization for s in pre_window]

    share_t, shares, rates = [], [], []
    for t, snap in zip(times, pre_window):
        entry = snap.switch_load(action.switch_id, src)
        if entry is not None:
            share_t.append(t)
            shares.append(entry.control_load_share)
            rates.append(entry.processed_packet_in_rate)
    if len(shares) < 2:
        raise ValueError(f"{action.switch_id}: switch load reported in {len(shares)} pre snapshots")
    last_entry = last.switch_load(action.switch_id, src)
    if last_entry is None:
        raise ValueError(f"{action.switch_id}: no switch load in the last pre snapshot")

    features: dict[str, float] = {}
    features.update(_controller_values(last, src, "src_"))
    features.update(_controller_values(last, dst, "dst_"))
    features["switch_control_load_share"] = last_entry.control_load_share
    features["switch_packet_in_rate"] = sum(rates) / len(rates)
    features["capacity_ratio_dst_src"] = features["dst_safe_capacity_pps"] / features[
        "src_safe_capacity_pps"]
    features["src_u_mean"] = statistics.fmean(src_u)
    features["src_u_std"] = statistics.pstdev(src_u)
    features["src_u_slope"] = slope(times, src_u)
    features["dst_u_mean"] = statistics.fmean(dst_u)
    features["dst_u_std"] = statistics.pstdev(dst_u)
    features["dst_u_slope"] = slope(times, dst_u)
    features["switch_share_mean"] = statistics.fmean(shares)
    features["switch_share_slope"] = slope(share_t, shares)
    features["imbalance_before"] = imbalance(last.utilization())
    features["max_u_before"] = max_utilization(last.utilization())
    for name, value in features.items():
        if not math.isfinite(value):
            raise ValueError(f"feature {name} is not finite: {value!r}")
    return {name: features[name] for name in CANDIDATE_FEATURES}


def is_forbidden(name: str) -> bool:
    return name in FORBIDDEN_FEATURES or name.startswith(FORBIDDEN_PREFIXES)


def select_model_features(
    train_rows: Sequence[dict[str, Any]], tolerance: float = 1e-12
) -> tuple[list[str], dict[str, str]]:
    """(model features, audit_only name -> reason) from the TRAIN rows only."""
    if not train_rows:
        raise ValueError("no train rows to select features from")
    audit: dict[str, str] = {name: "by design: encodes the controller id"
                             for name in AUDIT_BY_DESIGN}
    for name in CANDIDATE_FEATURES:
        if name in audit:
            continue
        values = [float(row[name]) for row in train_rows]
        if max(values) - min(values) <= tolerance:
            audit[name] = "constant on train"
    model = [name for name in CANDIDATE_FEATURES if name not in audit]
    leaked = [name for name in model if is_forbidden(name)]
    if leaked:
        raise ValueError(f"forbidden columns among model features: {leaked}")
    return model, audit


def feature_schema(model_features: Sequence[str], audit_only: dict[str, str]) -> dict[str, Any]:
    return {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "candidate_features": list(CANDIDATE_FEATURES),
        "groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "model_features": list(model_features),
        "audit_only": dict(audit_only),
        "forbidden": list(FORBIDDEN_FEATURES),
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "current": "last snapshot of the pre window",
        "temporal": "whole pre window; slope = least-squares slope per second",
        "switch_packet_in_rate": "mean processed rate of the switch, reported by the source",
        "hybrid_extra_features": ["analytical_u_src_post", "analytical_u_dst_post",
                                  "analytical_r_src95_post", "analytical_r_dst95_post"],
    }
