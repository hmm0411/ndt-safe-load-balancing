"""Runtime what-if predictor for the post-migration outcome (Week 12, Day 2 skeleton).

TwinState (dict from GET /api/v1/twin/state) + MigrationAction
    -> PostMigrationOutcomePrediction

Day 2 builds the fail-closed front of the flow: every case that must stop the
prediction returns NOT_READY with a reason and leaves the numbers as None (a
missing value is never replaced by 0). The numeric path (29 features,
analytical + residual + hybrid, margins) is added in Day 3 once the artifact
and build_features exist. There is no ``safe`` field: safety is Week 13.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.experiments.migration.schema import MigrationAction

READY = "READY"
NOT_READY = "NOT_READY"

MODEL_NAME = "post-migration-outcome"
MODEL_VERSION = "post-migration-v1"
FEATURE_SCHEMA = "outcome-features-v1"
TARGET_SCHEMA = "outcome-targets-v1"
MIN_HISTORY_SNAPSHOTS = 5

INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
INVALID_ACTION = "INVALID_ACTION"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
SWITCH_NOT_OWNED_BY_SOURCE = "SWITCH_NOT_OWNED_BY_SOURCE"
MISSING_SWITCH_TELEMETRY = "MISSING_SWITCH_TELEMETRY"
MISSING_CAPACITY = "MISSING_CAPACITY"
MISSING_FEATURE = "MISSING_FEATURE"
ARTIFACT_MISSING = "ARTIFACT_MISSING"
ARTIFACT_SCHEMA_MISMATCH = "ARTIFACT_SCHEMA_MISMATCH"
CAPACITY_VERSION_MISMATCH = "CAPACITY_VERSION_MISMATCH"
MODEL_NOT_LOADED = "MODEL_NOT_LOADED"

# Files and folders of data/models/outcome/post-migration-v1/ (plan section 46).
ARTIFACT_ENTRIES: tuple[str, ...] = (
    "analytical_config.json",
    "direct_model",
    "hybrid_model",
    "residual_models",
    "feature_schema.json",
    "target_schema.json",
    "hyperparameters.json",
    "validation_metrics.json",
    "test_metrics.json",
    "per_workload_metrics.json",
    "uncertainty_margin.json",
    "inference_latency.json",
    "training_manifest.json",
    "model_card.json",
)


@dataclass(frozen=True)
class PostMigrationOutcomePrediction:
    """Plan section 50. Numbers are None (never 0) when status is NOT_READY."""

    snapshot_id: str
    switch_id: str
    source_controller: str
    target_controller: str
    model_name: str
    model_version: str
    predicted_src_utilization: float | None = None
    predicted_dst_utilization: float | None = None
    predicted_src_response_p95_ms: float | None = None
    predicted_dst_response_p95_ms: float | None = None
    predicted_worst_response_p95_ms: float | None = None
    predicted_imbalance: float | None = None
    dst_utilization_margin: float | None = None
    response_margin_ms: float | None = None
    inference_time_ms: float | None = None
    quality_score: float | None = None
    consistency_score: float | None = None
    status: str = NOT_READY
    reason: str | None = None


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _non_negative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def artifact_problem(
    artifact_dir: Path | str | None, running_capacity_version: str | None
) -> str | None:
    """None when the artifact is usable, otherwise the NOT_READY reason."""
    if artifact_dir is None or not Path(artifact_dir).is_dir():
        return ARTIFACT_MISSING
    root = Path(artifact_dir)
    if any(not (root / entry).exists() for entry in ARTIFACT_ENTRIES):
        return ARTIFACT_SCHEMA_MISMATCH
    try:
        manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ARTIFACT_SCHEMA_MISMATCH
    if (
        not isinstance(manifest, dict)
        or manifest.get("model_version") != MODEL_VERSION
        or manifest.get("feature_schema") != FEATURE_SCHEMA
        or manifest.get("target_schema") != TARGET_SCHEMA
    ):
        return ARTIFACT_SCHEMA_MISMATCH
    recorded = manifest.get("capacity_version")
    if running_capacity_version is None or recorded != running_capacity_version:
        return CAPACITY_VERSION_MISMATCH
    return None


class PostMigrationOutcomePredictor:
    def __init__(
        self,
        artifact_dir: Path | str | None = None,
        running_capacity_version: str | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.running_capacity_version = running_capacity_version

    def predict(
        self,
        twin_state: Mapping[str, Any],
        action: MigrationAction,
        history_size: int = 0,
    ) -> PostMigrationOutcomePrediction:
        """history_size = consecutive valid snapshots kept so far (Day 3 supplies it)."""
        reason = self._first_problem(twin_state, action, history_size)
        # Day 2: the numeric path does not exist yet, so even a clean case stays closed.
        return self._not_ready(twin_state, action, reason or MODEL_NOT_LOADED)

    def _first_problem(
        self, twin_state: Mapping[str, Any], action: MigrationAction, history_size: int
    ) -> str | None:
        quality = twin_state.get("quality") or {}
        if quality.get("valid") is not True:
            return INVALID_SNAPSHOT
        if action.action_type != "MIGRATE" or not action.switch_id:
            return INVALID_ACTION
        if history_size < MIN_HISTORY_SNAPSHOTS:
            return INSUFFICIENT_HISTORY
        switch_id = action.switch_id
        owner = {
            str(item.get("switch_id")): str(item.get("owner_controller_id"))
            for item in twin_state.get("ownership") or []
        }
        if owner.get(switch_id) != action.source_controller:
            return SWITCH_NOT_OWNED_BY_SOURCE
        has_telemetry = any(
            str(row.get("switch_id")) == switch_id
            and str(row.get("controller_id")) == action.source_controller
            and _non_negative(row.get("control_load_share"))
            for row in twin_state.get("switches") or []
        )
        if not has_telemetry:
            return MISSING_SWITCH_TELEMETRY
        capacity_ok = {
            str(row.get("controller_id")): _positive(row.get("safe_capacity_pps"))
            for row in twin_state.get("controllers") or []
        }
        if not (
            capacity_ok.get(action.source_controller) and capacity_ok.get(action.target_controller)
        ):
            return MISSING_CAPACITY
        return artifact_problem(self.artifact_dir, self.running_capacity_version)

    @staticmethod
    def _not_ready(
        twin_state: Mapping[str, Any], action: MigrationAction, reason: str
    ) -> PostMigrationOutcomePrediction:
        quality = twin_state.get("quality") or {}
        return PostMigrationOutcomePrediction(
            snapshot_id=str(twin_state.get("snapshot_id", "")),
            switch_id=action.switch_id or "",
            source_controller=action.source_controller,
            target_controller=action.target_controller,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            quality_score=quality.get("quality_score"),
            consistency_score=quality.get("consistency_score"),
            status=NOT_READY,
            reason=reason,
        )
