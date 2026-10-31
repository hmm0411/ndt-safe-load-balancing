"""Runtime what-if predictor for the post-migration outcome (Week 12, Day 2-3).

TwinState (dict from GET /api/v1/twin/state) + MigrationAction
    -> PostMigrationOutcomePrediction

    observe the snapshot (last 5 consecutive valid snapshots, ``history.py``)
    -> fail-closed checks (reasons below)
    -> build_features (the training function) -> Analytical (load conservation)
    -> residual models -> Hybrid = max(0, Analytical + residual)
    -> r95_post and i_post derived -> uncertainty margins m_U, m_R attached

Every case that must stop the prediction returns NOT_READY with a reason and leaves
the numbers as None (a missing value is never replaced by 0). There is no ``safe``
field: the Safety Gate is Week 13.

Artifact contract read here (data/models/outcome/post-migration-v1/, plan section 46):
``feature_schema.json`` (``model_features``), ``model_card.json`` (``hybrid_family``:
random_forest or xgboost), ``residual_models/`` (files written by ``hybrid.save``),
``uncertainty_margin.json`` (``m_U``, ``m_R``), ``training_manifest.json``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.experiments.migration.schema import MigrationAction
from src.twin.outcome import analytical, hybrid, residual, trainer
from src.twin.outcome.features import (
    CANDIDATE_FEATURES,
    ActionSpec,
    build_features,
    is_forbidden,
)
from src.twin.outcome.history import OutcomeHistory
from src.twin.outcome.targets import BASE_TARGETS

READY = "READY"
NOT_READY = "NOT_READY"

MODEL_NAME = "post-migration-outcome"
MODEL_VERSION = "post-migration-v1"
FEATURE_SCHEMA = "outcome-features-v1"
TARGET_SCHEMA = "outcome-targets-v1"

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


@dataclass(frozen=True)
class _Runtime:
    models: dict[str, Any]
    hybrid_features: tuple[str, ...]
    m_u: float
    m_r: float


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def _margin(data: Mapping[str, Any], key: str) -> float:
    value = data[key]
    if not _non_negative(value):
        raise ValueError(f"uncertainty_margin.{key} must be a finite number >= 0, got {value!r}")
    return float(value)


def _load_runtime(root: Path) -> _Runtime:
    """Models and margins of a usable artifact; ValueError/OSError/KeyError when broken."""
    schema = _read_json(root / "feature_schema.json")
    if schema.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("feature_schema.json has another feature schema")
    model_features = [str(name) for name in schema["model_features"]]
    bad = [n for n in model_features if n not in CANDIDATE_FEATURES or is_forbidden(n)]
    if bad:
        raise ValueError(f"model features that build_features does not produce: {bad}")
    family = _read_json(root / "model_card.json").get("hybrid_family")
    if family not in trainer.FAMILIES:
        raise ValueError(f"model_card.hybrid_family must be one of {list(trainer.FAMILIES)}")
    margin = _read_json(root / "uncertainty_margin.json")
    return _Runtime(
        models=hybrid.load(root / "residual_models", str(family)),
        hybrid_features=tuple(residual.hybrid_features(model_features)),
        m_u=_margin(margin, "m_U"),
        m_r=_margin(margin, "m_R"),
    )


class PostMigrationOutcomePredictor:
    def __init__(
        self,
        artifact_dir: Path | str | None = None,
        running_capacity_version: str | None = None,
        history: OutcomeHistory | None = None,
    ) -> None:
        self.artifact_dir = artifact_dir
        self.running_capacity_version = running_capacity_version
        self.history = history if history is not None else OutcomeHistory()
        self.last_error: str | None = None  # detail of the last MISSING_FEATURE, for logs
        self._runtime: _Runtime | None = None

    def observe(self, twin_state: Mapping[str, Any]) -> None:
        """Feed one TwinState to the history (call once per snapshot; duplicates are ignored)."""
        try:
            self.history.observe(twin_state)
        except ValueError:
            pass  # malformed snapshot: not stored; predict() reports INVALID_SNAPSHOT

    def predict(
        self, twin_state: Mapping[str, Any], action: MigrationAction
    ) -> PostMigrationOutcomePrediction:
        """What-if outcome of ``action`` on the state. Only the in-memory history changes."""
        try:
            self.history.observe(twin_state)
        except ValueError:
            return self._not_ready(twin_state, action, INVALID_SNAPSHOT)
        reason = self._first_problem(twin_state, action)
        if reason is not None:
            return self._not_ready(twin_state, action, reason)
        runtime, reason = self._runtime_or_reason()
        if runtime is None:
            return self._not_ready(twin_state, action, reason or ARTIFACT_MISSING)
        assert action.switch_id is not None  # _first_problem rejected the other case
        return self._predict(twin_state, action, action.switch_id, runtime)

    def _first_problem(self, twin_state: Mapping[str, Any], action: MigrationAction) -> str | None:
        quality = twin_state.get("quality") or {}
        if quality.get("valid") is not True:
            return INVALID_SNAPSHOT
        if action.action_type != "MIGRATE" or not action.switch_id:
            return INVALID_ACTION
        if not self.history.ready():
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
        return None

    def _runtime_or_reason(self) -> tuple[_Runtime | None, str | None]:
        if self._runtime is not None:
            return self._runtime, None
        reason = artifact_problem(self.artifact_dir, self.running_capacity_version)
        if reason is not None:
            return None, reason
        try:
            self._runtime = _load_runtime(Path(str(self.artifact_dir)))
        except (OSError, ValueError, KeyError, TypeError):
            return None, ARTIFACT_SCHEMA_MISMATCH
        return self._runtime, None

    def _predict(
        self,
        twin_state: Mapping[str, Any],
        action: MigrationAction,
        switch_id: str,
        runtime: _Runtime,
    ) -> PostMigrationOutcomePrediction:
        source, target = action.source_controller, action.target_controller
        start = time.perf_counter_ns()
        try:
            window = self.history.window()
            spec = ActionSpec(switch_id, source, target)
            features = build_features(window, spec)
            base = analytical.predict(analytical.inputs_from_window(window, spec, source, target))
            row: dict[str, float] = dict(features)
            for name in BASE_TARGETS:
                row[f"{residual.ANALYTICAL_PREFIX}{name}"] = float(getattr(base, name))
            out = hybrid.predict(runtime.models, [row], list(runtime.hybrid_features))
            values = {name: float(series[0]) for name, series in out.items()}
        except (ValueError, KeyError) as exc:
            self.last_error = str(exc)
            return self._not_ready(twin_state, action, MISSING_FEATURE)
        if not all(math.isfinite(v) for v in values.values()):
            self.last_error = "non-finite prediction"
            return self._not_ready(twin_state, action, MISSING_FEATURE)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        quality = twin_state.get("quality") or {}
        return PostMigrationOutcomePrediction(
            snapshot_id=str(twin_state.get("snapshot_id", "")),
            switch_id=switch_id,
            source_controller=source,
            target_controller=target,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            predicted_src_utilization=values["u_src_post"],
            predicted_dst_utilization=values["u_dst_post"],
            predicted_src_response_p95_ms=values["r_src95_post"],
            predicted_dst_response_p95_ms=values["r_dst95_post"],
            predicted_worst_response_p95_ms=values["r95_post"],
            predicted_imbalance=values["i_post"],
            dst_utilization_margin=runtime.m_u,
            response_margin_ms=runtime.m_r,
            inference_time_ms=elapsed_ms,
            quality_score=quality.get("quality_score"),
            consistency_score=quality.get("consistency_score"),
            status=READY,
            reason=None,
        )

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
