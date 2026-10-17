"""Runtime forecaster: TwinState history -> U_hat(t+h) per controller (V2).

Reads ONE artifact directory of contract C1 (docs/forecast_artifact_contract.md)
and turns each TwinState snapshot into one ForecastPrediction per controller:

    TwinState -> forecast_rows (adds the load share the dataset recorded)
              -> TwinHistoryBuffer (11 consecutive valid samples)
              -> build_features (the training function) -> 27-feature vector
              -> OOD gate (model features + monitor-only columns)
              -> model (or persistence while the gate is on)
              -> U_hat + m_f >= 1  or  U >= 1   => overload_risk

The forecast never authorizes a migration; the planner decides (Week 11).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.schemas.forecasting import (
    DUPLICATE_SNAPSHOT,
    HISTORY_GAP,
    INSUFFICIENT_HISTORY,
    INVALID_SNAPSHOT,
    MISSING_CAPACITY,
    MISSING_FEATURE,
    ForecastPrediction,
)
from src.twin.decision.overload_detector import OverloadDetector
from src.twin.forecasting.artifact import validate_artifact
from src.twin.forecasting.features import build_features, feature_names
from src.twin.forecasting.history import AppendStatus, TwinHistoryBuffer
from src.twin.forecasting.ood import FeatureRanges, OODGate, monitored_row
from src.twin.forecasting.runtime_rows import forecast_rows

READY = "READY"
NOT_READY = "NOT_READY"

# (controller_id, observed_at ISO, feature vector) -> None; used by the
# feature-parity test (M4) to compare runtime vectors with the dataset.
FeatureHook = Callable[[str, str, list[float]], None]


class ForecastArtifactError(RuntimeError):
    """The artifact directory violates contract C1 or cannot be loaded."""


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _load_model(model_dir: Path, card: Mapping[str, Any]) -> tuple[Any, Any]:
    name = card["model_name"]
    if name == "persistence":
        return None, None
    if name == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor()
        model.load_model(str(model_dir / str(card["model_file"])))
        # One row per call: all-core prediction waits for busy cores
        # (p95 23 ms vs ~0.5 ms with the testbed running, 16 Oct).
        model.set_params(n_jobs=1)
        return model, None
    import joblib

    model = joblib.load(model_dir / str(card["model_file"]))
    if hasattr(model, "n_jobs"):
        model.set_params(n_jobs=1)
    scaler = joblib.load(model_dir / str(card["scaler_file"])) if card.get("scaler_file") else None
    return model, scaler


class RuntimeForecaster:
    def __init__(
        self,
        model_dir: Path,
        threshold: float = 1.0,
        on_features: FeatureHook | None = None,
        ood_ignore: Sequence[str] = (),
    ) -> None:
        model_dir = Path(model_dir)
        errors = validate_artifact(model_dir)
        if errors:
            raise ForecastArtifactError("; ".join(errors))
        self.model_dir = model_dir
        self.card = _read_json(model_dir / "model_card.json")
        schema = _read_json(model_dir / "feature_schema.json")
        self.feature_names = list(schema["feature_names"])
        if tuple(self.feature_names) != feature_names():
            raise ForecastArtifactError("FEATURE_SCHEMA_MISMATCH")
        self.margin = _read_json(model_dir / "error_margin.json")
        self.model, self.scaler = _load_model(model_dir, self.card)
        self.history = TwinHistoryBuffer(
            history_steps=int(schema["history_samples_required"])
        )
        ranges_path = model_dir / "feature_ranges.json"
        self.ood: OODGate | None = (
            OODGate(
                FeatureRanges.from_dict(_read_json(ranges_path), self.feature_names),
                ignore=ood_ignore,
            )
            if ranges_path.is_file()
            else None
        )
        self.threshold = threshold
        self.detector = OverloadDetector(threshold)
        self.on_features = on_features
        self.horizon_seconds = float(self.card.get("horizon_seconds") or 0.0) or None
        self._reset_reason: dict[str, str] = {}

    @property
    def model_name(self) -> str:
        return str(self.card["model_name"])

    @property
    def model_version(self) -> str:
        return str(self.card["model_version"])

    def margin_for(self, controller_id: str) -> float:
        return float(self.margin["controllers"].get(controller_id, self.margin["global"]))

    # ------------------------------------------------------------------

    def update(self, twin_state: Mapping[str, Any]) -> list[ForecastPrediction]:
        """Feed one TwinState (dict from GET /api/v1/twin/state); call once per snapshot."""
        quality = twin_state.get("quality") or {}
        valid = quality.get("valid") is True
        snapshot_id = str(twin_state.get("snapshot_id", ""))
        created_at = twin_state.get("created_at")
        if not isinstance(created_at, (str, int, float, datetime)) or not snapshot_id:
            raise ValueError("twin_state needs created_at and snapshot_id")
        out: list[ForecastPrediction] = []
        for row in forecast_rows(twin_state):
            cid = str(row["controller_id"])
            status = self.history.append(cid, created_at, snapshot_id, row, snapshot_valid=valid)
            if not valid:
                out.append(self._not_ready(twin_state, cid, INVALID_SNAPSHOT))
                continue
            if status == AppendStatus.REJECTED_DUPLICATE:
                out.append(self._not_ready(twin_state, cid, DUPLICATE_SNAPSHOT))
                continue
            if status in (AppendStatus.RESET_GAP, AppendStatus.RESET_OUT_OF_ORDER):
                self._reset_reason[cid] = HISTORY_GAP
                if self.ood is not None:
                    self.ood.reset(cid)
            if not self.history.ready(cid):
                out.append(
                    self._not_ready(twin_state, cid, self._reset_reason.get(cid, INSUFFICIENT_HISTORY))
                )
                continue
            self._reset_reason.pop(cid, None)
            capacity = float(row.get("safe_capacity_pps") or 0.0)
            if capacity <= 0:
                out.append(self._not_ready(twin_state, cid, MISSING_CAPACITY))
                continue
            vector = self._features(cid)
            if vector is None:
                out.append(self._not_ready(twin_state, cid, MISSING_FEATURE))
                continue
            out.append(self._ready(twin_state, row, cid, capacity, vector))
        return out

    # ------------------------------------------------------------------

    def _features(self, cid: str) -> list[float] | None:
        records = [
            {
                **entry.row,
                "sequence_id": cid,
                "observed_at": datetime.fromtimestamp(entry.observed_at, tz=timezone.utc).isoformat(),
            }
            for entry in self.history.get_history(cid)
        ]
        try:
            latest = build_features(records)[-1]
        except (KeyError, ValueError):
            return None
        values = [latest.get(name) for name in self.feature_names]
        if any(v is None for v in values):
            return None
        vector = [float(v) for v in values if v is not None]
        if self.on_features is not None:
            self.on_features(cid, str(records[-1]["observed_at"]), vector)
        return vector

    def _predict(self, vector: list[float], current: float) -> float:
        if self.model is None:  # persistence artifact
            return current
        x = np.asarray([vector], dtype=float)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        y = float(self.model.predict(x)[0])
        return current + y if self.card["output"] == "delta" else y

    def _ready(
        self,
        twin_state: Mapping[str, Any],
        row: Mapping[str, Any],
        cid: str,
        capacity: float,
        vector: list[float],
    ) -> ForecastPrediction:
        current = float(row["utilization"])
        ood_on = False
        ood_features: tuple[str, ...] = ()
        if self.ood is not None:
            decision = self.ood.update(
                cid, monitored_row(vector, row, self.ood.ranges.monitor_only)
            )
            ood_on, ood_features = decision.active, decision.features
        start = time.perf_counter_ns()
        predicted = current if ood_on else self._predict(vector, current)
        inference_ms = (time.perf_counter_ns() - start) / 1e6
        margin = self.margin_for(cid)
        assessment = self.detector.assess(cid, max(predicted, 0.0), margin)
        quality = twin_state.get("quality") or {}
        return ForecastPrediction(
            snapshot_id=str(twin_state.get("snapshot_id", "")),
            controller_id=cid,
            status=READY,
            reason=None,
            model_name=self.model_name,
            model_version=self.model_version,
            current_utilization=current,
            predicted_utilization=predicted,
            predicted_packet_in_rate=predicted * capacity,
            error_margin_utilization=margin,
            upper_utilization_bound=predicted + margin,
            # Training-wheels rule: a measured overload is a risk whatever the
            # forecast says (burst onsets are not predictable from history).
            overload_risk=bool(assessment.risk or current >= self.threshold),
            ood_fallback=ood_on,
            ood_features=ood_features,
            inference_time_ms=inference_ms,
            quality_score=quality.get("quality_score"),
            consistency_score=quality.get("consistency_score"),
            horizon_seconds=self.horizon_seconds,
            predicted_at=datetime.now(timezone.utc).isoformat(),
        )

    def _not_ready(
        self, twin_state: Mapping[str, Any], cid: str, reason: str
    ) -> ForecastPrediction:
        quality = twin_state.get("quality") or {}
        return ForecastPrediction(
            snapshot_id=str(twin_state.get("snapshot_id", "")),
            controller_id=cid,
            status=NOT_READY,
            reason=reason,
            model_name=self.model_name,
            model_version=self.model_version,
            quality_score=quality.get("quality_score"),
            consistency_score=quality.get("consistency_score"),
            horizon_seconds=self.horizon_seconds,
            predicted_at=datetime.now(timezone.utc).isoformat(),
        )
