from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.twin.forecasting.features import build_features
from src.twin.forecasting.history import (
    DEFAULT_MAX_GAP_SECONDS,
    HistoryNotReadyError,
    TwinHistoryBuffer,
)
from src.twin.forecasting.runtime_rows import forecast_rows

FeatureHook = Callable[[str, str, list[float]], None]


class RuntimeForecaster:
    """Runtime feature replay/forecast helper.

    The current parity checks use ``on_features`` to compare runtime vectors with
    dataset rows. Prediction output is intentionally simple (current utilization)
    because parity validation only asserts feature-vector equivalence.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        on_features: FeatureHook | None = None,
        history_steps: int | None = None,
        safety_buffer: int = 5,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    ) -> None:
        self.model_dir = model_dir
        self.feature_names, history_required = self._load_feature_schema(model_dir)
        if history_steps is None:
            history_steps = history_required
        self.on_features = on_features
        self.history = TwinHistoryBuffer(
            history_steps=history_steps,
            safety_buffer=safety_buffer,
            max_gap_seconds=max_gap_seconds,
        )

    @staticmethod
    def _load_feature_schema(model_dir: Path) -> tuple[list[str], int]:
        schema_path = model_dir / "feature_schema.json"
        with schema_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        names = data.get("feature_names")
        history_required = data.get("history_samples_required")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"invalid feature_schema.json in {model_dir}")
        if not isinstance(history_required, int) or history_required < 1:
            raise ValueError(f"invalid history_samples_required in {model_dir}")
        return list(names), history_required

    def update(self, twin_state: Mapping[str, Any]) -> list[dict[str, Any]]:
        quality = twin_state.get("quality")
        snapshot_valid = isinstance(quality, Mapping) and bool(quality.get("valid", False))

        rows = forecast_rows(twin_state)
        output: list[dict[str, Any]] = []

        for row in rows:
            controller_id = str(row["controller_id"])
            observed_at = str(row["observed_at"])
            snapshot_id = str(row["snapshot_id"])

            self.history.append(
                controller_id,
                observed_at,
                snapshot_id,
                row,
                snapshot_valid=snapshot_valid,
            )

            try:
                entries = self.history.get_history(controller_id)
            except HistoryNotReadyError:
                continue

            records = [
                {
                    **dict(entry.row),
                    "controller_id": controller_id,
                    "observed_at": str(entry.row["observed_at"]),
                    "sequence_id": controller_id,
                }
                for entry in entries
            ]
            featured = build_features(records)[-1]
            if any(featured.get(name) is None for name in self.feature_names):
                continue

            vector = [float(featured[name]) for name in self.feature_names]
            if self.on_features is not None:
                self.on_features(controller_id, observed_at, vector)

            output.append(
                {
                    "controller_id": controller_id,
                    "observed_at": observed_at,
                    "forecast_utilization": float(row["utilization"]),
                    "ready": True,
                }
            )

        return output
