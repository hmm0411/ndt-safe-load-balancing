"""Runtime forecasting inside the orchestrator (Week 10, V3).

build_snapshot() hands every new TwinState to ForecastService.on_twin_state();
GET /api/v1/twin/forecast returns ForecastService.payload(). A forecasting
failure is logged and kept out of the telemetry path: it never marks a
snapshot as failed and never stops the telemetry loop.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from src.orchestrator.current_state import CurrentStateStore
from src.telemetry.writer import JsonlWriter
from src.twin.forecasting.predictor import RuntimeForecaster

MODEL_NOT_LOADED = "MODEL_NOT_LOADED"
DISABLED = "DISABLED"
NOT_READY = "NOT_READY"
OK = "OK"


class ForecastService:
    def __init__(
        self,
        forecaster: RuntimeForecaster | None,
        store: CurrentStateStore,
        writer: JsonlWriter | None = None,
        output_dir: Path | None = None,
        *,
        enabled: bool = True,
        load_error: str | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.store = store
        self.writer = writer
        self.output_dir = output_dir
        self.enabled = enabled
        self.load_error = load_error
        self.last_error: str | None = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        root: Path,
        store: CurrentStateStore,
        writer: JsonlWriter | None = None,
    ) -> ForecastService:
        output_dir = root / str(config.get("output_dir", "data/raw/forecast"))
        if not config.get("enabled", False):
            return cls(None, store, writer, output_dir, enabled=False)
        try:
            forecaster = RuntimeForecaster(
                root / str(config["model_dir"]),
                threshold=float(config.get("overload_threshold", 1.0)),
                ood_ignore=tuple(str(n) for n in config.get("ood_ignore_features") or ()),
            )
        except Exception as exc:  # the orchestrator keeps running without forecasts
            print(f"[forecast] load error={exc}", flush=True)
            return cls(None, store, writer, output_dir, load_error=str(exc))
        return cls(forecaster, store, writer, output_dir)

    def on_twin_state(self, twin_state: Mapping[str, Any], run_id: str = "dev") -> None:
        if self.forecaster is None:
            return
        start = time.perf_counter_ns()
        try:
            predictions = self.forecaster.update(twin_state)
            # Whole decision-side cost of one snapshot (rows, history, features,
            # OOD gate, predict for every controller): input to write_horizon.py.
            service_ms = (time.perf_counter_ns() - start) / 1e6
            payload = {
                "status": OK,
                "snapshot_id": twin_state.get("snapshot_id"),
                "created_at": twin_state.get("created_at"),
                "model_name": self.forecaster.model_name,
                "model_version": self.forecaster.model_version,
                "horizon_seconds": self.forecaster.horizon_seconds,
                "ood_ignored_features": list(self.forecaster.ood.ignore)
                if self.forecaster.ood is not None
                else [],
                "service_time_ms": service_ms,
                "controllers": [p.to_dict() for p in predictions],
            }
            self.store.set_forecast(payload)
            if self.writer is not None and self.output_dir is not None:
                self.writer.append(self.output_dir / f"{run_id}.jsonl", payload)
            self.last_error = None
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[forecast] error={exc}", flush=True)

    def payload(self) -> dict[str, Any]:
        """Body of GET /api/v1/twin/forecast. Read-only; no migration fields."""
        if self.forecaster is None:
            status = MODEL_NOT_LOADED if self.enabled else DISABLED
            return {"status": status, "error": self.load_error}
        value = self.store.forecast_dict()
        if value is None:
            return {"status": NOT_READY, "error": self.last_error}
        if self.last_error is not None:
            value["last_error"] = self.last_error
        return value
