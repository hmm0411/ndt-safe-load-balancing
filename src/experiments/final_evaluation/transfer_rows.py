"""Day 5: turn the 4C transfer runs into forecast rows for evaluate_transfer_4c.py.

Same steps as build_forecast_dataset.py for the 2C training data (valid snapshots only,
sequences cut at gaps > 2.5 s, build_features, U(t+h) inside a sequence), then the frozen
deployed model of the artifact predicts. Nothing is refit. Rows without a full feature vector
(the first 10 s of a sequence) or without U(t+h) are dropped, as in training.

The model is the raw deployed model: the OOD gate of the runtime is not applied, so the rule
judges the model itself; persistence is the comparison (transfer_4c_v1.yaml).
U is read as recorded: the orchestrator computed it with the capacity file it ran with, which
must be the one named in the transfer config (checked by the caller from run metadata).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.twin.forecasting.dataset import assign_sequence_ids
from src.twin.forecasting.features import build_features
from src.twin.forecasting.labels import attach_future_utilization
from src.twin.forecasting.loader import load_jsonl

MAX_GAP_SECONDS = 2.5   # build_forecast_dataset.py
PREFIX = "transfer4c-"


class Regressor(Protocol):
    def predict(self, x: Any) -> Any: ...


def load_transfer_samples(root: Path, prefix: str = PREFIX) -> list[dict[str, Any]]:
    run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not run_dirs:
        raise ValueError(f"no {prefix}* runs under {root}")
    records = []
    for run_dir in run_dirs:
        for record in load_jsonl(run_dir / "forecast_samples.jsonl"):
            if record.get("run_id") != run_dir.name:
                raise ValueError(f"run_id mismatch in {run_dir}")
            if record.get("snapshot_valid"):
                records.append(record)
    return records


def _complete(row: dict[str, Any], names: Sequence[str]) -> bool:
    values = [row.get(n) for n in (*names, "utilization", "future_utilization")]
    return all(v is not None and math.isfinite(float(v)) for v in values)


def feature_rows(records: Iterable[dict[str, Any]], feature_names: Sequence[str],
                 horizon_steps: int) -> list[dict[str, Any]]:
    segmented = assign_sequence_ids(records, max_gap_seconds=MAX_GAP_SECONDS)
    labeled = attach_future_utilization(build_features(segmented), horizon_steps=horizon_steps)
    return [r for r in labeled if _complete(r, feature_names)]


def predict_rows(rows: Sequence[dict[str, Any]], feature_names: Sequence[str],
                 model: Regressor, scaler: Any, output: str) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("no complete rows to predict")
    x = np.asarray([[float(r[n]) for n in feature_names] for r in rows], dtype=float)
    if scaler is not None:
        x = scaler.transform(x)
    y = np.asarray(model.predict(x), dtype=float)
    out = []
    for row, value in zip(rows, y):
        current = float(row["utilization"])
        out.append({"run_id": row["run_id"], "observed_at": row["observed_at"],
                    "workload_type": row["workload_type"],
                    "controller_id": row["controller_id"], "utilization": current,
                    "future_utilization": float(row["future_utilization"]),
                    "prediction": current + float(value) if output == "delta" else float(value)})
    return out
