"""Day 5-6: outcome part of the 2C -> 4C transfer test, from the 4C probe-off collection.

My, 24/11 (option b): the attempts are the MIGRATE rows at the destination levels the 2C
models were trained on (0.45, 0.75) of migration-outcome-4c-noprobe-v1, collected by Vy with
the probe-off capacity, instead of the separate 50-attempt run made with the probe-on
capacity. Rows are built exactly as the 2C dataset (src/twin/outcome/dataset.build_rows: valid,
training-eligible, COMMITTED, the capacity of every snapshot checked against the file), then
the frozen post-migration-v1 roles predict: Analytical, direct_xgb, hybrid_xgb. Nothing refit.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.twin.outcome import hybrid, residual, trainer
from src.twin.outcome.targets import BASE_TARGETS
from src.twin.outcome.dataset import attempt_dirs, build_rows, capacity_info, saturation_flags

LEVELS = (0.45, 0.75)
TARGET = "u_dst_post"
Predict = Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Sequence[float]]]


def split_all(raw_root: Path, name: str = "transfer") -> dict[str, str]:
    """Every attempt in one split: the transfer test only scores frozen models."""
    out = {}
    for attempt in attempt_dirs(raw_root):
        meta = json.loads((attempt / "metadata.json").read_text(encoding="utf-8"))
        out[str(meta["attempt_id"])] = name
    return out


def collect(raw_root: Path, capacity: Path, config: Any,
            levels: Sequence[float] = LEVELS) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows, _noop, excluded = build_rows(raw_root, split_all(raw_root),
                                       capacity_info(capacity)["c_safe_pps"], config,
                                       saturation_flags(raw_root))
    keep = [r for r in rows
            if any(abs(float(r["destination_load_level"]) - lv) < 1e-9 for lv in levels)]
    return keep, excluded


def frozen_predictors(model_dir: Path) -> dict[str, Predict]:
    schema = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))
    features = [str(f) for f in schema["model_features"]]
    family = str(json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
                 ["hybrid_family"])
    direct = trainer.load_models(model_dir / "direct_model", family, BASE_TARGETS,
                                 prefix="direct")
    residuals = hybrid.load(model_dir / "residual_models", family)
    hybrid_features = residual.hybrid_features(features)
    return {
        "analytical": lambda rows: {TARGET: [float(r[f"analytical_{TARGET}"]) for r in rows]},
        "direct_xgb": lambda rows: trainer.predict_targets(direct, rows, features),
        "hybrid_xgb": lambda rows: hybrid.predict(residuals, rows, hybrid_features),
    }


def outcome_rows(rows: Sequence[Mapping[str, Any]],
                 predictors: Mapping[str, Predict]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("no eligible MIGRATE row at the transfer levels")
    preds = {name: list(fn(rows)[TARGET]) for name, fn in predictors.items()}
    out = []
    for i, row in enumerate(rows):
        item = {"attempt_id": row["attempt_id"], "workload": row["workload"],
                "phase": row["phase"], "destination_load_level": row["destination_load_level"],
                "direction": row["direction"], "repeat": row["repeat"],
                TARGET: float(row[TARGET])}
        item.update({f"pred_{name}": values[i] for name, values in preds.items()})
        out.append(item)
    return out
