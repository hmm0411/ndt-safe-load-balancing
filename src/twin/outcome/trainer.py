"""Direct ML training of the outcome targets (Week 12, Day 2).

Direct ML learns ``f(X, A) -> y`` for each base target with its own model (plan
sections 20-22). Every candidate of the grid is fitted on TRAIN and scored on
VALIDATION by the RMSE of its target; the lowest wins (ties keep grid order).
The test split never enters this module: callers pass train and validation rows
only, and rows of any other split are refused.

The same search is reused on Day 3 for the residual learners (target = actual -
analytical), through ``target_values``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from src.experiments.migration.metrics import imbalance
from src.twin.outcome import random_forest, xgboost_model
from src.twin.outcome.metrics import regression_metrics
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS

FAMILIES = {random_forest.FAMILY: random_forest, xgboost_model.FAMILY: xgboost_model}
SHORT_NAMES = {"random_forest": "rf", "xgboost": "xgb"}


@dataclass(frozen=True)
class FamilyConfig:
    fixed: dict[str, Any]
    grid: dict[str, list[Any]]

    def candidates(self) -> list[dict[str, Any]]:
        names = list(self.grid)
        return [dict(zip(names, values)) for values in itertools.product(
            *(self.grid[name] for name in names))]


@dataclass(frozen=True)
class ModelsConfig:
    targets: tuple[str, ...]
    families: dict[str, FamilyConfig]


def load_models_config(path: Path) -> ModelsConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    targets = tuple(str(t) for t in data.get("targets", BASE_TARGETS))
    unknown = [t for t in targets if t not in BASE_TARGETS]
    if unknown or not targets:
        raise ValueError(f"targets must be base targets {BASE_TARGETS}, got {list(targets)}")
    families = {}
    for name in FAMILIES:
        if name not in data:
            raise ValueError(f"{path}: missing section '{name}'")
        section = data[name]
        grid = {str(k): list(v) for k, v in dict(section.get("grid") or {}).items()}
        if not grid or any(not values for values in grid.values()):
            raise ValueError(f"{path}: {name}.grid needs at least one value per parameter")
        families[name] = FamilyConfig(fixed=dict(section.get("fixed") or {}), grid=grid)
    return ModelsConfig(targets=targets, families=families)


def matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not rows:
        raise ValueError("no rows")
    x = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("feature matrix has non-finite values")
    return x


def _check_split(rows: Sequence[Mapping[str, Any]], split: str) -> None:
    other = sorted({str(r["split"]) for r in rows} - {split})
    if other:
        raise ValueError(f"expected only {split} rows, got {other}")


def actual(rows: Sequence[Mapping[str, Any]], target: str) -> np.ndarray:
    return np.asarray([float(r[target]) for r in rows], dtype=float)


TargetFn = Callable[[Sequence[Mapping[str, Any]], str], np.ndarray]


@dataclass
class FittedTarget:
    family: str
    target: str
    params: dict[str, Any]
    model: Any
    validation_rmse: float


def search(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    config: ModelsConfig,
    target_values: TargetFn = actual,
    families: Sequence[str] = tuple(FAMILIES),
) -> tuple[dict[tuple[str, str], FittedTarget], list[dict[str, Any]]]:
    """Best model per (family, target) and one record per candidate."""
    _check_split(train, "train")
    _check_split(validation, "validation")
    x_train, x_val = matrix(train, features), matrix(validation, features)
    best: dict[tuple[str, str], FittedTarget] = {}
    records: list[dict[str, Any]] = []
    for family in families:
        module = FAMILIES[family]
        fam = config.families[family]
        for target in config.targets:
            y_train = target_values(train, target)
            y_val = target_values(validation, target)
            for index, params in enumerate(fam.candidates()):
                model = module.make({**fam.fixed, **params})
                model.fit(x_train, y_train)
                m = regression_metrics(list(y_val), list(model.predict(x_val)))
                records.append({"family": family, "target": target, "candidate": index,
                                "params": params, **{f"validation_{k}": v for k, v in m.items()}})
                key = (family, target)
                if key not in best or m["rmse"] < best[key].validation_rmse:
                    best[key] = FittedTarget(family, target, {**fam.fixed, **params}, model,
                                             m["rmse"])
    return best, records


def predict_targets(models: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                    features: Sequence[str]) -> dict[str, list[float]]:
    """Base targets from one model each, then r95_post and i_post derived from them."""
    x = matrix(rows, features)
    out = {target: [float(v) for v in model.predict(x)] for target, model in models.items()}
    return with_derived(out)


def with_derived(base: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    out = {t: [float(v) for v in base[t]] for t in BASE_TARGETS}
    out["r95_post"] = [max(a, b) for a, b in zip(out["r_src95_post"], out["r_dst95_post"])]
    out["i_post"] = [imbalance({"src": a, "dst": b})
                     for a, b in zip(out["u_src_post"], out["u_dst_post"])]
    return out


def metrics_table(name: str, split: str, rows: Sequence[Mapping[str, Any]],
                  predictions: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    table = []
    for target in ALL_TARGETS:
        m = regression_metrics([float(r[target]) for r in rows], list(predictions[target]))
        table.append({"model": name, "split": split, "target": target, **m})
    return table


def save_models(models: Mapping[tuple[str, str], FittedTarget], out: Path,
                prefix: str = "direct") -> dict[str, str]:
    """``<prefix>_<rf|xgb>_<target>`` files; returns target key -> file name."""
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for (family, target), fitted in sorted(models.items()):
        path = FAMILIES[family].save(fitted.model, out / f"{prefix}_{SHORT_NAMES[family]}_{target}")
        files[f"{family}/{target}"] = path.name
    return files


def load_models(out: Path, family: str, targets: Sequence[str],
                prefix: str = "direct") -> dict[str, Any]:
    module = FAMILIES[family]
    return {t: module.load(out / f"{prefix}_{SHORT_NAMES[family]}_{t}{module.SUFFIX}")
            for t in targets}


def json_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Parameters as JSON-safe values (None stays null, numpy scalars become floats)."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"parameter {key} is not finite")
        out[key] = value
    return out
