"""Random Forest and XGBoost forecasters for future controller utilization.

Design choices (see the forecast-models-v2 training manifest):

- Target = change in utilization, U(t+h) - U(t); prediction = U(t) + change.
  Tree models cannot predict outside the target range seen in training, so a
  level target collapses on unseen load levels. A delta target keeps the
  forecast anchored at the current load and only learns the trend.
- Hyperparameters are chosen on the VALIDATION split only (lowest RMSE of the
  final utilization forecast). Test and unseen are evaluated once, afterwards.
- No scaling: trees are invariant to monotone feature transforms.
- Fixed random_state so runs are reproducible.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from src.twin.forecasting.linear_model import records_to_matrix
from src.twin.forecasting.metrics import evaluate_regression

RANDOM_STATE = 42

RANDOM_FOREST_GRID: dict[str, tuple[Any, ...]] = {
    "max_depth": (6, 10, None),
    "min_samples_leaf": (5, 20),
    "max_features": (0.5, 1.0),
}
RANDOM_FOREST_FIXED: dict[str, Any] = {
    "n_estimators": 300,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGBOOST_GRID: dict[str, tuple[Any, ...]] = {
    "max_depth": (3, 5),
    "learning_rate": (0.05, 0.1),
    "min_child_weight": (1, 5),
}
XGBOOST_FIXED: dict[str, Any] = {
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def current_utilization(records: Iterable[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["utilization"]) for row in records], dtype=float)


def delta_target(records: list[dict[str, Any]]) -> np.ndarray:
    future = np.asarray(
        [float(row["future_utilization"]) for row in records], dtype=float
    )
    delta = future - current_utilization(records)
    if not np.isfinite(delta).all():
        raise ValueError("target contains non-finite values")
    return delta


@dataclass
class TreeForecastModel:
    """A fitted tree model that forecasts U(t+h) = U(t) + predicted change."""

    name: str
    feature_names: list[str]
    model: Any
    params: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    def predict(self, records: list[dict[str, Any]]) -> np.ndarray:
        features = records_to_matrix(records, self.feature_names)
        change = np.asarray(self.model.predict(features), dtype=float)
        return current_utilization(records) + change

    def feature_importance(self) -> dict[str, float]:
        values = getattr(self.model, "feature_importances_", None)
        if values is None:
            return {}
        pairs = zip(self.feature_names, (float(v) for v in values), strict=True)
        return dict(sorted(pairs, key=lambda item: item[1], reverse=True))


def _grid(grid: Mapping[str, tuple[Any, ...]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _select(
    name: str,
    candidates: list[dict[str, Any]],
    fit_one: Callable[[dict[str, Any]], TreeForecastModel],
    validation: list[dict[str, Any]],
) -> tuple[TreeForecastModel, list[dict[str, Any]]]:
    """Fit every candidate, keep the lowest validation RMSE."""
    if not validation:
        raise ValueError(f"{name}: validation split is empty")

    y_val = np.asarray(
        [float(row["future_utilization"]) for row in validation], dtype=float
    )
    best: TreeForecastModel | None = None
    best_rmse = math.inf
    table: list[dict[str, Any]] = []

    for params in candidates:
        model = fit_one(params)
        metrics = evaluate_regression(y_val, model.predict(validation))
        table.append({"params": params, **model.extra, **metrics})
        if metrics["rmse"] < best_rmse:
            best, best_rmse = model, metrics["rmse"]

    assert best is not None
    return best, table


def fit_random_forest(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    feature_names: list[str],
    grid: Mapping[str, tuple[Any, ...]] = RANDOM_FOREST_GRID,
) -> tuple[TreeForecastModel, list[dict[str, Any]]]:
    from sklearn.ensemble import RandomForestRegressor

    x_train = records_to_matrix(train, feature_names)
    y_train = delta_target(train)

    def fit_one(params: dict[str, Any]) -> TreeForecastModel:
        model = RandomForestRegressor(**RANDOM_FOREST_FIXED, **params)
        model.fit(x_train, y_train)
        return TreeForecastModel("random_forest", list(feature_names), model, params)

    return _select("random_forest", _grid(grid), fit_one, validation)


def fit_xgboost(
    train: list[dict[str, Any]],
    validation: list[dict[str, Any]],
    feature_names: list[str],
    grid: Mapping[str, tuple[Any, ...]] = XGBOOST_GRID,
) -> tuple[TreeForecastModel, list[dict[str, Any]]]:
    from xgboost import XGBRegressor

    x_train = records_to_matrix(train, feature_names)
    y_train = delta_target(train)
    x_val = records_to_matrix(validation, feature_names)
    y_val = delta_target(validation)

    def fit_one(params: dict[str, Any]) -> TreeForecastModel:
        model = XGBRegressor(**XGBOOST_FIXED, **params)
        # Early stopping watches the validation split, which is the
        # model-selection split; test and unseen are never seen here.
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        best_iteration = int(getattr(model, "best_iteration", -1))
        return TreeForecastModel(
            "xgboost",
            list(feature_names),
            model,
            params,
            extra={"best_iteration": best_iteration},
        )

    return _select("xgboost", _grid(grid), fit_one, validation)
