"""Model selection score S_m and single-sample inference latency.

S_m compares model families (persistence, moving average, Linear Regression,
Random Forest, XGBoost) on the VALIDATION split only:

    S_m = 0.45 * RMSE/RMSE_p + 0.30 * P95|e|/P95|e|_p + 0.15 * MAE/MAE_p
          + 0.10 * min(1, latency_p95_ms / 1000)

where _p is persistence on the same split. Lower is better; persistence
itself scores 0.90 plus its latency term. The grids inside Random Forest and
XGBoost are still chosen by validation RMSE (tree_models._select); S_m only
picks the family that is deployed.

Latency is measured on the runtime path: one row of 27 features, one
predict call, no batching.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.twin.forecasting.metrics import percentile

DEFAULT_WEIGHTS: dict[str, float] = {
    "rmse": 0.45,
    "p95": 0.30,
    "mae": 0.15,
    "latency": 0.10,
}

# Same key names as metrics.evaluate_regression.
_ERROR_KEYS = {"rmse": "rmse", "p95": "p95_absolute_error", "mae": "mae"}

LATENCY_CAP_MS = 1000.0

TREE_MODELS = ("random_forest", "xgboost")


def _positive_finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return number


def validate_weights(weights: Mapping[str, float]) -> dict[str, float]:
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(
            f"selection weights must have exactly {sorted(DEFAULT_WEIGHTS)}"
        )
    out = {name: _positive_finite(value, f"weight {name}") for name, value in weights.items()}
    if abs(sum(out.values()) - 1.0) > 1e-9:
        raise ValueError("selection weights must sum to 1")
    return out


def selection_score(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> float:
    """S_m of one model. metrics/baseline use evaluate_regression keys plus
    ``inference_p95_ms``; baseline is persistence on the same split."""
    w = validate_weights(weights)
    score = 0.0
    for name, key in _ERROR_KEYS.items():
        reference = _positive_finite(baseline[key], f"baseline {key}")
        if reference == 0.0:
            raise ValueError(f"baseline {key} is 0; cannot normalize")
        score += w[name] * _positive_finite(metrics[key], key) / reference
    latency = _positive_finite(metrics["inference_p95_ms"], "inference_p95_ms")
    score += w["latency"] * min(1.0, latency / LATENCY_CAP_MS)
    return score


@dataclass(frozen=True)
class SelectionResult:
    selected: str
    scores: dict[str, float]
    ranking: list[str]
    tree_gap: float | None
    tree_near_tie: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_model(
    scores: Mapping[str, float],
    tie_tolerance: float = 0.01,
) -> SelectionResult:
    """Lowest S_m wins. A Random Forest / XGBoost gap below tie_tolerance is
    flagged so the report says "comparable" instead of naming a clear winner."""
    if not scores:
        raise ValueError("no candidate scores")
    for name, value in scores.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"score of {name} is not finite")
    ranking = sorted(scores, key=lambda name: (float(scores[name]), name))
    gap: float | None = None
    if all(name in scores for name in TREE_MODELS):
        gap = abs(float(scores[TREE_MODELS[0]]) - float(scores[TREE_MODELS[1]]))
    return SelectionResult(
        selected=ranking[0],
        scores={name: float(scores[name]) for name in ranking},
        ranking=ranking,
        tree_gap=gap,
        tree_near_tie=gap is not None and gap < tie_tolerance,
    )


# ---------------------------------------------------------------------------
# Single-sample latency (runtime path).
# ---------------------------------------------------------------------------


def measure_latency_ms(
    predict_one: Callable[[], Any],
    *,
    warmup: int = 50,
    repeats: int = 1000,
) -> dict[str, float]:
    """Wall time of one call, in ms: warm-up first, then ``repeats`` timed calls."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats > 0")
    for _ in range(warmup):
        predict_one()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        predict_one()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "repeats": float(repeats),
        "warmup": float(warmup),
        "mean_ms": math.fsum(samples) / len(samples),
        "p50_ms": percentile(samples, 50.0),
        "p95_ms": percentile(samples, 95.0),
        "p99_ms": percentile(samples, 99.0),
        "max_ms": max(samples),
    }


def single_sample_predictors(
    feature_names: Sequence[str],
    row: np.ndarray,
    *,
    ma_window: int = 3,
    linear_model: Any = None,
    scaler: Any = None,
    random_forest: Any = None,
    xgboost: Any = None,
) -> dict[str, Callable[[], float]]:
    """One zero-argument callable per model, each forecasting U(t+h) for ``row``.

    ``row`` has shape (1, n_features) in ``feature_names`` order. Models that
    are None are skipped; persistence and the moving average are always there.
    """
    x = np.asarray(row, dtype=float)
    if x.shape != (1, len(feature_names)):
        raise ValueError(f"row must have shape (1, {len(feature_names)}), got {x.shape}")
    index = {name: i for i, name in enumerate(feature_names)}
    u_index = index["utilization"]
    ma_indices = [u_index] + [index[f"utilization_lag_{lag}"] for lag in range(1, ma_window)]

    def persistence() -> float:
        return float(x[0, u_index])

    def moving_average() -> float:
        return float(np.mean(x[0, ma_indices]))

    out: dict[str, Callable[[], float]] = {
        "persistence": persistence,
        f"moving_average_w{ma_window}": moving_average,
    }
    if linear_model is not None:
        if scaler is None:
            raise ValueError("linear_model needs its scaler")

        def linear() -> float:
            return float(linear_model.predict(scaler.transform(x))[0])

        out["linear_regression"] = linear
    for name, model in (("random_forest", random_forest), ("xgboost", xgboost)):
        if model is not None:
            out[name] = _delta_predictor(model, x, u_index)
    return out


def _delta_predictor(model: Any, x: np.ndarray, u_index: int) -> Callable[[], float]:
    def predict() -> float:
        return float(x[0, u_index]) + float(model.predict(x)[0])

    return predict
