"""Error metrics and the validation selection score of the outcome models.

Per target: MAE, RMSE, P95 of |error| (linear interpolation) and R^2. The
selection score (plan section 35) weights normalized RMSE per target:

    S = sum_t w_t * RMSE_t / (P95(y_t) - P5(y_t) + eps)        (lower is better)

with P5/P95 of the true values of the SAME split. It never replaces the per-target
table: a bad target must stay visible.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

DEFAULT_WEIGHTS = {"u_dst_post": 0.35, "u_src_post": 0.25, "r95_post": 0.25, "i_post": 0.15}


def _pairs(y_true: Sequence[float], y_pred: Sequence[float]) -> list[tuple[float, float]]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} true vs {len(y_pred)} predicted")
    if not y_true:
        raise ValueError("no values")
    pairs = [(float(t), float(p)) for t, p in zip(y_true, y_pred)]
    if not all(math.isfinite(t) and math.isfinite(p) for t, p in pairs):
        raise ValueError("values must be finite")
    return pairs


def quantile(values: Sequence[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("quantile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    s = sorted(float(v) for v in values)
    k = (len(s) - 1) * q
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    pairs = _pairs(y_true, y_pred)
    errors = [p - t for t, p in pairs]
    n = len(pairs)
    mean_t = sum(t for t, _ in pairs) / n
    ss_tot = sum((t - mean_t) ** 2 for t, _ in pairs)
    ss_res = sum(e * e for e in errors)
    return {
        "n": float(n),
        "mae": sum(abs(e) for e in errors) / n,
        "rmse": math.sqrt(ss_res / n),
        "p95_abs_error": quantile([abs(e) for e in errors], 0.95),
        "bias": sum(errors) / n,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def normalized_rmse(rmse: float, y_true: Sequence[float], eps: float = 1e-9) -> float:
    spread = quantile(y_true, 0.95) - quantile(y_true, 0.05)
    return rmse / (spread + eps)


def selection_score(
    y_true: Mapping[str, Sequence[float]],
    y_pred: Mapping[str, Sequence[float]],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    eps: float = 1e-9,
) -> dict[str, float]:
    """Weighted normalized RMSE plus each target's term (keys ``n_<target>``)."""
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("selection weights must sum to 1")
    out: dict[str, float] = {}
    score = 0.0
    for target, weight in weights.items():
        rmse = regression_metrics(y_true[target], y_pred[target])["rmse"]
        term = normalized_rmse(rmse, y_true[target], eps)
        out[f"n_{target}"] = term
        score += weight * term
    out["score"] = score
    return out
