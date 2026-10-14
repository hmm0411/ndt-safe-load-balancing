"""Forecast error margin m_f (split conformal) and its coverage.

m_f is the finite-sample q95 of |U - U_hat| on the VALIDATION split of the
deployed model: with n validation errors sorted ascending, take element
k = ceil((n + 1) * 0.95) (1-based) instead of np.quantile. For exchangeable
data this guarantees coverage >= 95% on new samples (split conformal).

The model must NOT be refit on train + validation after m_f is computed:
m_f has to be the error of the exact model that is deployed, on data it has
not seen.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

METHOD = "validation_abs_error_finite_sample_q95"


def finite_sample_quantile(values: Any, level: float = 0.95) -> float:
    """Element ceil((n + 1) * level) of the sorted values (clipped to n)."""
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    v = np.sort(np.asarray(values, dtype=float).reshape(-1))
    if v.size == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(v).all():
        raise ValueError("values must be finite")
    k = math.ceil((len(v) + 1) * level)
    return float(v[min(k, len(v)) - 1])


def _errors(y_true: Any, y_pred: Any) -> np.ndarray:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    return true - pred


def error_margins(
    y_true: Any,
    y_pred: Any,
    controller_ids: Sequence[str],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """error_margin.json content (contract C1). e > 0: forecast below actual."""
    e = _errors(y_true, y_pred)
    ids = np.asarray([str(cid) for cid in controller_ids])
    if ids.shape != e.shape:
        raise ValueError("controller_ids must have one entry per error")
    level = 1.0 - alpha
    return {
        "method": METHOD,
        "alpha": alpha,
        "global": finite_sample_quantile(np.abs(e), level),
        "controllers": {
            cid: finite_sample_quantile(np.abs(e[ids == cid]), level)
            for cid in sorted(set(ids.tolist()))
        },
        "n_validation": int(e.size),
        "n_validation_per_controller": {
            cid: int((ids == cid).sum()) for cid in sorted(set(ids.tolist()))
        },
        # Reported only: the overload rule cares about under-forecasts.
        "one_sided_q95": finite_sample_quantile(e, level),
    }


def margin_for(margins: dict[str, Any], controller_id: str) -> float:
    """Per-controller m_f, falling back to the global value."""
    return float(margins["controllers"].get(controller_id, margins["global"]))


def coverage(y_true: Any, y_pred: Any, margin: Any) -> float:
    """Share of samples with |U - U_hat| <= m_f (margin: scalar or per sample)."""
    e = np.abs(_errors(y_true, y_pred))
    if e.size == 0:
        raise ValueError("inputs must not be empty")
    return float(np.mean(e <= np.asarray(margin, dtype=float)))


def overload_confusion(
    y_true: Any,
    upper_bound: Any,
    threshold: float = 1.0,
) -> dict[str, float]:
    """Alarm quality of U_hat + m_f >= threshold against the actual U(t+h)."""
    actual = np.asarray(y_true, dtype=float).reshape(-1) >= threshold
    alarm = np.asarray(upper_bound, dtype=float).reshape(-1) >= threshold
    if actual.shape != alarm.shape:
        raise ValueError("y_true and upper_bound must have the same length")
    missed = int((actual & ~alarm).sum())
    false_alarm = int((alarm & ~actual).sum())
    positives = int(actual.sum())
    negatives = int(actual.size) - positives
    return {
        "samples": float(actual.size),
        "actual_overload": float(positives),
        "alarms": float(alarm.sum()),
        "missed_overload": float(missed),
        "false_alarm": float(false_alarm),
        "missed_overload_rate": missed / positives if positives else float("nan"),
        "false_alarm_rate": false_alarm / negatives if negatives else float("nan"),
    }


def coverage_by_group(
    y_true: Any,
    y_pred: Any,
    margins: Sequence[float],
    groups: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Coverage and sample count per group label (controller or workload)."""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    m = np.asarray(margins, dtype=float).reshape(-1)
    index: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        index[str(group)].append(i)
    return {
        group: {
            "coverage": coverage(true[rows], pred[rows], m[rows]),
            "samples": float(len(rows)),
        }
        for group, rows in sorted(index.items())
    }
