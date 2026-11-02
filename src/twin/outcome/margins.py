"""One-sided uncertainty margins and their coverage (Week 12, Day 5).

Plan sections 37-39: the Safety Gate checks an upper bound, so only under-prediction
is dangerous. On VALIDATION:

    e+ = max(0, actual - predicted)          m = finite_sample_quantile(e+, 1 - alpha)

with the split-conformal rank ceil((n + 1)(1 - alpha)) (n = 60, alpha = 0.05 -> the 58th
smallest). The TEST split only reports coverage, P(actual <= predicted + m); a margin is
never re-tuned after the test is seen.

``gate_confusion`` counts what a gate "predicted + m >= threshold -> reject" would have
done on the rows: missed overloads are the safety failures, false alarms the cost.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from src.twin.forecasting.margin import finite_sample_quantile


def _pairs(actual: Sequence[float], predicted: Sequence[float]) -> list[tuple[float, float]]:
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} actual vs {len(predicted)} predicted values")
    if not actual:
        raise ValueError("no values")
    pairs = [(float(a), float(p)) for a, p in zip(actual, predicted)]
    if not all(math.isfinite(a) and math.isfinite(p) for a, p in pairs):
        raise ValueError("values must be finite")
    return pairs


def under_errors(actual: Sequence[float], predicted: Sequence[float]) -> list[float]:
    """e+ = max(0, actual - predicted): how much the prediction fell short."""
    return [max(0.0, a - p) for a, p in _pairs(actual, predicted)]


def one_sided_margin(actual: Sequence[float], predicted: Sequence[float],
                     level: float = 0.95) -> dict[str, Any]:
    errors = under_errors(actual, predicted)
    n = len(errors)
    return {
        "value": finite_sample_quantile(errors, level),
        "n": n,
        "rank": min(math.ceil((n + 1) * level), n),
        "level": level,
        "under_predicted": sum(1 for e in errors if e > 0),
        "max_under_error": max(errors),
        "method": "split conformal, one-sided: element ceil((n + 1) * level) of the sorted "
                  "max(0, actual - predicted) on validation",
    }


def coverage(actual: Sequence[float], predicted: Sequence[float],
             margin: float) -> dict[str, Any]:
    pairs = _pairs(actual, predicted)
    covered = sum(1 for a, p in pairs if a <= p + margin)
    return {"n": len(pairs), "covered": covered, "coverage": covered / len(pairs)}


def gate_confusion(actual: Sequence[float], predicted: Sequence[float], margin: float,
                   threshold: float = 1.0) -> dict[str, Any]:
    """Overload = actual >= threshold; flagged = predicted + margin >= threshold."""
    tp = fp = fn = tn = 0
    for a, p in _pairs(actual, predicted):
        over, flagged = a >= threshold, p + margin >= threshold
        if over and flagged:
            tp += 1
        elif flagged:
            fp += 1
        elif over:
            fn += 1
        else:
            tn += 1
    return {"threshold": threshold, "overload": tp + fn, "flagged": tp + fp,
            "true_positive": tp, "false_alarm": fp, "missed_overload": fn, "true_negative": tn,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else None,
            "miss_rate": fn / (tp + fn) if tp + fn else None}


def coverage_by_group(rows: Sequence[Mapping[str, Any]], actual_key: str,
                      predicted: Sequence[float], margin: float,
                      groups: Mapping[str, Callable[[Mapping[str, Any]], Any]]
                      ) -> list[dict[str, Any]]:
    """Coverage on all rows and on each value of each group function."""
    if len(rows) != len(predicted):
        raise ValueError("rows and predictions differ in length")
    out = [{"group": "all", "value": "all",
            **coverage([float(r[actual_key]) for r in rows], predicted, margin)}]
    for name, key in groups.items():
        values = sorted({str(key(r)) for r in rows})
        for value in values:
            index = [i for i, r in enumerate(rows) if str(key(r)) == value]
            out.append({"group": name, "value": value,
                        **coverage([float(rows[i][actual_key]) for i in index],
                                   [predicted[i] for i in index], margin)})
    return out
