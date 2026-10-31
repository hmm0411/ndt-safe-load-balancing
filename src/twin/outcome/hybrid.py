"""Hybrid outcome model: Analytical prediction plus a learned residual (Week 12, Day 3).

    y_H = max(0, y_A + r_hat)       for each base target
    r95_post, i_post                derived from the four base values (targets.py)

The floor at 0 only removes physically impossible values (a negative rate or
response time); the Analytical model already floors the source rate the same way.
There is no upper clamp: U > 1 is the overload the Safety Gate must see. With a
zero residual, Hybrid returns exactly the Analytical prediction.

Residual models are saved as ``residual_<rf|xgb>_<target>`` next to the Direct ones.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from src.twin.outcome import trainer
from src.twin.outcome.residual import ANALYTICAL_PREFIX
from src.twin.outcome.targets import BASE_TARGETS

PREFIX = "residual"


def analytical_base(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """The four Analytical base predictions of each row."""
    return {t: [float(r[f"{ANALYTICAL_PREFIX}{t}"]) for r in rows] for t in BASE_TARGETS}


def _sums(analytical: Mapping[str, Sequence[float]],
          residuals: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    out = {}
    for target in BASE_TARGETS:
        a, r = analytical[target], residuals[target]
        if len(a) != len(r):
            raise ValueError(f"{target}: {len(a)} analytical vs {len(r)} residual values")
        out[target] = [float(x) + float(y) for x, y in zip(a, r)]
    return out


def combine(analytical: Mapping[str, Sequence[float]],
            residuals: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Hybrid predictions of the six targets from Analytical and residual base values."""
    sums = _sums(analytical, residuals)
    return trainer.with_derived({t: [max(0.0, v) for v in sums[t]] for t in BASE_TARGETS})


def floored(analytical: Mapping[str, Sequence[float]],
            residuals: Mapping[str, Sequence[float]]) -> dict[str, int]:
    """How many values per base target the floor at 0 changed (reported, expected 0)."""
    return {t: sum(1 for v in values if v < 0.0)
            for t, values in _sums(analytical, residuals).items()}


def predict_residuals(models: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      features: Sequence[str]) -> dict[str, list[float]]:
    x = trainer.matrix(rows, features)
    missing = [t for t in BASE_TARGETS if t not in models]
    if missing:
        raise ValueError(f"missing residual models: {missing}")
    return {t: [float(v) for v in models[t].predict(x)] for t in BASE_TARGETS}


def predict(models: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
            features: Sequence[str]) -> dict[str, list[float]]:
    """Hybrid predictions; ``features`` are the Hybrid features (residual.hybrid_features)."""
    return combine(analytical_base(rows), predict_residuals(models, rows, features))


def save(models: Mapping[tuple[str, str], trainer.FittedTarget], out: Path) -> dict[str, str]:
    return trainer.save_models(models, out, prefix=PREFIX)


def load(out: Path, family: str) -> dict[str, Any]:
    return trainer.load_models(out, family, BASE_TARGETS, prefix=PREFIX)
