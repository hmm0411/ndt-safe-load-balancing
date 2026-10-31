"""Residual targets and Hybrid features of the outcome model (Week 12, Day 3).

Hybrid = Analytical + learned correction (plan sections 23-25):

    r = y - y_A                    residual of one base target
    r_hat = g(X, A, y_A)           learned on TRAIN, chosen on VALIDATION
    y_H = y_A + r_hat              see ``src.twin.outcome.hybrid``

The residual learner sees the model features of the dataset plus the four
Analytical predictions of the base targets. Those are computed from the pre
window only (``analytical.inputs_from_window``), so they carry no post-action
information. Derived Analytical columns (r95_post, i_post) are not added: they
are functions of the four base ones.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.twin.outcome.features import is_forbidden
from src.twin.outcome.targets import BASE_TARGETS

ANALYTICAL_PREFIX = "analytical_"
ANALYTICAL_FEATURES = tuple(f"{ANALYTICAL_PREFIX}{t}" for t in BASE_TARGETS)


def hybrid_features(model_features: Sequence[str]) -> list[str]:
    """Model features of the dataset followed by the four Analytical predictions."""
    names = list(model_features)
    if not names:
        raise ValueError("no model features")
    if len(set(names)) != len(names):
        raise ValueError("duplicate model features")
    bad = [n for n in names if is_forbidden(n) or n.startswith(ANALYTICAL_PREFIX)]
    if bad:
        raise ValueError(f"not allowed as residual features: {bad}")
    return names + list(ANALYTICAL_FEATURES)


def _check_target(target: str) -> None:
    if target not in BASE_TARGETS:
        raise ValueError(f"residuals are learned for base targets {BASE_TARGETS}, got {target}")


def residual(row: Mapping[str, Any], target: str) -> float:
    """r = y - y_A of one row."""
    _check_target(target)
    value = float(row[target]) - float(row[f"{ANALYTICAL_PREFIX}{target}"])
    if not math.isfinite(value):
        raise ValueError(f"{row.get('attempt_id', '?')}: residual of {target} is not finite")
    return value


def residual_values(rows: Sequence[Mapping[str, Any]], target: str) -> np.ndarray:
    """Residual of one base target for every row (``target_values`` of ``trainer.search``)."""
    return np.asarray([residual(row, target) for row in rows], dtype=float)
