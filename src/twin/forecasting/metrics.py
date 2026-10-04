from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


MAPE_EPSILON = 1e-8


def evaluate_regression(
    y_true: Any,
    y_pred: Any,
    *,
    mape_epsilon: float = MAPE_EPSILON,
) -> dict[str, float]:
    true = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    pred = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    if true.size == 0:
        raise ValueError(
            "cannot evaluate empty arrays"
        )

    if true.shape != pred.shape:
        raise ValueError(
            "y_true and y_pred must have "
            "the same shape"
        )

    if not np.isfinite(true).all():
        raise ValueError(
            "y_true contains non-finite values"
        )

    if not np.isfinite(pred).all():
        raise ValueError(
            "y_pred contains non-finite values"
        )

    if (
        not np.isfinite(mape_epsilon)
        or mape_epsilon <= 0
    ):
        raise ValueError(
            "mape_epsilon must be positive "
            "and finite"
        )

    absolute_error = np.abs(
        true - pred
    )

    denominator = np.maximum(
        np.abs(true),
        mape_epsilon,
    )

    return {
        "mae": float(
            mean_absolute_error(
                true,
                pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    true,
                    pred,
                )
            )
        ),
        "mape": float(
            np.mean(
                absolute_error
                / denominator
            )
            * 100.0
        ),
        "r2": float(
            r2_score(
                true,
                pred,
            )
        ),
        "p95_absolute_error": float(
            np.percentile(
                absolute_error,
                95,
            )
        ),
    }
