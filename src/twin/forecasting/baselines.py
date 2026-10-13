from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def persistence_predict(
    records: Iterable[dict[str, Any]],
) -> np.ndarray:
    """
    Persistence baseline:

        U_hat(t+h) = U(t)
    """
    predictions: list[float] = []

    for record in records:
        value = float(
            record["utilization"]
        )

        if not np.isfinite(value):
            raise ValueError(
                "utilization must be finite"
            )

        predictions.append(value)

    return np.asarray(
        predictions,
        dtype=float,
    )


def moving_average_predict(
    records: Iterable[dict[str, Any]],
    *,
    window: int,
) -> np.ndarray:
    """
    Historical moving-average baseline.

    The processed dataset already contains leakage-safe rolling
    features computed before history-incomplete rows are removed.
    Reuse those features instead of reconstructing history from
    the filtered ML-ready dataset.
    """
    if window not in (3, 5, 10):
        raise ValueError(
            "moving-average window must be one of: "
            "3, 5, 10"
        )

    field = f"util_mean_{window}"
    predictions: list[float] = []

    for record in records:
        if field not in record:
            raise ValueError(
                f"missing moving-average feature: {field}"
            )

        raw_value = record[field]

        if raw_value is None:
            raise ValueError(
                f"{field} must not be null"
            )

        value = float(raw_value)

        if not np.isfinite(value):
            raise ValueError(
                f"{field} must be finite"
            )

        predictions.append(value)

    return np.asarray(
        predictions,
        dtype=float,
    )
