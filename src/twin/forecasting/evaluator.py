from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from time import perf_counter
from typing import Any

import numpy as np

from src.twin.forecasting.metrics import (
    evaluate_regression,
)


PredictionFunction = Callable[
    [list[dict[str, Any]]],
    np.ndarray,
]


def evaluate_predictions(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
) -> dict[str, float]:
    if len(records) != len(predictions):
        raise ValueError(
            "record and prediction counts differ"
        )

    targets = np.asarray(
        [
            float(record["future_utilization"])
            for record in records
        ],
        dtype=float,
    )

    return evaluate_regression(
        targets,
        predictions,
    )


def evaluate_predictor(
    records: list[dict[str, Any]],
    predictor: PredictionFunction,
) -> tuple[dict[str, float], np.ndarray]:
    start = perf_counter()

    predictions = predictor(records)

    elapsed_seconds = (
        perf_counter() - start
    )

    metrics = evaluate_predictions(
        records,
        predictions,
    )

    metrics["inference_total_ms"] = (
        elapsed_seconds * 1000.0
    )

    metrics["inference_per_sample_ms"] = (
        elapsed_seconds
        * 1000.0
        / len(records)
    )

    return metrics, predictions


def evaluate_by_workload(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    if len(records) != len(predictions):
        raise ValueError(
            "record and prediction counts differ"
        )

    grouped: dict[
        str,
        list[tuple[float, float]],
    ] = defaultdict(list)

    for record, prediction in zip(
        records,
        predictions,
        strict=True,
    ):
        workload = str(
            record["workload_type"]
        )

        grouped[workload].append(
            (
                float(
                    record[
                        "future_utilization"
                    ]
                ),
                float(prediction),
            )
        )

    result: dict[
        str,
        dict[str, float],
    ] = {}

    for workload, values in sorted(
        grouped.items()
    ):
        y_true = [
            value[0]
            for value in values
        ]

        y_pred = [
            value[1]
            for value in values
        ]

        result[workload] = (
            evaluate_regression(
                y_true,
                y_pred,
            )
        )

    return result
