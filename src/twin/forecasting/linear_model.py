from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.preprocessing import (
    StandardScaler,
)


@dataclass
class LinearForecastModel:
    feature_names: list[str]
    scaler: StandardScaler
    model: LinearRegression

    def predict(
        self,
        records: list[dict[str, Any]],
    ) -> np.ndarray:
        features = records_to_matrix(
            records,
            self.feature_names,
        )

        scaled = self.scaler.transform(
            features
        )

        return np.asarray(
            self.model.predict(scaled),
            dtype=float,
        )


def records_to_matrix(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [
                float(record[name])
                for name in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "feature matrix must be 2D"
        )

    if not np.isfinite(matrix).all():
        raise ValueError(
            "feature matrix contains "
            "non-finite values"
        )

    return matrix


def fit_linear_regression(
    train_records: list[dict[str, Any]],
    feature_names: list[str],
) -> LinearForecastModel:
    if not train_records:
        raise ValueError(
            "training records cannot be empty"
        )

    features = records_to_matrix(
        train_records,
        feature_names,
    )

    targets = np.asarray(
        [
            float(
                record[
                    "future_utilization"
                ]
            )
            for record in train_records
        ],
        dtype=float,
    )

    if not np.isfinite(targets).all():
        raise ValueError(
            "training targets contain "
            "non-finite values"
        )

    scaler = StandardScaler()

    scaled_features = (
        scaler.fit_transform(features)
    )

    model = LinearRegression()

    model.fit(
        scaled_features,
        targets,
    )

    return LinearForecastModel(
        feature_names=list(
            feature_names
        ),
        scaler=scaler,
        model=model,
    )
