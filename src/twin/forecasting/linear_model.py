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


def near_constant_features(
    train_records: list[dict[str, Any]],
    feature_names: list[str],
    *,
    min_relative_std: float = 0.05,
) -> list[str]:
    """
    Features whose training spread is below min_relative_std of their scale.

    Such a column carries almost no signal in training, yet any different
    value at inference time becomes a huge standardized input.
    """
    features = records_to_matrix(
        train_records,
        feature_names,
    )

    std = features.std(axis=0)
    scale = np.maximum(
        np.abs(features.mean(axis=0)),
        1e-12,
    )

    return [
        name
        for name, ratio in zip(
            feature_names,
            std / scale,
            strict=True,
        )
        if ratio < min_relative_std
    ]


def feature_shift_report(
    train_records: list[dict[str, Any]],
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    How far an evaluation split sits outside the training distribution.

    For every feature: the largest |z-score| against the training mean/std
    and the fraction of rows outside the training [min, max] range.
    """
    train = records_to_matrix(
        train_records,
        feature_names,
    )
    other = records_to_matrix(
        records,
        feature_names,
    )

    mean = train.mean(axis=0)
    std = np.where(
        train.std(axis=0) > 0,
        train.std(axis=0),
        1.0,
    )

    max_abs_z = np.abs((other - mean) / std).max(axis=0)

    outside = (
        (other < train.min(axis=0))
        | (other > train.max(axis=0))
    ).mean(axis=0)

    return {
        name: {
            "max_abs_z": float(z),
            "fraction_outside_train_range": float(fraction),
        }
        for name, z, fraction in zip(
            feature_names,
            max_abs_z,
            outside,
            strict=True,
        )
    }
