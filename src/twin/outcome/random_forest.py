"""Random Forest regressor of one outcome target (Direct ML and residual learner)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
from sklearn.ensemble import RandomForestRegressor

FAMILY = "random_forest"
SUFFIX = ".joblib"


def make(params: Mapping[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(**dict(params))


def save(model: RandomForestRegressor, path: Path) -> Path:
    path = path.with_suffix(SUFFIX)
    joblib.dump(model, path)
    return path


def load(path: Path) -> RandomForestRegressor:
    model = joblib.load(path)
    if not isinstance(model, RandomForestRegressor):
        raise ValueError(f"{path} is not a RandomForestRegressor")
    return model
