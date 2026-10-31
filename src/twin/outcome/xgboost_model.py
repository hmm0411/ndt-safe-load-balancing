"""XGBoost regressor of one outcome target (Direct ML and residual learner).

Saved in XGBoost's own JSON format (portable across versions, unlike a pickle).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from xgboost import XGBRegressor

FAMILY = "xgboost"
SUFFIX = ".json"


def make(params: Mapping[str, Any]) -> XGBRegressor:
    return XGBRegressor(**dict(params))


def save(model: XGBRegressor, path: Path) -> Path:
    path = path.with_suffix(SUFFIX)
    model.save_model(str(path))
    return path


def load(path: Path) -> XGBRegressor:
    model = XGBRegressor()
    model.load_model(str(path))
    return model
