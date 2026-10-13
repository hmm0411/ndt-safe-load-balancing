"""Single source of truth for forecasting dataset and model versions.

v1 = week-9 baselines on the 4-vCPU testbed (controller log bug present,
     run ids forecast-<workload>-2c20s-rNN).
v2 = data collected on e2-standard-8 after the controller fix
     (capacity-v3, forecast-raw-v2, run ids <workload>-rNN-sSEED).
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
FORECAST_CONFIG = _REPO_ROOT / "configs" / "forecasting" / "forecast.yaml"

DATASET_GENERATION = "v2"
MODEL_VERSION = "forecast-models-v2"


def provisional_horizon(config_path: Path = FORECAST_CONFIG) -> tuple[int, float]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    provisional = config["horizon"]["provisional"]
    return int(provisional["horizon_steps"]), float(
        provisional["effective_horizon_seconds"]
    )


def dataset_version(config_path: Path = FORECAST_CONFIG) -> str:
    steps, _ = provisional_horizon(config_path)
    return f"forecast-dataset-{DATASET_GENERATION}-h{steps}-provisional"


PROCESSED_ROOT = Path("data/forecasting/processed")
MODELS_ROOT = Path("data/models/forecasting")
