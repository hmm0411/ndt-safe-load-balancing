"""Single source of truth for forecasting dataset and model versions.

v1 = week-9 baselines on the 4-vCPU testbed (controller log bug present,
     run ids forecast-<workload>-2c20s-rNN).
v2 = data collected on e2-standard-8 after the controller fix
     (capacity-v3, forecast-raw-v2, run ids <workload>-rNN-sSEED).
v3 = re-collection on the current VM with capacity-v4 (c_safe 933 pps,
     forecast-raw-v3); same plan, seeds and run-level split as v2.
v4 = 4C40S retrain (24/11, after the 2C -> 4C transfer test failed): 25 runs forecast4c-*,
     seeds 9921-9925, probe-off capacity v5_noprobe, same features, horizon, split and
     hyperparameters (configs/forecasting/models.yaml) as v3. Used by 4C40S only; 2C20S keeps v3.

The profile is chosen by NDT_FORECAST_PROFILE (default 2c20s), read once at import, so every
pipeline script (build dataset/splits, train, margin, export) follows the same profile:
    NDT_FORECAST_PROFILE=4c40s python3 scripts/experiments/build_forecast_dataset.py ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
FORECAST_CONFIG = _REPO_ROOT / "configs" / "forecasting" / "forecast.yaml"



@dataclass(frozen=True)
class ForecastProfile:
    name: str
    dataset_generation: str
    model_version: str
    experiment_config: Path   # official run plan (splitter.planned_run_ids)


PROFILES: dict[str, ForecastProfile] = {
    "2c20s": ForecastProfile(
        "2c20s", "v3", "forecast-models-v3",
        _REPO_ROOT / "configs" / "experiments" / "forecast_dataset_2c20s.yaml"),
    "4c40s": ForecastProfile(
        "4c40s", "v4", "forecast-models-v4",
        _REPO_ROOT / "configs" / "experiments" / "forecast_collection_4c_v1.yaml"),
}
DEFAULT_PROFILE = "2c20s"


def active_profile(environ: dict[str, str] | None = None) -> ForecastProfile:
    name = (os.environ if environ is None else environ).get(
        "NDT_FORECAST_PROFILE", DEFAULT_PROFILE)
    if name not in PROFILES:
        raise ValueError(f"NDT_FORECAST_PROFILE={name!r}; one of {sorted(PROFILES)}")
    return PROFILES[name]


PROFILE = active_profile()
DATASET_GENERATION = PROFILE.dataset_generation
MODEL_VERSION = PROFILE.model_version
EXPERIMENT_CONFIG = PROFILE.experiment_config


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
