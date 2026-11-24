"""The orchestrator's runtime forecaster must point at the current forecast artifact.

runtime.yaml is the 2C20S runtime, so it is checked against the default profile even when
NDT_FORECAST_PROFILE=4c40s is set in the shell (25/11: make ci failed on the VM that way)."""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.twin.forecasting.artifact import validate_artifact
from src.twin.forecasting.versions import DEFAULT_PROFILE, PROFILES

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_YAML = ROOT / "configs" / "forecasting" / "runtime.yaml"


class RuntimeForecastConfigTest(unittest.TestCase):
    def test_runtime_points_at_the_current_model_version(self) -> None:
        cfg = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        self.assertEqual(Path(cfg["model_dir"]).name, PROFILES[DEFAULT_PROFILE].model_version)

    def test_artifact_of_the_runtime_config_is_valid(self) -> None:
        cfg = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        self.assertEqual(validate_artifact(ROOT / cfg["model_dir"]), [])


if __name__ == "__main__":
    unittest.main()
