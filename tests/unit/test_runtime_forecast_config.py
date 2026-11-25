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
RUNTIME_4C_YAML = ROOT / "configs" / "forecasting" / "runtime_4c40s.yaml"


class RuntimeForecastConfigTest(unittest.TestCase):
    def test_runtime_points_at_the_current_model_version(self) -> None:
        cfg = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        self.assertEqual(Path(cfg["model_dir"]).name, PROFILES[DEFAULT_PROFILE].model_version)

    def test_artifact_of_the_runtime_config_is_valid(self) -> None:
        cfg = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        self.assertEqual(validate_artifact(ROOT / cfg["model_dir"]), [])

    def test_4c_runtime_differs_only_by_the_v4_model(self) -> None:
        cfg2 = yaml.safe_load(RUNTIME_YAML.read_text(encoding="utf-8"))
        cfg4 = yaml.safe_load(RUNTIME_4C_YAML.read_text(encoding="utf-8"))
        self.assertEqual({k for k in cfg2.keys() | cfg4.keys() if cfg2.get(k) != cfg4.get(k)},
                         {"model_dir"})
        self.assertEqual(Path(cfg4["model_dir"]).name, PROFILES["4c40s"].model_version)
        self.assertEqual(validate_artifact(ROOT / cfg4["model_dir"]), [])


if __name__ == "__main__":
    unittest.main()
