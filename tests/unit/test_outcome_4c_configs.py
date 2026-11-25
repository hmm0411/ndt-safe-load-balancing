"""post-migration-v2 (4C40S) configs change only versions and paths, never the v1 rules."""
from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.twin.outcome import predictor
from src.twin.outcome.artifact import load_artifact_config
from src.twin.outcome.dataset import capacity_info, load_dataset_config

ROOT = Path(__file__).resolve().parents[2]
OUTCOME = ROOT / "configs" / "outcome"


def read(name: str) -> dict[str, object]:
    data = yaml.safe_load((OUTCOME / name).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


class Outcome4CConfigTest(unittest.TestCase):
    def test_dataset_rules_equal_v1(self) -> None:
        v1, v2 = read("dataset.yaml"), read("dataset_4c_v2.yaml")
        changed = {k for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}
        self.assertEqual(changed, {"dataset_version", "source_dataset"})
        self.assertEqual(v2["dataset_version"], "migration-outcome-processed-v2")
        load_dataset_config(OUTCOME / "dataset_4c_v2.yaml")

    def test_artifact_rules_equal_v1(self) -> None:
        v1, v2 = read("artifact.yaml"), read("artifact_4c_v2.yaml")
        changed = {k for k in v1.keys() | v2.keys() if v1.get(k) != v2.get(k)}
        self.assertEqual(changed, {"model_version", "artifact_dir"})
        config = load_artifact_config(OUTCOME / "artifact_4c_v2.yaml")
        self.assertIn(config.model_version, predictor.SUPPORTED_MODEL_VERSIONS)
        self.assertEqual(config.artifact_dir.name, config.model_version)

    def test_4c_runtime_points_at_v2_and_the_probe_off_capacity(self) -> None:
        runtime = read("runtime_4c40s.yaml")
        self.assertEqual(Path(str(runtime["artifact_dir"])).name, "post-migration-v2")
        capacity = capacity_info(ROOT / str(runtime["capacity_file"]))
        self.assertEqual(sorted(capacity["c_safe_pps"]), ["c1", "c2", "c3", "c4"])

    def test_2c_runtime_unchanged(self) -> None:
        self.assertEqual(Path(str(read("runtime.yaml")["artifact_dir"])).name,
                         predictor.MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
