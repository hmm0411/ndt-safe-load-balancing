from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments import derive_controller_capacity as derive

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "data" / "benchmarks" / "controller_capacity.json"
RUNS = ROOT / "data" / "benchmarks" / "capacity_runs.csv"


class DeriveControllerCapacityTests(unittest.TestCase):
    def test_constants_match_the_artifact_method(self) -> None:
        method = json.loads(ARTIFACT.read_text())["method"]
        self.assertEqual(derive.SAFE_REFERENCE_RATE, method["safe_reference_rate_fps"])
        self.assertEqual(derive.THROUGHPUT_KNEE_RATE, method["throughput_knee_rate_fps"])
        self.assertEqual(derive.SETUP_KNEE_RATE, method["setup_knee_rate_fps"])

    def test_regenerated_c_safe_matches_the_artifact(self) -> None:
        artifact = json.loads(ARTIFACT.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "capacity.json"
            with (
                mock.patch.object(derive, "INPUT_PATH", RUNS),
                mock.patch.object(derive, "OUTPUT_PATH", out),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                derive.main()
            regenerated = json.loads(out.read_text())
        self.assertEqual(sorted(regenerated["controllers"]), sorted(artifact["controllers"]))
        for cid, item in artifact["controllers"].items():
            self.assertAlmostEqual(
                regenerated["controllers"][cid]["c_safe_pps"], item["c_safe_pps"], places=9
            )


if __name__ == "__main__":
    unittest.main()
