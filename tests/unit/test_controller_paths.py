from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from src.experiments.workloads import controller_paths as cp

ROOT = Path(__file__).resolve().parents[2]


class ControllerPathsTest(unittest.TestCase):
    def test_2c20s_gives_back_the_former_literals(self) -> None:
        paths = cp.load_paths(ROOT / "configs/controllers_2c20s.yaml")
        self.assertEqual(cp.controller_ids(paths), ("c1", "c2"))
        self.assertEqual(cp.sinks(paths), (("h10", "10.0.0.10", 9000),
                                           ("h20", "10.0.0.20", 9000)))
        self.assertEqual(cp.generators(paths), (("h1", "10.0.0.1", "h10", "10.0.0.10"),
                                                ("h11", "10.0.0.11", "h20", "10.0.0.20")))
        self.assertEqual(cp.controller_hosts(paths), {"c1": tuple(range(1, 11)),
                                                      "c2": tuple(range(11, 21))})
        self.assertEqual(cp.hot_hosts(paths), {"c1": 1, "c2": 11})

    def test_4c40s_has_four_slices(self) -> None:
        paths = cp.load_paths(ROOT / "configs/controllers_4c40s.yaml")
        self.assertEqual(cp.controller_ids(paths), ("c1", "c2", "c3", "c4"))
        self.assertEqual(cp.generators(paths)[3], ("h31", "10.0.0.31", "h40", "10.0.0.40"))

    def test_env_selects_the_config_and_default_is_2c(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(cp.ENV, None)
            self.assertEqual(len(cp.load_paths()), 2)
        with mock.patch.dict(os.environ, {cp.ENV: "configs/controllers_4c40s.yaml"}):
            self.assertEqual(len(cp.load_paths()), 4)

    def test_non_contiguous_or_unknown_owner_is_rejected(self) -> None:
        base: dict[str, object] = {"controllers": {"c1": {}, "c2": {}}}
        with self.assertRaisesRegex(ValueError, "contiguous"):
            cp.paths_from_config(dict(base, initial_ownership={"s1": "c1", "s3": "c1",
                                                                "s2": "c2", "s4": "c2"}))
        with self.assertRaisesRegex(ValueError, "unknown"):
            cp.paths_from_config(dict(base, initial_ownership={"s1": "c9", "s2": "c9"}))

    def test_runners_keep_their_2c_tables(self) -> None:
        os.environ.pop(cp.ENV, None)
        from scripts.experiments import run_forecast_hot_switch_2c20s as hot
        from scripts.experiments import run_forecast_stable_2c20s as stable
        self.assertEqual(stable.CONTROLLER_IDS, ("c1", "c2"))
        self.assertEqual(stable.GENERATORS[1], ("h11", "10.0.0.11", "h20", "10.0.0.20"))
        self.assertEqual(hot.HOT_HOST, {"c1": 1, "c2": 11})


if __name__ == "__main__":
    unittest.main()
