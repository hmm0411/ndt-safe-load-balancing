from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.experiments.capacity import sweep_4c40s as sw
from src.experiments.common.config import load_yaml, validate_capacity_config

ROOT = Path(__file__).resolve().parents[2]
CONTROLLERS = load_yaml(ROOT / "configs/controllers_4c40s.yaml")
CONFIGS = {
    name: load_yaml(ROOT / f"configs/experiments/{name}_4c40s.yaml")
    for name in ("probe_off", "probe_sweep", "capacity_coarse", "capacity_fine")
}


class ConfigTest(unittest.TestCase):
    def test_committed_configs_pass_both_validators(self) -> None:
        for name, config in CONFIGS.items():
            with self.subTest(name=name):
                validate_capacity_config(config)
                sw.validate_sweep_config(config, CONTROLLERS)

    def test_plan_sizes_match_the_plan(self) -> None:
        sizes = {
            name: len(sw.plan_runs(config, sw.sweep_controllers(config, CONTROLLERS)))
            for name, config in CONFIGS.items()
        }
        self.assertEqual(sizes, {"probe_off": 5, "probe_sweep": 60,
                                 "capacity_coarse": 56, "capacity_fine": 320})

    def test_day3_background_and_probe_day4_idle(self) -> None:
        self.assertEqual(sw.background_rate(CONFIGS["probe_sweep"]), 30.0)
        self.assertIsNone(sw.setup_probe_spec(CONFIGS["probe_off"]))
        self.assertIsNone(sw.background_rate(CONFIGS["capacity_fine"]))
        probe = sw.setup_probe_spec(CONFIGS["capacity_fine"])
        assert probe is not None
        self.assertEqual(probe["port"], 9101)

    def test_path_not_owned_by_target_is_rejected(self) -> None:
        bad = copy.deepcopy(CONFIGS["probe_sweep"])
        bad["scenarios"]["c2"]["path_switches"][0] = "s1"
        with self.assertRaisesRegex(ValueError, "not owned by c2"):
            sw.validate_sweep_config(bad, CONTROLLERS)

    def test_probe_port_collision_is_rejected(self) -> None:
        bad = copy.deepcopy(CONFIGS["probe_sweep"])
        bad["setup_probe"]["port"] = 9001
        with self.assertRaisesRegex(ValueError, "collides"):
            sw.validate_sweep_config(bad, CONTROLLERS)

    def test_unsorted_rates_and_bad_background_are_rejected(self) -> None:
        bad = copy.deepcopy(CONFIGS["probe_sweep"])
        bad["workload"]["rates_fps"] = [90, 30]
        with self.assertRaises(ValueError):
            sw.validate_sweep_config(bad, CONTROLLERS)
        bad = copy.deepcopy(CONFIGS["probe_sweep"])
        bad["background"]["rate_fps"] = 0
        with self.assertRaises(ValueError):
            sw.validate_sweep_config(bad, CONTROLLERS)

    def test_unknown_sweep_controller_is_rejected(self) -> None:
        bad = dict(CONFIGS["probe_off"], sweep_controllers=["c9"])
        with self.assertRaises(ValueError):
            sw.sweep_controllers(bad, CONTROLLERS)

    def test_run_ids_follow_make_run_id(self) -> None:
        runs = sw.plan_runs(CONFIGS["probe_sweep"], ["c3"], [90])
        self.assertEqual([r.run_id for r in runs],
                         [f"probe-sweep-4c40s-c3-90-r0{i}" for i in range(1, 6)])


class StatusTest(unittest.TestCase):
    def test_four_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(sw.run_status(root / "none"), "missing")
            layouts: dict[str, dict[str, dict[str, object]]] = {
                "valid": {"validation.json": {"valid": True}},
                "failed": {"validation.json": {"valid": False}},
                "crashed": {"failure.json": {"error": "x"}},
                "incomplete": {"metadata.json": {}},
            }
            for name, files in layouts.items():
                (root / name).mkdir()
                for file, body in files.items():
                    (root / name / file).write_text(json.dumps(body), encoding="utf-8")
            self.assertEqual(sw.run_status(root / "valid"), "valid")
            self.assertEqual(sw.run_status(root / "failed"), "failed")
            self.assertEqual(sw.run_status(root / "crashed"), "failed")
            self.assertEqual(sw.run_status(root / "incomplete"), "incomplete")


class ProbeSummaryTest(unittest.TestCase):
    def test_window_owner_and_timeout(self) -> None:
        t0 = datetime(2026, 11, 18, tzinfo=timezone.utc)
        rows: dict[str, list[tuple[int, bool, float | None]]] = {
            "s1": [(1, True, 5.0), (2, True, 7.0), (3, False, None), (100, True, 1.0)],
            "s11": [(1, True, 3.0)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            probe_dir = Path(tmp)
            for switch, values in rows.items():
                with (probe_dir / f"qos_{switch}.jsonl").open("w", encoding="utf-8") as f:
                    for second, ok, latency in values:
                        f.write(json.dumps({
                            "observed_at": (t0 + timedelta(seconds=second)).isoformat(),
                            "success": ok, "flow_setup_latency_ms": latency}) + "\n")
            stats = sw.probe_summary(probe_dir, {"s1": "c1", "s11": "c2", "s21": "c3"},
                                     t0, t0 + timedelta(seconds=60), 1.0)
        # s1: three samples in the window, the timeout counts as 1000 ms (worst value).
        self.assertEqual(stats["c1"], {"setup_p95_ms": 1000.0, "samples": 3, "timeouts": 1})
        self.assertEqual(stats["c2"]["samples"], 1)
        self.assertIsNone(stats["c3"])  # no sample is None, never 0.0


if __name__ == "__main__":
    unittest.main()
