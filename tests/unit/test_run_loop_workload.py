from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decision import run_loop_workload as rlw
from src.experiments.migration.runner import RunnerSettings, build_workload_command


class LoopWorkloadTest(unittest.TestCase):
    def settings(self) -> RunnerSettings:
        root = Path("/repo")
        return RunnerSettings(root, root / "out", root / "work", root / "cfg.yaml", "cfg.yaml",
                              root / "cap.json", root / "runs.csv")

    def test_destination_level_and_seed_reach_the_workload_command(self) -> None:
        plan = rlw.make_plan("gradual", "run", 0.95, "c1", "c2", rlw.DEFAULT_SEED)
        self.assertEqual(plan.attempt_id, "loop-gradual-run-dst95-c1-c2-s9601")
        cmd = build_workload_command(plan, self.settings(), Path("/w"), plan.attempt_id, 150.0)
        self.assertIn("--forecast-gradual", cmd)
        self.assertIn("c1=gradual", cmd)
        self.assertIn("c2=constant:0.95", cmd)
        self.assertEqual(cmd[cmd.index("--duration") + 1], "150.0")
        self.assertEqual(cmd[cmd.index("--seed") + 1], "9601")

    def test_hot_switch_level_and_load_check_off(self) -> None:
        plan = rlw.make_plan("hot_switch", "peak", 0.45, "c1", "c2", 9601)
        cmd = build_workload_command(plan, self.settings(), Path("/w"), plan.attempt_id, 120.0)
        self.assertIn("NDT_HOT_SWITCH_LEVEL=peak", cmd)
        self.assertIn("NDT_HOT_SWITCH_CHECK_LOAD=0", cmd)

    def test_source_level_replaces_only_the_source_profile(self) -> None:
        plan = rlw.make_plan("stable", "run", 0.45, "c1", "c2", 9601)
        cmd = build_workload_command(plan, self.settings(), Path("/w"), plan.attempt_id, 180.0)
        self.assertEqual(rlw.source_profile(cmd, plan, None), cmd)
        out = rlw.source_profile(cmd, plan, 1.05)
        self.assertIn("c1=constant:1.05", out)
        self.assertNotIn("c1=stable", out)
        self.assertIn("c2=constant:0.45", out)
        gradual = rlw.make_plan("gradual", "run", 0.45, "c1", "c2", 9601)
        with self.assertRaises(ValueError):
            rlw.source_profile(cmd, gradual, 1.05)
        with self.assertRaises(ValueError):
            rlw.source_profile(cmd, plan, 3.0)

    def test_seed_is_not_a_data_seed_and_bad_plans_are_refused(self) -> None:
        used = {101, 102, 103, 104, 105, 9201, 9202, 9203, 9204, 9205, 9301, 9401, 9501}
        self.assertNotIn(rlw.DEFAULT_SEED, used)
        for args in (("hot_switch", "run", 0.45, "c1", "c2"), ("gradual", "run", 2.0, "c1", "c2"),
                     ("gradual", "run", 0.45, "c1", "c1"), ("nope", "run", 0.45, "c1", "c2")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                rlw.make_plan(*args, 9601)


if __name__ == "__main__":
    unittest.main()
