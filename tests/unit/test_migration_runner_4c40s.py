"""The migration runner on 4C40S: what changes with the topology, and that 2C20S does not."""
from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from src.experiments.migration import runner
from src.experiments.migration.config import AttemptPlan


def make_plan(**kw: Any) -> AttemptPlan:
    values: dict[str, Any] = dict(
        attempt_id="mig-burst-burst_1-dst75-c1-c2-r01", action_type="MIGRATE", workload="burst",
        phase="burst_1", destination_load_level=0.75, source_controller="c1",
        target_controller="c2", repeat="r01", seed=9801, split="train")
    values.update(kw)
    return AttemptPlan(**values)


def settings(topology: str) -> runner.RunnerSettings:
    root = Path("/repo")
    return runner.RunnerSettings(
        repo_root=root, output_root=root / "out", work_root=root / "work",
        config_path=root / "c.yaml", controllers_config="configs/controllers_4c40s.yaml",
        capacity_json=root / "cap.json", capacity_runs=root / "runs.csv", python="python3",
        topology=topology)


class TopologySpecTest(unittest.TestCase):
    def test_sizes(self) -> None:
        two, four = runner.topology_spec("2C20S"), runner.topology_spec("4C40S")
        self.assertEqual((two.switches, two.openflow_ports, two.controllers),
                         (20, (6653, 6654), ("c1", "c2")))
        self.assertEqual((four.switches, four.openflow_ports, four.controllers),
                         (40, (6653, 6654, 6655, 6656), ("c1", "c2", "c3", "c4")))

    def test_unknown_topology(self) -> None:
        with self.assertRaises(runner.RunnerError):
            runner.topology_spec("3C30S")

    def test_2c_constants_unchanged(self) -> None:
        self.assertEqual(runner.EXPECTED_SWITCHES, 20)
        self.assertEqual(runner.OPENFLOW_PORTS, (6653, 6654))


class WorkloadCommand4CTest(unittest.TestCase):
    def test_every_other_controller_is_held_at_the_destination_level(self) -> None:
        cmd = runner.build_workload_command(make_plan(), settings("4C40S"), Path("/w"), "rid", 120.0)
        self.assertIn("src/experiments/topologies/runtime_4c40s.py", cmd)
        self.assertNotIn("src/experiments/topologies/runtime_2c20s.py", cmd)
        self.assertEqual(cmd[cmd.index("--forecast") + 1], "burst")
        self.assertEqual(cmd[cmd.index("--controllers-config") + 1], "configs/controllers_4c40s.yaml")
        profiles = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--profile"]
        self.assertEqual(profiles, ["c1=burst", "c2=constant:0.75", "c3=constant:0.75",
                                    "c4=constant:0.75"])
        self.assertEqual(cmd[cmd.index("--seed") + 1], "9801")
        self.assertEqual(cmd[cmd.index("--duration") + 1], "120.0")

    def test_source_other_than_c1(self) -> None:
        plan = make_plan(source_controller="c3", target_controller="c1", workload="hot_switch")
        cmd = runner.build_workload_command(plan, settings("4C40S"), Path("/w"), "rid", 120.0)
        profiles = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--profile"]
        self.assertEqual(profiles, ["c3=hot-switch", "c1=constant:0.75", "c2=constant:0.75",
                                    "c4=constant:0.75"])
        self.assertEqual(cmd[cmd.index("--forecast") + 1], "hot_switch")

    def test_2c_command_is_unchanged(self) -> None:
        cmd = runner.build_workload_command(make_plan(workload="stable"), settings("2C20S"),
                                            Path("/w"), "rid", 120.0)
        self.assertIn("src/experiments/topologies/runtime_2c20s.py", cmd)
        self.assertIn("--forecast-stable", cmd)
        self.assertIn("c1=stable", cmd)
        self.assertIn("c2=constant:0.75", cmd)
        self.assertNotIn("c3=constant:0.75", cmd)


class Preflight4CTest(unittest.TestCase):
    def test_needs_four_openflow_ports(self) -> None:
        with self.assertRaises(runner.RunnerError) as ctx:
            runner.preflight(settings("4C40S"), port_open=lambda p: p in (6653, 6654),
                             list_bridges=lambda: "")
        self.assertIn("6655", str(ctx.exception))
        self.assertIn("6656", str(ctx.exception))

    def test_2c_ports_still_enough_for_2c(self) -> None:
        cap = Path("/tmp/_cap_4c_test.json")
        cap.write_text("{}")
        try:
            s = runner.RunnerSettings(
                repo_root=Path("/repo"), output_root=Path("/o"), work_root=Path("/w"),
                config_path=Path("/c.yaml"), controllers_config="c.yaml", capacity_json=cap,
                capacity_runs=Path("/r.csv"), topology="2C20S")
            runner.preflight(s, port_open=lambda p: p in (6653, 6654), list_bridges=lambda: "")
        finally:
            cap.unlink()


class ReadyAndPreconditions4CTest(unittest.TestCase):
    @staticmethod
    def raw(n: int) -> dict[str, Any]:
        owners = ["c1", "c2", "c3", "c4"]
        return {"quality": {"valid": True},
                "controllers": [{"controller_id": c, "managed_switch_count": n // 4} for c in owners],
                "ownership": [{"switch_id": f"s{i + 1}", "owner_controller_id": owners[i * 4 // n],
                               "source_role": "MASTER", "target_role": "SLAVE"} for i in range(n)]}

    def test_ready_needs_the_topology_switch_count(self) -> None:
        raw = self.raw(40)
        self.assertTrue(runner.is_ready_state(object(), raw, 40))
        self.assertFalse(runner.is_ready_state(object(), raw))  # 2C default: 20 expected

    def test_preconditions_use_expected_switches(self) -> None:
        raw = self.raw(40)
        ok = runner.check_preconditions(raw, make_plan(), "s1", workload_alive=True,
                                        capacity_exists=True, expected_switches=40)
        self.assertEqual(ok, [])
        bad = runner.check_preconditions(raw, make_plan(), "s1", workload_alive=True,
                                         capacity_exists=True)
        self.assertTrue(any(r.startswith("switches_not_connected") for r in bad))


class HotSwitch4CTest(unittest.TestCase):
    def test_pick_switch_uses_the_given_hot_lookup(self) -> None:
        plan = next(p for p in (make_plan(workload="hot_switch", repeat=f"r{n:02d}",
                                          phase="peak") for n in range(1, 6))
                    if runner.is_target_hot(p))
        ownership = [{"switch_id": f"s{i}", "owner_controller_id": "c1"} for i in range(1, 11)]
        switch, rule, hot = runner.pick_switch(plan, ownership, lambda source: "s7")
        self.assertEqual((switch, rule, hot), ("s7", "hot_switch_alternating", True))


if __name__ == "__main__":
    unittest.main()
