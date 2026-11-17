import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.experiments.topologies.setup_probe_launcher import (
    launch_setup_probes,
    parse_switches,
)


class FakeProcess:
    def __init__(self, command: list[str]) -> None:
        self.command = command
        self.stopped = False

    def poll(self) -> int | None:
        return 0 if self.stopped else None

    def terminate(self) -> None:
        self.stopped = True

    def wait(self, timeout: float) -> int:
        return 0


class FakeHost:
    def __init__(self, name: str, ip: str) -> None:
        self.name, self.ip = name, ip
        self.started: list[FakeProcess] = []

    def IP(self) -> str:
        return self.ip

    def popen(self, command: list[str], **_: Any) -> FakeProcess:
        process = FakeProcess(command)
        self.started.append(process)
        return process


class FakeNet:
    def __init__(self) -> None:
        self.hosts = {}
        for i in range(1, 41):
            self.hosts[f"h{i}"] = FakeHost(f"h{i}", f"10.0.0.{i}")
            self.hosts[f"p{i}"] = FakeHost(f"p{i}", f"10.0.0.{100 + i}")

    def get(self, name: str) -> FakeHost:
        return self.hosts[name]


class SetupProbeLauncherTest(unittest.TestCase):
    def test_parse_switches(self) -> None:
        self.assertEqual(parse_switches("all", 3), [1, 2, 3])
        self.assertEqual(parse_switches("3,1,3", 40), [1, 3])
        with self.assertRaises(ValueError):
            parse_switches("41", 40)

    def test_probe_goes_from_p_to_h_on_the_same_switch(self) -> None:
        net = FakeNet()
        with tempfile.TemporaryDirectory() as tmp:
            started = launch_setup_probes(
                net, [5], Path(tmp), "r1", 9001, 2.0, 10.0, 1.0, Path(tmp)
            )
            started.stop()
            command = net.hosts["p5"].started[0].command
        self.assertEqual(command[command.index("--source-host") + 1], "p5")
        self.assertEqual(command[command.index("--target-host") + 1], "h5")
        self.assertEqual(command[command.index("--target-ip") + 1], "10.0.0.5")
        self.assertEqual(command[command.index("--source-ip") + 1], "10.0.0.105")

    def test_echo_runs_on_the_target_host_and_stop_ends_everything(self) -> None:
        net = FakeNet()
        with tempfile.TemporaryDirectory() as tmp:
            started = launch_setup_probes(
                net, [2, 3], Path(tmp), "r1", 9001, 2.0, 10.0, 1.0, Path(tmp)
            )
            self.assertEqual(len(net.hosts["h2"].started), 1)
            self.assertIn("qos_echo", " ".join(net.hosts["h3"].started[0].command))
            started.stop()
        self.assertTrue(all(p.stopped for p in started.echoes + started.probes))


if __name__ == "__main__":
    unittest.main()
