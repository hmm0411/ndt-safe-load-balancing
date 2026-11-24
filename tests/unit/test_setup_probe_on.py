from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.experiments.migration import setup_probe_on as on


class SetupProbeOnTest(unittest.TestCase):
    def test_off_adds_nothing(self) -> None:
        self.assertEqual(on.workload_args(False, Path("/w"), "run", 120.0), [])

    def test_on_uses_the_capacity_v5_sweep_probe(self) -> None:
        args = on.workload_args(True, Path("/w"), "run-1", 120.0)
        pairs = dict(zip(args[0::2], args[1::2]))
        self.assertEqual(pairs["--setup-probe-dir"], "/w/setup_probe")
        self.assertEqual(pairs["--setup-probe-run-id"], "run-1")
        self.assertEqual(pairs["--setup-probe-rate"], "1.0")
        self.assertEqual(pairs["--setup-probe-timeout"], "1.0")
        self.assertEqual(pairs["--setup-probe-port"], "9101")
        self.assertEqual(pairs["--setup-probe-switches"], "all")
        self.assertEqual(pairs["--hold-seconds"], "150.0")

    def test_push_command_outlives_the_post_window(self) -> None:
        cmd = on.push_command("python3", Path("/w/setup_probe"), "http://127.0.0.1:9000", 120.0, 60.0)
        pairs = dict(zip(cmd[2::2], cmd[3::2]))
        self.assertEqual(cmd[:2], ["python3", "scripts/experiments/push_setup_probe.py"])
        self.assertEqual(pairs["--window-seconds"], "10.0")
        self.assertEqual(pairs["--period-seconds"], "1.0")
        self.assertEqual(pairs["--timeout-seconds"], "1.0")
        self.assertEqual(pairs["--duration-seconds"], "300.0")

    def test_written_files_counts_only_probe_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("qos_s1.jsonl", "qos_s2.jsonl", "probe_s1.log", "echo_s1.log"):
                (Path(tmp) / name).write_text("x", encoding="utf-8")
            self.assertEqual(on.written_files(Path(tmp)), 2)

    def test_parameters_are_saved_with_the_attempt(self) -> None:
        self.assertEqual(on.parameters()["port"], 9101)
        self.assertEqual(on.parameters()["push_period_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
