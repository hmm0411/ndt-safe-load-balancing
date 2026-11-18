from __future__ import annotations

import csv
import inspect
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.workloads import forecast_4c as f4c

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["run_id", "controller", "valid", "target_new_flow_rate",
          "processed_packet_in_rate_mean"]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def row(run_id: str, controller: str, valid: str = "true") -> dict[str, str]:
    return {"run_id": run_id, "controller": controller, "valid": valid,
            "target_new_flow_rate": "90", "processed_packet_in_rate_mean": "900"}


class Forecast4CTest(unittest.TestCase):
    def test_plan_from_the_committed_config(self) -> None:
        config = yaml.safe_load((ROOT / "configs/experiments/transfer_4c_v1.yaml")
                                .read_text(encoding="utf-8"))
        plan = f4c.transfer_plan(config)
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan[0].run_id, "transfer4c-stable-r01-s9701")
        self.assertEqual(plan[-1].run_id, "transfer4c-hot-switch-r02-s9702")
        self.assertEqual(len({r.run_id for r in plan}), 10)

    def test_only_valid_fine_rows_are_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "all.csv", Path(tmp) / "used.csv"
            rows = [row(f"capacity-4c40s-fine-{c}-90fps-r01", c) for c in ("c1", "c2", "c3",
                                                                          "c4")]
            rows += [row("probe-sweep-4c40s-c1-90fps-r01", "c1"),
                     row("capacity-4c40s-fine-c1-100fps-r01", "c1", valid="false")]
            write_rows(src, rows)
            self.assertEqual(f4c.filter_capacity_rows(src, dst), 4)
            with dst.open(encoding="utf-8") as handle:
                kept = list(csv.DictReader(handle))
        self.assertTrue(all(r["run_id"].startswith("capacity-4c40s-fine-") for r in kept))

    def test_a_controller_without_rows_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "all.csv"
            write_rows(src, [row("capacity-4c40s-fine-c1-90fps-r01", "c1")])
            with self.assertRaises(ValueError):
                f4c.filter_capacity_rows(src, Path(tmp) / "used.csv")

    def test_kwargs_match_every_runner(self) -> None:
        for workload in f4c.RUNNERS:
            kwargs = f4c.runner_kwargs(
                workload, capacity_runs=Path("a"), controller_capacity=Path("b"),
                output_dir=Path("c"), duration=1.0, sample_interval=1.0, orchestrator_url="u",
                run_id="r", seed=1, repeat_index=1, experiment_type="e", git_commit="g",
                profiles=None, target_utilization=0.6)
            inspect.signature(f4c.load_runner(workload)).bind(None, **kwargs)

    def test_hot_switch_tables_follow_the_4c_config(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(ROOT),
               **f4c.runner_env(ROOT / "configs/controllers_4c40s.yaml")}
        code = ("from scripts.experiments import run_forecast_hot_switch_2c20s as h; "
                "print(h.SINK_PREFIX, h.ALL_HOSTS[0], h.ALL_HOSTS[-1], sorted(h.HOT_HOST.items()))")
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "p 1 40 [('c1', 1), ('c2', 11), ('c3', 21), ('c4', 31)]")

    def test_hot_switch_2c_default_is_unchanged(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k not in ("NDT_CONTROLLERS_CONFIG", "NDT_LOCAL_SINK_PREFIX")}
        env["PYTHONPATH"] = str(ROOT)
        code = ("from scripts.experiments import run_forecast_hot_switch_2c20s as h; "
                "print(h.SINK_PREFIX, list(h.ALL_HOSTS) == list(range(1, 21)))")
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "hs True")


if __name__ == "__main__":
    unittest.main()
