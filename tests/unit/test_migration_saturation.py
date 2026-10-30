"""Saturation check (Week 12): CPU p95 >= 95 % per window, CPU by U."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.experiments.migration import fixtures as fx
from src.experiments.migration import saturation as sat
from src.experiments.migration.writer import write_attempt


def snapshot(cpu: dict[str, float], u: dict[str, float]) -> dict[str, Any]:
    return {"snapshot_id": "s", "controllers": [
        {"controller_id": cid, "process_cpu_percent": cpu[cid], "utilization": u[cid],
         "processed_packet_in_rate": u[cid] * 933.0, "safe_capacity_pps": 933.0}
        for cid in cpu]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")


class SaturationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "raw"
        healthy = [snapshot({"c1": 60.0, "c2": 50.0}, {"c1": 0.7, "c2": 0.5})] * 5
        hot = [snapshot({"c1": 99.0, "c2": 50.0}, {"c1": 1.05, "c2": 0.5})] * 5
        for name, before, after in (("mig-a-r01", healthy, hot), ("mig-b-r01", hot, healthy),
                                    ("noop-c-r01", healthy, healthy)):
            write_jsonl(self.root / name / "before_samples.jsonl", before)
            write_jsonl(self.root / name / "transition_samples.jsonl", healthy[:2])
            write_jsonl(self.root / name / "after_samples.jsonl", after)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_percentile_matches_linear_interpolation(self) -> None:
        self.assertEqual(sat.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95), 4.8)
        self.assertEqual(sat.percentile([7.0], 95), 7.0)
        with self.assertRaises(ValueError):
            sat.percentile([], 95)

    def test_classes(self) -> None:
        rows = {r["run_id"]: r for r in sat.attempt_saturation(self.root)}
        self.assertEqual(rows["mig-a-r01"]["class"], "sat_after_only")
        self.assertEqual(rows["mig-b-r01"]["class"], "sat_before_only")
        self.assertEqual(rows["noop-c-r01"]["class"], "healthy_both")
        self.assertEqual(rows["noop-c-r01"]["kind"], "NO_OP")
        self.assertEqual(rows["mig-a-r01"]["cpu_p95_after_max"], 99.0)

    def test_cpu_by_u_and_per_attempt(self) -> None:
        n, samples, keys = sat.cpu_samples(self.root)
        self.assertEqual(n, 3)
        self.assertEqual(len(samples), 3 * (5 + 2 + 5) * 2)
        self.assertIn("process_cpu_percent", keys)
        per = {(r["run_id"], r["controller"]): r for r in sat.per_attempt_cpu(samples)}
        self.assertEqual(per[("mig-a-r01", "c1")]["u_max"], 1.05)
        self.assertEqual(per[("noop-c-r01", "c1")]["cpu_max"], 60.0)
        lines = sat.cpu_vs_u_report(n, samples, keys, list(per.values()))
        self.assertIn("1.00-9.00      10      99.0      99.0     100.0%", lines)
        self.assertIn("with u_max >= 0.72: 2; of those CPU p95 >= 95%: 2", lines)

    def test_cli_writes_relative_paths(self) -> None:
        out = Path(self.tmp.name) / "out"
        result = subprocess.run(
            [sys.executable, "scripts/analysis/migration_saturation.py", "--raw-root",
             str(self.root), "--out", str(out)], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (out / "attempt_saturation.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["sat_after"] for r in rows], ["True", "False", "False"])
        report = (out / "cpu_vs_u_v2.txt").read_text()
        self.assertTrue(report.splitlines()[-1].startswith(f"saved: {out}"))

    def test_runner_attempt_format(self) -> None:
        root = Path(self.tmp.name) / "fixture"
        write_attempt(root, **fx.attempt_inputs("committed"))
        try:
            rows = sat.attempt_saturation(root)
            self.assertEqual([r["class"] for r in rows], ["healthy_both"])
            self.assertEqual(rows[0]["cpu_p95_before_max"], 60.0)
        finally:
            fx.unlock_tree(root)

    def test_missing_root_is_an_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            sat.attempt_saturation(Path(self.tmp.name) / "nope")


if __name__ == "__main__":
    unittest.main()
