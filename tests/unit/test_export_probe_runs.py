from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/experiments/export_probe_runs.py"
SPEC = importlib.util.spec_from_file_location("export_probe_runs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
IDS = ("c1", "c2", "c3", "c4")


def make_run(root: Path, name: str, rate: float, *, probe: bool, valid: bool) -> None:
    run = root / name
    run.mkdir()
    meta = {
        "run_id": name,
        "target_controller": "c1",
        "target_new_flow_rate": rate,
        "measurement_started_at": "2026-11-18T10:00:20Z",
        "measurement_ended_at": "2026-11-18T10:01:20Z",
    }
    (run / "metadata.json").write_text(json.dumps(meta))
    (run / "validation.json").write_text(json.dumps({"valid": valid}))
    (run / "summary.json").write_text(json.dumps({"flow_setup_latency_p95_ms": 110.0}))
    if probe:
        item = {"setup_p95_ms": 13.0, "samples": 590, "timeouts": 2}
        (run / "setup_probe_summary.json").write_text(json.dumps({c: item for c in IDS}))
    stamps = [f"2026-11-18T10:{s // 60:02d}:{s % 60:02d}Z" for s in range(100)]
    with (run / "controllers.jsonl").open("w") as stream:
        for stamp in stamps:
            for cid in IDS:
                row: dict[str, Any] = {"controller_id": cid, "observed_at": stamp}
                row.update({"processed_packet_in_rate": 933.0, "response_p95_ms": 0.8})
                stream.write(json.dumps(row) + "\n")
    owners = [{"owner_controller_id": f"c{i % 4 + 1}"} for i in range(40)]
    with (run / "snapshots.jsonl").open("w") as stream:
        for stamp in stamps:
            snap = {"created_at": stamp, "ownership": owners, "ownership_version": 1}
            snap["quality"] = {"valid": True}
            stream.write(json.dumps(snap) + "\n")


class ExportProbeRunsTest(unittest.TestCase):
    def test_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            runs.mkdir()
            make_run(runs, "probe-sweep-4c40s-c1-90-r01", 90.0, probe=True, valid=True)
            make_run(runs, "probe-off-4c40s-c1-30-r01", 30.0, probe=False, valid=True)
            make_run(runs, "probe-sweep-4c40s-c1-140-r01", 140.0, probe=True, valid=False)
            cap = root / "cap.json"
            cap.write_text(json.dumps({"controllers": {c: {"c_safe_pps": 933.0} for c in IDS}}))
            out, man = root / "o.csv", root / "m.json"
            kept, skipped = MODULE.discover(runs)
            self.assertEqual(len(kept), 2)
            self.assertEqual(skipped, ["probe-sweep-4c40s-c1-140-r01"])
            capacity = MODULE.load_capacity(cap)
            rows = [r for run in kept for r in MODULE.build_rows(run, capacity)]
            MODULE.write_outputs(rows, kept, skipped, cap, out, man)
            with out.open(newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(len(table), 8)
            first = table[0]
            self.assertEqual(first["group"], "probe_off")
            self.assertEqual(first["probe_setup_p95_ms"], "")
            self.assertAlmostEqual(float(first["u_mean"]), 1.0)
            self.assertEqual(first["owned_switches_mean"], "10.0")
            sweep = table[4]
            self.assertEqual(sweep["probe_timeouts"], "2")
            self.assertEqual(sweep["controller_samples"], "60")
            self.assertEqual(sweep["host_flow_setup_p95_ms"], "110.0")
            self.assertEqual(table[5]["host_flow_setup_p95_ms"], "")
            doc = json.loads(man.read_text())
            self.assertEqual(doc["rows"], 8)
            self.assertEqual(len(doc["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
