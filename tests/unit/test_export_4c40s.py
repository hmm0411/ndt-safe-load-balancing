from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.experiments.capacity import export_4c40s as ex

T0 = datetime(2026, 11, 18, 10, tzinfo=timezone.utc)


def write(path: Path, body: Any) -> None:
    path.write_text(json.dumps(body), encoding="utf-8")


def make_run(root: Path, name: str, rate: float, pps: dict[str, float], valid: bool = True,
             probe: bool = True) -> None:
    run = root / name
    run.mkdir()
    write(run / "metadata.json", {
        "target_controller": name.split("-")[-3], "repeat_index": int(name[-2:]),
        "target_new_flow_rate": rate, "git_commit": "abc",
        "measurement_started_at": T0.isoformat(),
        "measurement_ended_at": (T0 + timedelta(seconds=60)).isoformat()})
    write(run / "summary.json", {"processed_packet_in_rate_mean": pps["c1"],
                                 "qos_success_ratio": 1.0, "flow_setup_latency_p95_ms": 20.0})
    write(run / "validation.json", {"valid": valid, "errors": [], "warnings": []})
    write(run / "run_conditions.json", {"cpu_model": "AMD EPYC 7B12",
                                        "background_rate_fps": 30.0,
                                        "setup_probe": {"port": 9101} if probe else None})
    lines = []
    for second in (5, 30, 90):  # 90 s is outside the measurement window
        for cid, value in pps.items():
            lines.append(json.dumps({"controller_id": cid, "processed_packet_in_rate":
                                     value if second < 60 else 9999.0,
                                     "observed_at": (T0 + timedelta(seconds=second)).isoformat()}))
    (run / "controllers.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if probe:
        write(run / "setup_probe_summary.json", {
            cid: {"setup_p95_ms": 10.0, "samples": 600, "timeouts": 1} for cid in pps})


class ExportTest(unittest.TestCase):
    def test_rows_and_probe_load(self) -> None:
        on = {"c1": 350.0, "c2": 340.0, "c3": 345.0, "c4": 341.0}
        off = {"c1": 340.0, "c2": 330.0, "c3": 335.0, "c4": 331.0}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_run(root, "probe-sweep-4c40s-c1-30-r01", 30, on)
            make_run(root, "probe-sweep-4c40s-c1-90-r01", 90, dict(on, c1=950.0))
            make_run(root, "probe-sweep-4c40s-c2-90-r01", 90, on, valid=False)
            make_run(root, "probe-off-4c40s-c1-30-r01", 30, off, probe=False)
            (root / "capacity-2c20s-fine-c1-30-r01").mkdir()  # not a 4C40S sweep
            rows = ex.collect(root)
        self.assertEqual(len(rows), 4)
        first = next(r for r in rows if r["run_id"] == "probe-sweep-4c40s-c1-30-r01")
        self.assertEqual(first["processed_pps_c2"], 340.0)  # the 90 s sample is left out
        self.assertEqual(first["setup_samples_c4"], 600)
        summary = ex.probe_sweep_summary(rows, 1000.0)
        self.assertEqual(summary["valid_runs"], {"probe_sweep": 2, "probe_off": 1})
        self.assertAlmostEqual(summary["probe_load"]["c1"]["delta_pps"], 10.0)
        self.assertAlmostEqual(summary["probe_load"]["c1"]["delta_u"], 0.01)
        self.assertEqual(summary["curves"]["c1"]["90"]["c1"]["processed_pps_mean"], 950.0)
        self.assertNotIn("c2", summary["curves"])  # its only run is invalid

    def test_unfinished_run_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "capacity-4c40s-fine-c1-30-r01"
            run.mkdir()
            write(run / "metadata.json", {})
            self.assertIsNone(ex.run_row(run))


if __name__ == "__main__":
    unittest.main()
