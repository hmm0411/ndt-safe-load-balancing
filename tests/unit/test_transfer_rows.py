from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.experiments import build_transfer_forecast_rows as cli
from src.experiments.final_evaluation import transfer_rows as tr

ROOT = Path(__file__).resolve().parents[2]
NAMES = json.loads((ROOT / "data/models/forecasting/forecast-models-v3/feature_schema.json")
                   .read_text(encoding="utf-8"))["feature_names"]
T0 = datetime(2026, 11, 21, tzinfo=timezone.utc)


def sample(run_id: str, cid: str, k: int, u: float, valid: bool = True) -> dict[str, Any]:
    return {"run_id": run_id, "controller_id": cid, "workload_type": "stable",
            "observed_at": (T0 + timedelta(seconds=k)).isoformat().replace("+00:00", "Z"),
            "snapshot_valid": valid, "utilization": u,
            "processed_packet_in_rate": u * 842.0, "flow_mod_rate": u * 400.0,
            "process_cpu_percent": 40.0 * u, "process_memory_rss_mb": 120.0,
            "response_p95_ms": 2.0, "managed_switch_count": 10,
            "max_switch_control_load_share": 0.1}


def write_run(root: Path, run_id: str, n: int, capacity: str) -> None:
    d = root / run_id
    d.mkdir(parents=True)
    with (d / "forecast_samples.jsonl").open("w", encoding="utf-8") as handle:
        for cid in ("c1", "c2", "c3", "c4"):
            for k in range(n):
                handle.write(json.dumps(sample(run_id, cid, k, 0.5 + 0.01 * (k % 7))) + "\n")
    (d / "metadata.json").write_text(json.dumps({"capacity_artifact": capacity}))


class ConstantDelta:
    def predict(self, x: Any) -> list[float]:
        return [0.05] * len(x)


class TransferRowsTest(unittest.TestCase):
    def test_rows_need_10_s_history_and_a_future_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "transfer4c-stable-r01-s9701", 30, "x/controller_capacity_v6.json")
            records = tr.load_transfer_samples(root)
            rows = tr.feature_rows(records, NAMES, horizon_steps=3)
        # 30 samples: the first 10 lack a full history, the last 3 lack U(t+3)
        self.assertEqual(len(rows), 4 * (30 - 10 - 3))
        out = tr.predict_rows(rows, NAMES, ConstantDelta(), None, "delta")
        self.assertAlmostEqual(out[0]["prediction"], out[0]["utilization"] + 0.05)
        self.assertEqual({r["controller_id"] for r in out}, {"c1", "c2", "c3", "c4"})

    def test_invalid_snapshots_are_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "transfer4c-burst-r01-s9701"
            d.mkdir()
            (d / "forecast_samples.jsonl").write_text(
                json.dumps(sample(d.name, "c1", 0, 0.5, valid=False)) + "\n")
            self.assertEqual(tr.load_transfer_samples(root), [])

    def test_capacity_check_and_frozen_model_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "transfer4c-stable-r01-s9701", 25, "data/benchmarks/v6.json")
            self.assertTrue(cli.check_capacity(root, Path("v5.json")))
            out = root / "rows.jsonl"
            self.assertEqual(cli.main(["--runs", str(root), "--capacity", "v6.json",
                                       "--out", str(out)]), 0)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4 * (25 - 10 - 3))
        self.assertTrue(all(abs(r["prediction"] - r["utilization"]) < 1.0 for r in rows))


if __name__ == "__main__":
    unittest.main()
