"""M4: the replay used for feature parity works end to end on synthetic runs.

A synthetic raw run (with one invalid snapshot and one 4 s gap) goes through
the dataset pipeline (segment + build_features) and through the runtime
replay; every runtime vector must equal its dataset row. The same functions
run on the real test runs in tests/integration/test_feature_parity.py.
"""
from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.analysis.feature_parity import compare, key_of, replay_run
from scripts.experiments.build_forecast_dataset import MAX_GAP_SECONDS, has_complete_features
from src.twin.forecasting.dataset import assign_sequence_ids
from src.twin.forecasting.features import build_features, feature_names

MODEL = Path("data/models/forecasting/forecast-v1")
T0 = datetime(2026, 10, 1, tzinfo=timezone.utc)


def synthetic_run(run_id: str = "burst-r05-s105", seconds: int = 60) -> list[dict[str, Any]]:
    samples = []
    for t in range(seconds):
        if 45 <= t < 49:  # 4 s gap: both pipelines must restart history
            continue
        for k, cid in enumerate(("c1", "c2")):
            u = 0.5 + 0.3 * math.sin(t / 5 + k) + (0.4 if 20 <= t < 26 else 0.0)
            samples.append({
                "run_id": run_id, "controller_id": cid,
                "observed_at": (T0 + timedelta(seconds=t, milliseconds=3 * k)).isoformat(),
                "workload_type": "burst", "workload_phase": "p",
                "processed_packet_in_rate": u * 1433.0, "flow_mod_rate": 10 + t % 7,
                "process_cpu_percent": 30 + t % 11, "process_memory_rss_mb": 120.0 + t / 10,
                "response_p95_ms": 2 + (t % 5) / 10, "managed_switch_count": 10,
                "safe_capacity_pps": 1433.0, "utilization": u,
                "max_switch_control_load_share": 0.1 + 0.001 * (t % 3),
                "snapshot_valid": t != 30, "snapshot_id": f"snap-{t}",
            })
    return samples


@unittest.skipUnless((MODEL / "model_card.json").is_file(), "forecast-v1 not present")
class FeatureParityReplayTest(unittest.TestCase):
    def test_runtime_vectors_equal_dataset_rows(self) -> None:
        raw = synthetic_run()
        # Same observed_at for both controllers of one instant (as recorded).
        for s in raw:
            s["observed_at"] = s["observed_at"][:19] + "+00:00"
        valid = [r for r in raw if r["snapshot_valid"]]
        featured = build_features(assign_sequence_ids(valid, max_gap_seconds=MAX_GAP_SECONDS))
        expected = {
            key_of(r["run_id"], r["controller_id"], r["observed_at"]): r
            for r in featured if has_complete_features(r)
        }
        result = compare(expected, replay_run(raw, MODEL), list(feature_names()))
        self.assertEqual(result["mismatched_rows"], 0, result["first_mismatches"])
        self.assertEqual(result["compared_rows"], result["expected_rows"])
        self.assertGreater(result["expected_rows"], 40)


if __name__ == "__main__":
    unittest.main()
