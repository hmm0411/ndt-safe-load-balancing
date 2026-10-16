"""M4 tier 1: the runtime rows match what the data collection recorded.

ForecastDataRunner._samples_from_twin_state produced every training sample;
forecast_rows() feeds the runtime predictor. Both start from the same
TwinState, so every field build_features reads must come out identical.
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from src.experiments.runners.forecast_data_runner import ForecastDataRunner
from src.twin.forecasting.features import CURRENT_FEATURES
from src.twin.forecasting.runtime_rows import forecast_rows


def controller(cid: str, u: float) -> dict[str, Any]:
    return {"controller_id": cid, "utilization": u, "processed_packet_in_rate": u * 1433.0,
            "safe_capacity_pps": 1433.0, "collection_latency_ms": 40.0, "flow_mod_rate": 5.0,
            "process_cpu_percent": 30.0, "process_memory_rss_mb": 120.0,
            "response_p95_ms": 3.0, "managed_switch_count": 2}


def twin_state() -> dict[str, Any]:
    return {
        "snapshot_id": "s1", "created_at": "2026-10-17T00:00:00+00:00",
        "quality": {"valid": True, "age_of_twin_ms": 100.0, "twinning_rate": 1.0,
                    "completeness_ratio": 1.0, "synchronization_jitter_ms": 5.0},
        "controllers": [controller("c1", 0.7), controller("c2", 0.5)],
        "switches": [  # c2 also sees s1 (SLAVE) but does not own it
            {"switch_id": "s1", "controller_id": "c1", "control_load_share": 0.6},
            {"switch_id": "s2", "controller_id": "c1", "control_load_share": 0.4},
            {"switch_id": "s1", "controller_id": "c2", "control_load_share": 0.9},
            {"switch_id": "s3", "controller_id": "c2", "control_load_share": 0.3},
            {"switch_id": "s4", "controller_id": "c2", "control_load_share": 0.7},
        ],
        "ownership": [{"switch_id": s, "owner_controller_id": o}
                      for s, o in (("s1", "c1"), ("s2", "c1"), ("s3", "c2"), ("s4", "c2"))],
    }


def offline_samples(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fake_runner = SimpleNamespace(
        config=SimpleNamespace(controller_ids=("c1", "c2"), workload_type="stable")
    )
    samples = ForecastDataRunner._samples_from_twin_state(
        fake_runner,  # type: ignore[arg-type]
        run_id="stable-r01-s101",
        observed_at=datetime(2026, 10, 17, tzinfo=timezone.utc),
        twin_state=state,
        workload_point=SimpleNamespace(phase="steady"),  # type: ignore[arg-type]
    )
    return {s.controller_id: s.to_dict() for s in samples}


class RuntimeRowsParityTest(unittest.TestCase):
    def test_every_feature_input_matches_the_data_collection(self) -> None:
        state = twin_state()
        offline = offline_samples(state)
        runtime = {r["controller_id"]: r for r in forecast_rows(state)}
        self.assertEqual(set(runtime), set(offline))
        for cid in runtime:
            for field in CURRENT_FEATURES:
                self.assertAlmostEqual(float(runtime[cid][field]), float(offline[cid][field]),
                                       msg=f"{cid}.{field}")

    def test_share_counts_owned_switches_only(self) -> None:
        runtime = {r["controller_id"]: r for r in forecast_rows(twin_state())}
        self.assertAlmostEqual(runtime["c1"]["max_switch_control_load_share"], 0.6)
        self.assertAlmostEqual(runtime["c2"]["max_switch_control_load_share"], 0.7)  # not 0.9

    def test_controller_without_owned_switch_has_zero_share(self) -> None:
        state = twin_state()
        state["ownership"] = [o for o in state["ownership"] if o["owner_controller_id"] == "c1"]
        runtime = {r["controller_id"]: r for r in forecast_rows(state)}
        offline = offline_samples(state)
        self.assertEqual(runtime["c2"]["max_switch_control_load_share"], 0.0)
        self.assertEqual(offline["c2"]["max_switch_control_load_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
