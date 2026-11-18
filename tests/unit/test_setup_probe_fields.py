from __future__ import annotations

import unittest
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any

from src.schemas.snapshot import NetworkSnapshot, SnapshotQuality
from src.schemas.telemetry import ControllerTelemetry
from src.schemas.twin import ControllerTwinState, TwinQuality
from src.twin.capacity.model import CapacityModel
from src.twin.state.builder import TwinStateBuilder

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
SETUP = ("setup_p95_ms", "setup_samples", "setup_timeouts")


def telemetry(**extra) -> ControllerTelemetry:
    return ControllerTelemetry(
        controller_id="c1", observed_at=NOW, ingested_at=NOW,
        packet_in_total=0, packet_in_rate=0.0,
        processed_packet_in_total=0, processed_packet_in_rate=100.0,
        flow_mod_total=0, flow_mod_rate=0.0,
        process_cpu_percent=0.0, process_memory_rss_mb=0.0,
        response_mean_ms=0.0, response_p95_ms=0.0,
        managed_switch_count=10, **extra,
    )


def build(sample: ControllerTelemetry) -> ControllerTwinState:
    quality = TwinQuality(
        age_of_twin_ms=1.0, twinning_rate=1.0, completeness_ratio=1.0,
        synchronization_jitter_ms=1.0, consistent=True, valid=True,
    )
    snapshot = NetworkSnapshot(
        snapshot_id="s", created_at=NOW, topology_version=1, ownership_version=1,
        controllers=[sample], switches=[], ownership=[],
        quality=SnapshotQuality(fresh=True, complete=True, consistent=True, valid=True),
    )
    builder = TwinStateBuilder(CapacityModel({"c1": 1000.0}), lambda _s: quality)
    return builder.build(snapshot).controllers[0]


class SetupProbeFieldsTests(unittest.TestCase):
    def test_probe_off_gives_none_everywhere(self):
        state = build(telemetry())
        for name in SETUP:
            self.assertIsNone(getattr(telemetry(), name))
            self.assertIsNone(getattr(state, name))

    def test_probe_values_reach_the_twin_state(self):
        state = build(telemetry(setup_p95_ms=42.5, setup_samples=100, setup_timeouts=3))
        self.assertEqual(
            (state.setup_p95_ms, state.setup_samples, state.setup_timeouts),
            (42.5, 100, 3),
        )

    def test_artifact_without_the_fields_still_loads(self):
        old: dict[str, Any] = {
            f.name: 1.0 for f in fields(ControllerTwinState) if f.name not in SETUP
        }
        old["controller_id"] = "c1"
        old["managed_switch_count"] = 10
        state = ControllerTwinState(**old)
        self.assertIsNone(state.setup_p95_ms)
        for name in SETUP:
            self.assertIsNone(state.to_dict()[name])


if __name__ == "__main__":
    unittest.main()
