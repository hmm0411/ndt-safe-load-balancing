import unittest
from datetime import datetime, timezone

from src.schemas.snapshot import OwnershipState
from src.schemas.telemetry import ControllerTelemetry
from src.telemetry.snapshot_builder import SnapshotBuilder
from src.telemetry.snapshot_validator import SnapshotValidator


def controller(controller_id):
    now = datetime.now(timezone.utc)
    return ControllerTelemetry(
        controller_id=controller_id,
        observed_at=now,
        ingested_at=now,
        packet_in_total=0,
        packet_in_rate=0.0,
        processed_packet_in_total=0,
        processed_packet_in_rate=0.0,
        flow_mod_total=0,
        flow_mod_rate=0.0,
        process_cpu_percent=0.0,
        process_memory_rss_mb=0.0,
        response_mean_ms=0.0,
        response_p95_ms=0.0,
        managed_switch_count=1,
    )


class SnapshotBuilderTests(unittest.TestCase):
    def test_build_valid_snapshot(self):
        validator = SnapshotValidator({"c1", "c2"}, {"s1"}, 2500)
        builder = SnapshotBuilder(validator)
        snapshot = builder.build(
            topology_version=1,
            ownership_version=2,
            controllers=[controller("c1"), controller("c2")],
            switches=[],
            ownership=[OwnershipState("s1", "c1", "MASTER", "SLAVE", 1, 2)],
            role_matrix={
                ("s1", "c1"): {"role": "MASTER", "connected": True},
                ("s1", "c2"): {"role": "SLAVE", "connected": True},
            },
        )
        self.assertEqual(snapshot.ownership_version, 2)
        self.assertTrue(snapshot.quality.valid)


if __name__ == "__main__":
    unittest.main()
