import unittest
from datetime import datetime, timezone

from src.schemas.telemetry import ControllerTelemetry, SwitchTelemetry


class TelemetrySchemaTests(unittest.TestCase):
    def test_controller_serialization_contains_processed_rate(self):
        now = datetime.now(timezone.utc)
        item = ControllerTelemetry(
            controller_id="c1",
            observed_at=now,
            ingested_at=now,
            packet_in_total=10,
            packet_in_rate=5.0,
            processed_packet_in_total=8,
            processed_packet_in_rate=4.0,
            flow_mod_total=7,
            flow_mod_rate=3.0,
            process_cpu_percent=12.0,
            process_memory_rss_mb=100.0,
            response_mean_ms=1.0,
            response_p95_ms=2.0,
            managed_switch_count=2,
        )
        self.assertEqual(item.to_dict()["processed_packet_in_rate"], 4.0)

    def test_switch_serialization_contains_processed_rate(self):
        now = datetime.now(timezone.utc)
        item = SwitchTelemetry(
            switch_id="s1",
            controller_id="c1",
            observed_at=now,
            packet_in_total=10,
            packet_in_rate=5.0,
            processed_packet_in_total=8,
            processed_packet_in_rate=4.0,
            flow_mod_total=7,
            flow_mod_rate=3.0,
            control_load_share=0.5,
        )
        self.assertEqual(item.to_dict()["processed_packet_in_rate"], 4.0)


if __name__ == "__main__":
    unittest.main()
