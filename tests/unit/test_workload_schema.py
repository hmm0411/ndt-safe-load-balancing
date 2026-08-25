from datetime import datetime, timezone
import unittest

from src.schemas.workload import WorkloadSample


class WorkloadSchemaTests(unittest.TestCase):
    def test_serialization(self):
        sample = WorkloadSample(
            run_id="workload-smoke-c1-10-r01",
            observed_at=datetime.now(timezone.utc),
            source_host="h1",
            source_ip="10.0.0.1",
            target_host="h2",
            target_ip="10.0.0.2",
            target_port=9000,
            protocol="udp",
            pattern="stable",
            target_new_flow_rate=10.0,
            emitted_new_flow_rate=9.9,
            interval_seconds=1.0,
            attempted_flows=10,
            emitted_flows=10,
            send_errors=0,
            late_events=0,
            max_schedule_lag_ms=0.2,
            cumulative_attempted_flows=10,
            cumulative_emitted_flows=10,
            cumulative_send_errors=0,
            first_source_port=12000,
            last_source_port=12009,
        )

        payload = sample.to_dict()

        self.assertEqual(
            payload["target_new_flow_rate"],
            10.0,
        )

        self.assertEqual(
            payload["emitted_new_flow_rate"],
            9.9,
        )

        self.assertEqual(
            payload["source_host"],
            "h1",
        )


if __name__ == "__main__":
    unittest.main()