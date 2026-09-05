import json
import unittest
from datetime import datetime, timezone
# from pathlib import Path
from unittest.mock import patch
from argparse import Namespace
from src.experiments.workloads.udp_new_flow import (
    PortAllocator,
    sleep_until_ns,
    validate_config,
)
from src.schemas.workload import WorkloadSample

class TestPortAllocator(unittest.TestCase):
    def test_port_allocator_increments_sequentially(self):
        allocator = PortAllocator(10000, 10003)

        ports = [allocator.next_port() for _ in range(4)]

        self.assertEqual(ports, [10000, 10001, 10002, 10003])
    def test_port_allocator_wraps_to_start(self):
        allocator = PortAllocator(10000, 10002)

        ports = [allocator.next_port() for _ in range(5)]

        self.assertEqual(ports, [10000, 10001, 10002, 10000, 10001])

class TestGeneratorValidation(unittest.TestCase):
    def make_config(self, rate):
        return Namespace(
            rate=rate,
            duration=1.0,
            target_port=5000,
            source_port_start=12000,
            source_port_end=12010,
        )
    def test_rate_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(self.make_config(0))
    def test_negative_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config(self.make_config(-1))

class TestDeadlinePacer(unittest.TestCase):
    def test_deadline_pacer_sleeps_for_remaining_time(self):
        deadline_ns = 2_000_000_000

        with patch(
            "src.experiments.workloads.udp_new_flow.time.monotonic_ns",
            side_effect=[500_000_000, 2_000_000_000],
        ) as monotonic_mock, patch(
            "src.experiments.workloads.udp_new_flow.time.sleep"
        ) as sleep_mock:
            sleep_until_ns(deadline_ns)

        sleep_mock.assert_called_once_with(1.5)
        self.assertEqual(monotonic_mock.call_count, 2)

class TestWorkloadSample(unittest.TestCase):
    def make_sample(self):
        return WorkloadSample(
            run_id="test-run",
            observed_at=datetime(2026, 9, 4, 12, 0, 0, tzinfo=timezone.utc),
            source_host="h1",
            source_ip="10.0.0.1",
            target_host="h2",
            target_ip="10.0.0.2",
            target_port=5000,
            protocol="UDP",
            pattern="new-flow",
            target_new_flow_rate=100.0,
            emitted_new_flow_rate=99.5,
            interval_seconds=1.0,
            attempted_flows=100,
            emitted_flows=99,
            send_errors=1,
            late_events=0,
            max_schedule_lag_ms=0.5,
            cumulative_attempted_flows=100,
            cumulative_emitted_flows=99,
            cumulative_send_errors=1,
            first_source_port=12000,
            last_source_port=12099,
        )
    def test_workload_sample_serializes_to_json(self):
        sample = self.make_sample()
        data = sample.to_dict()
        serialized = json.dumps(data)
        decoded = json.loads(serialized)
        self.assertEqual(decoded["run_id"], "test-run")
        self.assertEqual(decoded["protocol"], "UDP")
        self.assertEqual(decoded["pattern"], "new-flow")
        self.assertEqual(decoded["attempted_flows"], 100)
        self.assertEqual(decoded["emitted_flows"], 99)
        self.assertEqual(
            decoded["observed_at"],
            "2026-09-04T12:00:00+00:00",
        )

if __name__ == "__main__":
    unittest.main()
