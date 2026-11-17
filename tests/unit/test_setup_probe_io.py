import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.workloads.setup_probe import OwnershipTimeline
from src.experiments.workloads.setup_probe_io import read_samples, row_to_sample

T0 = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def timeline() -> OwnershipTimeline:
    line = OwnershipTimeline()
    line.record(T0 - 100, "s1", "c1")
    line.record(T0 + 5, "s1", "c2")
    return line


class SetupProbeIoTest(unittest.TestCase):
    def test_success_row_keeps_latency_and_owner_at_send(self) -> None:
        row = {"observed_at": iso(4.9), "success": True, "flow_setup_latency_ms": 12.5}
        sample = row_to_sample(row, timeline(), "s1")
        self.assertEqual(sample.controller_id, "c1")
        self.assertEqual(sample.latency_ms, 12.5)

    def test_failed_row_is_a_timeout(self) -> None:
        row = {"observed_at": iso(6), "success": False, "error": "timed out"}
        sample = row_to_sample(row, timeline(), "s1")
        self.assertEqual(sample.controller_id, "c2")
        self.assertTrue(sample.timed_out)

    def test_naive_time_is_rejected(self) -> None:
        row = {"observed_at": "2026-11-17T10:00:00", "success": True}
        with self.assertRaises(ValueError):
            row_to_sample(row, timeline(), "s1")

    def test_read_samples_skips_blank_lines(self) -> None:
        rows = [
            {"observed_at": iso(1), "success": True, "flow_setup_latency_ms": 3.0},
            {"observed_at": iso(2), "success": False},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qos.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
            samples = read_samples(path, timeline(), "s1")
        self.assertEqual([s.timed_out for s in samples], [False, True])


if __name__ == "__main__":
    unittest.main()
