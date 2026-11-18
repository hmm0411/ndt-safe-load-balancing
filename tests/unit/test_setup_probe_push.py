import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.workloads.setup_probe import OwnershipTimeline, SetupSample
from src.experiments.workloads.setup_probe_push import (
    RawSample,
    SampleTail,
    payload,
    to_samples,
    window_stats,
)

T0 = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def row(offset: float, latency: float | None) -> str:
    when = datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()
    data = {"observed_at": when, "success": latency is not None}
    if latency is not None:
        data["flow_setup_latency_ms"] = latency
    return json.dumps(data)


class SampleTailTest(unittest.TestCase):
    def test_reads_only_complete_new_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qos_s1.jsonl"
            tail = SampleTail(path, "s1")
            self.assertEqual(tail.read_new(), [])
            path.write_text(row(1, 5.0) + "\n" + row(2, None) + "\n" + row(3, 7.0)[:20])
            first = tail.read_new()
            self.assertEqual([r.latency_ms for r in first], [5.0, None])
            with path.open("a") as handle:
                handle.write(row(3, 7.0)[20:] + "\n")
            self.assertEqual([r.latency_ms for r in tail.read_new()], [7.0])
            self.assertEqual(tail.read_new(), [])


class WindowTest(unittest.TestCase):
    def test_samples_before_the_owner_is_known_are_counted(self) -> None:
        line = OwnershipTimeline()
        line.record(T0 + 5, "s1", "c1")
        samples, unknown = to_samples(
            [RawSample("s1", T0 + 1, 3.0), RawSample("s1", T0 + 6, 4.0)], line
        )
        self.assertEqual((len(samples), unknown), (1, 1))

    def test_rolled_back_window_and_outside_samples_are_left_out(self) -> None:
        samples = [
            SetupSample("s1", "c1", T0 + 1, 1.0),
            SetupSample("s1", "c1", T0 + 3, 100.0),
            SetupSample("s1", "c1", T0 + 50, 2.0),
        ]
        stats, excluded = window_stats(
            samples, [("s1", T0 + 2, T0 + 4)], T0, T0 + 10, 1000.0, ["c1", "c2"]
        )
        self.assertEqual(excluded, 1)
        self.assertEqual(stats["c1"].samples, 1)
        self.assertNotIn("c2", stats)

    def test_payload_has_a_window_end_with_timezone(self) -> None:
        samples = [SetupSample("s1", "c1", T0 + 1, 9.0)]
        stats, _ = window_stats(samples, [], T0, T0 + 10, 1000.0, ["c1"])
        body = payload(stats, datetime.fromtimestamp(T0 + 10, timezone.utc))
        item = body["controllers"]["c1"]
        self.assertEqual((item["p95_ms"], item["samples"], item["timeouts"]), (9.0, 1, 0))
        self.assertTrue(item["window_end"].endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
