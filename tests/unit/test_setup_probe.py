from __future__ import annotations

import unittest

from src.experiments.workloads.setup_probe import (
    OwnershipTimeline,
    SetupSample,
    SetupWindowStats,
    make_sample,
    summarize_window,
)

TIMEOUT = 1000.0


def window(samples, controller_id="c1", start=0.0, end=100.0) -> SetupWindowStats:
    result = summarize_window(samples, controller_id, start, end, TIMEOUT)
    assert result is not None
    return result


class SetupProbeTests(unittest.TestCase):
    def test_sample_sent_just_before_migration_counts_for_old_controller(self):
        timeline = OwnershipTimeline()
        timeline.record(0.0, "s1", "c1")
        timeline.record(100.0, "s1", "c2")  # s1 migrates c1 -> c2 at t=100
        before = make_sample(timeline, "s1", 99.9, 30.0)
        after = make_sample(timeline, "s1", 100.1, 40.0)
        self.assertEqual((before.controller_id, after.controller_id), ("c1", "c2"))
        samples = [before, after]
        old = window(samples, "c1", 90.0, 110.0)
        new = window(samples, "c2", 90.0, 110.0)
        self.assertEqual((old.samples, old.p95_ms), (1, 30.0))
        self.assertEqual((new.samples, new.p95_ms), (1, 40.0))

    def test_timeout_counts_as_worst_value_in_p95(self):
        samples = [SetupSample("s1", "c1", float(i), 10.0) for i in range(18)]
        samples += [SetupSample("s1", "c1", 18.0 + i, None) for i in range(2)]
        stats = window(samples)
        self.assertEqual((stats.samples, stats.timeouts), (20, 2))
        self.assertEqual(stats.p95_ms, TIMEOUT)
        kept = [s for s in samples if not s.timed_out]
        self.assertEqual(window(kept).p95_ms, 10.0)  # dropping timeouts hides them

    def test_only_timeouts_give_the_timeout_value(self):
        samples = [SetupSample("s1", "c1", 1.0, None)]
        stats = window(samples)
        self.assertEqual((stats.p95_ms, stats.timeouts), (TIMEOUT, 1))

    def test_window_without_samples_returns_none_not_zero(self):
        self.assertIsNone(summarize_window([], "c1", 0.0, 10.0, TIMEOUT))
        other_controller = [SetupSample("s1", "c2", 5.0, 12.0)]
        self.assertIsNone(summarize_window(other_controller, "c1", 0.0, 10.0, TIMEOUT))
        outside = [SetupSample("s1", "c1", 50.0, 12.0)]
        self.assertIsNone(summarize_window(outside, "c1", 0.0, 10.0, TIMEOUT))

    def test_window_start_is_included_and_end_is_excluded(self):
        samples = [SetupSample("s1", "c1", t, 10.0) for t in (0.0, 5.0, 10.0)]
        self.assertEqual(window(samples, start=0.0, end=10.0).samples, 2)

    def test_unknown_owner_is_rejected(self):
        timeline = OwnershipTimeline()
        timeline.record(10.0, "s1", "c1")
        self.assertIsNone(timeline.owner_at("s1", 9.0))
        with self.assertRaises(ValueError):
            make_sample(timeline, "s1", 9.0, 10.0)

    def test_ownership_changes_must_be_in_time_order(self):
        timeline = OwnershipTimeline()
        timeline.record(10.0, "s1", "c1")
        with self.assertRaises(ValueError):
            timeline.record(5.0, "s1", "c2")


if __name__ == "__main__":
    unittest.main()
