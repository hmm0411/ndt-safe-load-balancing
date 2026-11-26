import unittest
from datetime import datetime, timezone

from src.schemas.snapshot import SnapshotQuality
from src.twin.state.twinning_rate import TwinningRateTracker, migration_transient


class TwinningRateTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def test_ten_successes_have_rate_one(self):
        tracker = TwinningRateTracker(window_seconds=60)
        for _ in range(10):
            tracker.record(True, self.now)

        self.assertEqual(tracker.rate(self.now), 1.0)

    def test_nine_successes_and_one_failure_have_rate_point_nine(self):
        tracker = TwinningRateTracker(window_seconds=60)
        for _ in range(9):
            tracker.record(True, self.now)
        tracker.record(False, self.now)

        self.assertEqual(tracker.rate(self.now), 0.9)

    def test_empty_tracker_has_zero_rate(self):
        tracker = TwinningRateTracker(window_seconds=60)

        self.assertEqual(tracker.rate(self.now), 0.0)

    def test_reset_clears_existing_measurements(self):
        tracker = TwinningRateTracker(window_seconds=60)
        tracker.record(True, self.now)
        tracker.record(False, self.now)

        self.assertEqual(tracker.rate(self.now), 0.5)

        tracker.reset()

        self.assertEqual(tracker.rate(self.now), 0.0)

    def test_new_measurement_after_reset_starts_fresh_window(self):
        tracker = TwinningRateTracker(window_seconds=60)
        tracker.record(False, self.now)
        tracker.record(False, self.now)

        tracker.reset()
        tracker.record(True, self.now)

        self.assertEqual(tracker.rate(self.now), 1.0)


def bad(**kw):
    return SnapshotQuality(fresh=not kw.get("stale_sources"), complete=not kw.get("missing_fields"),
                           consistent=not kw.get("conflicts"), valid=False, **kw)


class MigrationTransientTests(unittest.TestCase):
    """Pilot 3 (26/11): role mismatches of the switch being moved, and only those."""

    def test_role_mismatch_on_the_moving_switch_is_transient(self):
        q = bad(conflicts=["role_mismatch:s12:c2:expected_MASTER:actual_SLAVE",
                           "role_mismatch:s12:c4:expected_SLAVE:actual_MASTER"])
        self.assertTrue(migration_transient(q, {"s12"}))

    def test_other_switch_or_no_transaction_is_a_failure(self):
        q = bad(conflicts=["role_mismatch:s7:c1:expected_MASTER:actual_SLAVE"])
        self.assertFalse(migration_transient(q, {"s12"}))
        self.assertFalse(migration_transient(q, set()))
        mixed = bad(conflicts=["role_mismatch:s12:c4:expected_SLAVE:actual_MASTER",
                               "role_mismatch:s3:c1:expected_MASTER:actual_EQUAL"])
        self.assertFalse(migration_transient(mixed, {"s12"}))

    def test_faults_stay_failures(self):
        for q in (bad(conflicts=["disconnected:s12:c4"]),
                  bad(conflicts=["role_state_missing:s12:c4"]),
                  bad(conflicts=["role_mismatch:s12:c4:expected_SLAVE:actual_MASTER"],
                      stale_sources=["controller:c4"]),
                  bad(missing_fields=["switch:s12:ownership"])):
            self.assertFalse(migration_transient(q, {"s12"}))

    def test_valid_snapshot_is_not_transient(self):
        ok = SnapshotQuality(fresh=True, complete=True, consistent=True, valid=True)
        self.assertFalse(migration_transient(ok, {"s12"}))

    def test_skip_neither_succeeds_nor_fails(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        tracker = TwinningRateTracker(window_seconds=60)
        for _ in range(12):
            tracker.record(True, now)
        tracker.skip()
        tracker.skip()
        self.assertEqual(tracker.rate(now), 1.0)   # was 12/14 = 0.857 in pilot 3
        self.assertEqual(tracker.skipped, 2)
        tracker.reset()
        self.assertEqual(tracker.skipped, 0)


if __name__ == "__main__":
    unittest.main()
