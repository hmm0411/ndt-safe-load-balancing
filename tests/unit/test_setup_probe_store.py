import unittest
from datetime import datetime, timedelta, timezone

from src.schemas.snapshot import NetworkSnapshot, SnapshotQuality
from src.schemas.telemetry import ControllerTelemetry
from src.twin.state.setup_probe_store import SetupProbeStore, SetupStats

NOW = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc)


def controller(cid: str) -> ControllerTelemetry:
    return ControllerTelemetry(
        cid, NOW, NOW, 0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 10
    )


def snapshot() -> NetworkSnapshot:
    return NetworkSnapshot(
        "snap-1", NOW, 1, 1, [controller("c1"), controller("c2")], [], [],
        SnapshotQuality(True, True, True, True),
    )


def stats(p95: float, age_ms: float) -> SetupStats:
    return SetupStats(p95, 40, 1, NOW - timedelta(milliseconds=age_ms))


class SetupProbeStoreTest(unittest.TestCase):
    def test_fresh_stats_reach_only_their_controller(self) -> None:
        store = SetupProbeStore(2500.0)
        store.update({"c1": stats(12.5, 500)})
        c1, c2 = store.overlay(snapshot()).controllers
        self.assertEqual((c1.setup_p95_ms, c1.setup_samples, c1.setup_timeouts), (12.5, 40, 1))
        self.assertIsNone(c2.setup_p95_ms)

    def test_stats_older_than_the_limit_are_not_used(self) -> None:
        store = SetupProbeStore(2500.0)
        store.update({"c1": stats(12.5, 2501)})
        self.assertIsNone(store.overlay(snapshot()).controllers[0].setup_p95_ms)

    def test_update_replaces_so_a_missing_controller_loses_its_stats(self) -> None:
        store = SetupProbeStore(2500.0)
        store.update({"c1": stats(12.5, 0), "c2": stats(9.0, 0)})
        store.update({"c1": stats(13.0, 0)})
        c1, c2 = store.overlay(snapshot()).controllers
        self.assertEqual(c1.setup_p95_ms, 13.0)
        self.assertIsNone(c2.setup_p95_ms)

    def test_empty_store_leaves_the_snapshot_unchanged(self) -> None:
        self.assertEqual(SetupProbeStore(2500.0).overlay(snapshot()), snapshot())

    def test_naive_window_end_is_rejected(self) -> None:
        store = SetupProbeStore(2500.0)
        with self.assertRaises(ValueError):
            store.update({"c1": SetupStats(1.0, 1, 0, datetime(2026, 11, 17))})


if __name__ == "__main__":
    unittest.main()
