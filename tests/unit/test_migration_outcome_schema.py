"""Schema contract of the Week 11 State-Action-Outcome record (required fields, enums)."""
from __future__ import annotations

import copy
import unittest

from src.experiments.migration import fixtures as fx
from src.experiments.migration.schema import (
    MigrationAction,
    MigrationTransactionView,
    StateSnapshotView,
)


class SnapshotViewTest(unittest.TestCase):
    def test_reads_raw_twin_state(self) -> None:
        snap = StateSnapshotView.from_twin_state(fx.make_twin_state({"c1": 0.8, "c2": 0.45}))
        self.assertEqual(snap.utilization(), {"c1": 0.8, "c2": 0.45})
        self.assertEqual(len(snap.switches), 20)
        self.assertEqual(snap.owner_of("s1"), "c1")
        self.assertEqual(snap.owner_of("s11"), "c2")
        self.assertAlmostEqual(snap.controller("c1").safe_capacity_pps, fx.SAFE_CAPACITY)

    def test_quality_fields_are_kept(self) -> None:
        raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45}, consistency_score=0.975, quality_score=0.9)
        quality = StateSnapshotView.from_twin_state(raw).quality
        self.assertEqual(quality.consistency_score, 0.975)
        self.assertEqual(quality.quality_score, 0.9)
        self.assertFalse(quality.consistent)

    def test_not_ready_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StateSnapshotView.from_twin_state({"status": "NOT_READY"})

    def test_missing_required_field_is_rejected(self) -> None:
        raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45})
        del raw["controllers"][0]["utilization"]
        with self.assertRaisesRegex(ValueError, "utilization"):
            StateSnapshotView.from_twin_state(raw)

    def test_nan_utilization_is_rejected(self) -> None:
        raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45})
        raw["controllers"][1]["utilization"] = float("nan")
        with self.assertRaises(ValueError):
            StateSnapshotView.from_twin_state(raw)

    def test_switch_reported_by_two_controllers(self) -> None:
        # After a migration both controllers can report s1; the view keeps both.
        raw = fx.make_twin_state({"c1": 0.7, "c2": 0.5})
        stale = copy.deepcopy(raw["switches"][0])
        stale.update(controller_id="c2", processed_packet_in_rate=0.0, control_load_share=0.0)
        raw["switches"].append(stale)
        snap = StateSnapshotView.from_twin_state(raw)
        c1_entry = snap.switch_load("s1", "c1")
        c2_entry = snap.switch_load("s1", "c2")
        assert c1_entry is not None and c2_entry is not None
        self.assertGreater(c1_entry.processed_packet_in_rate, 0.0)
        self.assertEqual(c2_entry.processed_packet_in_rate, 0.0)

    def test_controller_agnostic_4c(self) -> None:
        raw = fx.make_twin_state({"c1": 0.9, "c2": 0.4, "c3": 0.5, "c4": 0.6})
        snap = StateSnapshotView.from_twin_state(raw)
        self.assertEqual(len(snap.switches), 40)
        self.assertEqual(snap.owner_of("s31"), "c4")

    def test_memory_is_optional_and_none_when_missing(self) -> None:
        raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45})
        for controller in raw["controllers"]:
            controller.pop("process_memory_rss_mb", None)
        view = StateSnapshotView.from_twin_state(raw).controller("c1")
        self.assertIsNone(view.process_memory_rss_mb)
        self.assertNotIn("process_memory_rss_mb", view.to_dict())

    def test_memory_is_read_when_present(self) -> None:
        raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45})
        cid = raw["controllers"][0]["controller_id"]
        raw["controllers"][0]["process_memory_rss_mb"] = 123.4
        view = StateSnapshotView.from_twin_state(raw).controller(cid)
        self.assertEqual(view.process_memory_rss_mb, 123.4)
        self.assertEqual(view.to_dict()["process_memory_rss_mb"], 123.4)

    def test_memory_that_is_not_a_finite_number_is_rejected(self) -> None:
        for bad in ("lots", float("nan")):
            raw = fx.make_twin_state({"c1": 0.8, "c2": 0.45})
            raw["controllers"][0]["process_memory_rss_mb"] = bad
            with self.assertRaisesRegex(ValueError, "process_memory_rss_mb"):
                StateSnapshotView.from_twin_state(raw)


class ActionTest(unittest.TestCase):
    def test_valid_migrate_and_noop(self) -> None:
        MigrationAction("MIGRATE", "c1", "c2", switch_id="s3")
        MigrationAction("NO_OP", "c2", "c1")

    def test_invalid_actions(self) -> None:
        with self.assertRaises(ValueError):
            MigrationAction("MOVE", "c1", "c2", switch_id="s3")
        with self.assertRaises(ValueError):
            MigrationAction("MIGRATE", "c1", "c2")
        with self.assertRaises(ValueError):
            MigrationAction("MIGRATE", "c1", "c1", switch_id="s3")
        with self.assertRaises(ValueError):
            MigrationAction("NO_OP", "c1", "c2", failure_injected=True)


class TransactionTest(unittest.TestCase):
    def test_statuses(self) -> None:
        self.assertEqual(
            MigrationTransactionView.from_response(fx.committed_response()).status, "COMMITTED"
        )
        restored = MigrationTransactionView.from_response(fx.restored_response())
        self.assertEqual(restored.status, "RESTORED")
        self.assertEqual(restored.history[-1].state, "RESTORED")
        self.assertEqual(MigrationTransactionView.failed(fx.failed_detail()).status, "FAILED")
        self.assertEqual(MigrationTransactionView.skipped("source_not_master").status, "SKIPPED")

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MigrationTransactionView(status="ROLLED_BACK")

    def test_committed_needs_matching_history(self) -> None:
        payload = fx.committed_response()
        payload["transaction"]["history"] = payload["transaction"]["history"][:-1]
        with self.assertRaises(ValueError):
            MigrationTransactionView.from_response(payload)

    def test_skipped_needs_reason(self) -> None:
        with self.assertRaises(ValueError):
            MigrationTransactionView(status="SKIPPED")


if __name__ == "__main__":
    unittest.main()
