"""Last 5 snapshots of the what-if predictor (Week 12, Day 3)."""

from __future__ import annotations

import unittest
from typing import Any

from src.experiments.migration import fixtures as fx
from src.twin.forecasting.history import AppendStatus, HistoryNotReadyError
from src.twin.outcome.history import WINDOW_SNAPSHOTS, OutcomeHistory


def _state(index: int, valid: bool = True) -> dict[str, Any]:
    return fx.make_twin_state({"c1": 0.8, "c2": 0.3}, index=index, valid=valid)


class OutcomeHistoryTest(unittest.TestCase):
    def test_window_is_five(self) -> None:
        self.assertEqual(WINDOW_SNAPSHOTS, 5)

    def test_ready_after_five_consecutive_snapshots(self) -> None:
        history = OutcomeHistory()
        for i in range(4):
            history.observe(_state(i))
            self.assertFalse(history.ready())
        history.observe(_state(4))
        self.assertTrue(history.ready())
        self.assertEqual([v.snapshot_id for v in history.window()][-1], "snap-0004")

    def test_keeps_only_the_latest_five_in_order(self) -> None:
        history = OutcomeHistory()
        for i in range(9):
            history.observe(_state(i))
        ids = [v.snapshot_id for v in history.window()]
        self.assertEqual(ids, [f"snap-{i:04d}" for i in range(4, 9)])

    def test_duplicate_is_rejected(self) -> None:
        history = OutcomeHistory()
        history.observe(_state(0))
        self.assertEqual(history.observe(_state(0)), AppendStatus.REJECTED_DUPLICATE)
        self.assertEqual(history.size(), 1)

    def test_invalid_snapshot_is_not_stored(self) -> None:
        history = OutcomeHistory()
        history.observe(_state(0))
        status = history.observe(_state(1, valid=False))
        self.assertEqual(status, AppendStatus.REJECTED_INVALID_SNAPSHOT)
        self.assertEqual(history.size(), 1)

    def test_gap_clears_the_history(self) -> None:
        history = OutcomeHistory()
        for i in range(5):
            history.observe(_state(i))
        self.assertEqual(history.observe(_state(20)), AppendStatus.RESET_GAP)
        self.assertEqual(history.size(), 1)
        self.assertFalse(history.ready())

    def test_backwards_timestamp_clears_the_history(self) -> None:
        history = OutcomeHistory()
        for i in range(5, 8):
            history.observe(_state(i))
        self.assertEqual(history.observe(_state(1)), AppendStatus.RESET_OUT_OF_ORDER)
        self.assertEqual(history.size(), 1)

    def test_not_ready_state_payload_is_ignored(self) -> None:
        history = OutcomeHistory()
        self.assertIsNone(history.observe({"status": "NOT_READY"}))
        self.assertEqual(history.size(), 0)

    def test_missing_ids_raise(self) -> None:
        history = OutcomeHistory()
        with self.assertRaises(ValueError):
            history.observe({"controllers": []})

    def test_window_before_ready_raises(self) -> None:
        history = OutcomeHistory()
        history.observe(_state(0))
        with self.assertRaises(HistoryNotReadyError):
            history.window()

    def test_clear(self) -> None:
        history = OutcomeHistory()
        history.observe(_state(0))
        history.clear()
        self.assertEqual(history.size(), 0)


if __name__ == "__main__":
    unittest.main()
