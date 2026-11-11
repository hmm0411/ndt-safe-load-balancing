"""TransactionManager exposes what the decision layer needs: activity and last finish."""
from __future__ import annotations

import unittest

from src.orchestrator.transaction_manager import TransactionManager


class TransactionManagerStateTest(unittest.TestCase):
    def test_empty_manager(self) -> None:
        manager = TransactionManager()
        self.assertIsNone(manager.last_finished_at())
        self.assertEqual(manager.active_count(), 0)

    def test_active_until_finished(self) -> None:
        manager = TransactionManager()
        tx = manager.create("s1", "c1", "c2")
        self.assertEqual(manager.active_count(), 1)
        self.assertIsNone(manager.last_finished_at())
        manager.finish(tx)
        self.assertEqual(manager.active_count(), 0)
        self.assertEqual(manager.last_finished_at(), tx.finished_at)

    def test_last_finished_is_the_latest(self) -> None:
        manager = TransactionManager()
        first = manager.create("s1", "c1", "c2")
        manager.finish(first)
        second = manager.create("s2", "c2", "c1")
        manager.finish(second)
        self.assertEqual(
            manager.last_finished_at(), max(first.finished_at, second.finished_at)
        )


if __name__ == "__main__":
    unittest.main()
