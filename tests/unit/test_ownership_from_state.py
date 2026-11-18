import unittest
from datetime import datetime, timezone
from typing import Any

from src.experiments.workloads.ownership_from_state import (
    ambiguous_windows,
    build_timeline,
)

T0 = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def at(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def tx(state: str, target: str, events: list[tuple[str, float]]) -> dict[str, Any]:
    return {
        "transaction_id": "tx-1",
        "switch_id": "s1",
        "target_controller": target,
        "state": state,
        "history": [{"state": s, "at": at(o)} for s, o in events],
    }


COMMITTED = tx("COMMITTED", "c2", [("ROLE_SWITCHING", 4), ("VERIFYING", 5), ("COMMITTED", 7)])
RESTORED = tx("RESTORED", "c3", [("ROLE_SWITCHING", 20), ("FAILED", 22), ("RESTORED", 24)])


class OwnershipFromStateTest(unittest.TestCase):
    def test_owner_changes_at_verifying_not_at_finish(self) -> None:
        line = build_timeline({"s1": "c1"}, [COMMITTED], T0)
        self.assertEqual(line.owner_at("s1", T0 + 4.9), "c1")
        self.assertEqual(line.owner_at("s1", T0 + 5.1), "c2")

    def test_rolled_back_transaction_does_not_change_owner(self) -> None:
        line = build_timeline({"s1": "c1"}, [RESTORED], T0)
        self.assertEqual(line.owner_at("s1", T0 + 30), "c1")

    def test_rolled_back_transaction_is_an_ambiguous_window(self) -> None:
        windows = ambiguous_windows([COMMITTED, RESTORED], T0)
        self.assertEqual(windows, [("s1", T0 + 20, T0 + 24)])

    def test_committed_without_verifying_is_an_error(self) -> None:
        bad = tx("COMMITTED", "c2", [("ROLE_SWITCHING", 4)])
        with self.assertRaises(ValueError):
            build_timeline({"s1": "c1"}, [bad], T0)

    def test_migration_before_since_is_already_in_the_snapshot(self) -> None:
        line = build_timeline({"s1": "c2"}, [COMMITTED], T0 + 100)
        self.assertEqual(line.owner_at("s1", T0 + 101), "c2")


if __name__ == "__main__":
    unittest.main()
