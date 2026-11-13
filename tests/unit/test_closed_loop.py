from __future__ import annotations

import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from typing import Any

from decision_testdata import NOW, make_roles, make_twin
from src.schemas.decision import MigrationCandidate
from src.twin.decision import closed_loop as cl
from src.twin.decision import decision_context as dc


def candidate(switch: str = "s1") -> MigrationCandidate:
    return MigrationCandidate("snap-1", switch, "c1", "c2", 0.95, 0.40, 0.1, NOW)


def state(index: int, valid: bool = True, c1: float = 0.7, c2: float = 0.6) -> dict[str, Any]:
    return {
        "snapshot_id": f"snap-{index}",
        "created_at": (NOW + timedelta(seconds=index)).isoformat(),
        "quality": {"valid": valid},
        "controllers": [{"controller_id": "c1", "utilization": c1},
                        {"controller_id": "c2", "utilization": c2}],
    }


class ConfigTest(unittest.TestCase):
    def test_repo_config_loads(self) -> None:
        cfg = cl.load_config()
        self.assertEqual(cfg.version, "closed-loop-2c-v1")
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.post_window_valid_snapshots, 5)

    def test_bad_configs_are_refused(self) -> None:
        good = {"version": "v", "enabled": True, "record_dir": "d", "post_settle_seconds": 1,
                "post_window_valid_snapshots": 5, "max_cached_decisions": 8}
        for key, value in (("enabled", "yes"), ("post_settle_seconds", -1),
                           ("post_window_valid_snapshots", 0), ("max_cached_decisions", True),
                           ("record_dir", "")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                cl.parse_config({**good, key: value})


class DecisionContextTest(unittest.TestCase):
    def test_id_is_a_function_of_mode_and_snapshot(self) -> None:
        a = dc.DecisionContext.pin("dry_run", "stable", "snap/1 x", NOW, NOW)
        self.assertEqual(a.decision_id, "dec-dry_run-snap_1_x")
        self.assertEqual(a.decision_id, dc.make_decision_id("dry_run", "snap/1 x"))
        self.assertNotEqual(a.decision_id, dc.make_decision_id("execute", "snap/1 x"))
        json.dumps(a.to_dict())

    def test_invalid_contexts(self) -> None:
        with self.assertRaises(ValueError):
            dc.make_decision_id("maybe", "snap-1")
        with self.assertRaises(ValueError):
            dc.make_decision_id("execute", " ")
        with self.assertRaises(ValueError):
            dc.DecisionContext("dec-x", "execute", "s", "snap-1", NOW, NOW)
        with self.assertRaises(ValueError):
            dc.DecisionContext.pin("execute", "", "snap-1", NOW, NOW)


class RevalidateTest(unittest.TestCase):
    def check(self, **over: Any) -> tuple[str, ...]:
        twin = over.pop("twin", make_twin())
        args: dict[str, Any] = dict(
            role_matrix=make_roles(twin), destination_forecast_risk=False,
            active_transaction=False, seconds_since_last_migration=None, cooldown_seconds=37.0,
        )
        args.update(over)
        return cl.revalidate(candidate(), twin, **args)

    def test_unchanged_state_passes_and_unknown_risk_is_not_a_reason(self) -> None:
        self.assertEqual(self.check(), ())
        self.assertEqual(self.check(destination_forecast_risk=None), ())
        self.assertEqual(self.check(seconds_since_last_migration=37.0), ())

    def test_each_change_gives_its_reason(self) -> None:
        self.assertEqual(self.check(twin=make_twin(valid=False)), ("INVALID_SNAPSHOT",))
        moved = make_twin(owners={"s1": "c2", "s2": "c1"})
        self.assertIn("OWNER_MISMATCH", self.check(twin=moved))
        self.assertEqual(self.check(role_matrix={}), ("DESTINATION_DISCONNECTED",))
        self.assertEqual(self.check(active_transaction=True), ("ACTIVE_TRANSACTION_CONFLICT",))
        self.assertEqual(self.check(seconds_since_last_migration=5.0), ("COOLDOWN_ACTIVE",))
        self.assertEqual(self.check(destination_forecast_risk=True),
                         ("DESTINATION_ALREADY_AT_RISK",))


class PostWindowTest(unittest.TestCase):
    def test_mean_of_valid_snapshots_after_the_settle_time(self) -> None:
        window = cl.PostWindow("d1", "c1", "c2", NOW, settle_seconds=3, window=2)
        self.assertIsNone(window.observe(state(1)))             # before settle
        self.assertIsNone(window.observe(state(3, valid=False)))  # invalid
        self.assertIsNone(window.observe(state(4, c1=0.70, c2=0.60)))
        self.assertIsNone(window.observe(state(4, c1=0.70, c2=0.60)))  # duplicate
        actual = window.observe(state(5, c1=0.80, c2=0.50))
        assert actual is not None
        self.assertAlmostEqual(actual["u_src"], 0.75)
        self.assertAlmostEqual(actual["u_dst"], 0.55)
        self.assertAlmostEqual(actual["i"], 0.10)
        self.assertEqual(actual["snapshot_ids"], ["snap-4", "snap-5"])
        self.assertIsNone(window.observe(state(6)))             # already complete


class RecordTest(unittest.TestCase):
    def test_record_is_written_atomically_and_rewritten_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = cl.write_record(Path(tmp), {"decision_id": "d1", "x": float("nan"), "at": NOW})
            self.assertEqual(json.loads(path.read_text())["x"], None)
            cl.write_record(Path(tmp), {"decision_id": "d1", "x": 2})
            self.assertEqual(json.loads(path.read_text())["x"], 2)
            self.assertEqual([p.name for p in path.parent.iterdir()], ["record.json"])


if __name__ == "__main__":
    unittest.main()
