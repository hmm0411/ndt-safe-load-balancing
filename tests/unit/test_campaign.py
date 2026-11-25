from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from src.experiments.final_evaluation import campaign as cp

T0 = datetime(2026, 12, 2, 1, 0, 0, tzinfo=timezone.utc)
CAP = {"c1": 900.0, "c2": 900.0}


def snap(t: float, r1: float | None, r2: float = 450.0) -> dict[str, Any]:
    ctrl = [{"controller_id": "c2", "processed_packet_in_rate": r2}]
    if r1 is not None:
        ctrl.append({"controller_id": "c1", "processed_packet_in_rate": r1})
    return {"created_at": (T0 + timedelta(seconds=t)).isoformat(), "controllers": ctrl}


def record(t: float, status: str, u_dst: float | None = None,
           method: str = "full") -> dict[str, Any]:
    rec: dict[str, Any] = {"decided_at": (T0 + timedelta(seconds=t)).isoformat(),
                           "status": status, "method": method,
                           "timings_ms": {"forecast": 1.0, "risk": 0.5}}
    if status in cp.EXECUTED:
        rec["transaction"] = {"status": status}
        rec["selected"] = {"switch_id": "s1", "source_controller": "c1",
                           "target_controller": "c2"}
        rec["actual_post"] = None if u_dst is None else {"u_dst": u_dst}
    return rec


class PlanTest(unittest.TestCase):
    def test_full_factorial_shuffled_reproducibly(self) -> None:
        plan = cp.plan_episodes(["static", "full"], ["stable", "hot_switch"], [9801, 9802], 7)
        self.assertEqual(len(plan), 8)
        self.assertEqual(len({e.episode_id for e in plan}), 8)
        self.assertEqual([e.order for e in plan], list(range(1, 9)))
        self.assertEqual(plan, cp.plan_episodes(["static", "full"], ["stable", "hot_switch"],
                                                [9801, 9802], 7))
        self.assertNotEqual([e.episode_id for e in plan],
                            [e.episode_id for e in cp.plan_episodes(
                                ["static", "full"], ["stable", "hot_switch"], [9801, 9802], 8)])
        self.assertIn("ep-full-hot-switch-s9802", {e.episode_id for e in plan})
        with self.assertRaises(ValueError):
            cp.plan_episodes(["full", "full"], ["stable"], [1], 0)


class EpisodeRowTest(unittest.TestCase):
    def test_metrics_after_warmup(self) -> None:
        ep = cp.PlannedEpisode(1, "ep-full-burst-s9801", "full", "burst", 9801)
        snaps = [snap(t, 990.0 if 40 <= t < 45 else 720.0) for t in range(0, 60)]
        snaps.append(snap(50.5, None))                    # c1 missing: skipped, not a 0
        recs = [record(10, "COMMITTED", 1.2),             # in the warm-up: dropped
                record(35, "NO_RISK"), record(41, "COMMITTED", 1.05),
                record(50, "RESTORED"), record(52, "COMMITTED")]
        row = cp.episode_row(ep, snaps, recs, CAP, T0.timestamp(), warmup_s=30)
        self.assertEqual(row["samples"], 30)
        self.assertEqual(row["skipped_samples"], 1)
        self.assertAlmostEqual(row["overload_time_s"], 5.0)      # U = 1.1 for 5 s
        self.assertEqual((row["migrations"], row["committed"], row["rollbacks"]), (3, 2, 1))
        self.assertEqual((row["unsafe_migrations"], row["unsafe_unknown"]), (1, 1))
        self.assertEqual(row["decisions"], 4)
        self.assertAlmostEqual(row["decision_p95_ms"], 1.5)
        self.assertIsNone(row["response_p95_ms"])
        self.assertEqual(json.loads(row["statuses"]),
                         {"COMMITTED": 2, "NO_RISK": 1, "RESTORED": 1})

    def test_records_of_another_method_are_refused(self) -> None:
        ep = cp.PlannedEpisode(1, "ep-full-burst-s9801", "full", "burst", 9801)
        with self.assertRaises(ValueError):
            cp.episode_row(ep, [snap(t, 720.0) for t in range(40)],
                           [record(35, "NO_RISK", method="threshold")], CAP, T0.timestamp())
        with self.assertRaises(ValueError):
            cp.episode_row(ep, [snap(t, 720.0) for t in range(10)], [], CAP, T0.timestamp())


if __name__ == "__main__":
    unittest.main()
