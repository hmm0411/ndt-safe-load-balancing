"""Fault scenarios D1-D7: plan, specs, the runner's sequencing and the expected-behaviour check."""
from __future__ import annotations

import threading
import unittest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

from src.experiments.final_evaluation import faults as F
from src.experiments.final_evaluation.fault_runner import FaultRunner, owner_of, pick_switch

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "configs/experiments/campaign_4c_v1.yaml").read_text())["faults"]
T0 = datetime(2026, 11, 27, tzinfo=timezone.utc)


def rec(t: float, status: str, target: str | None = None, u_dst: float | None = None) -> dict:
    return {"decided_at": (T0 + timedelta(seconds=t)).isoformat(), "status": status,
            "selected": {"target_controller": target} if target else None,
            "actual_post": {"u_dst": u_dst} if u_dst is not None else None}


class PlanTest(unittest.TestCase):
    def test_118_episodes_as_decided(self) -> None:
        plan = F.plan_fault_episodes(CFG, int(CFG["order_seed"]))
        self.assertEqual(len(plan), 118)          # 6 x 3 + 5 x 2 x 10 (28/11)
        self.assertEqual(len({e.episode_id for e in plan}), 118)
        by = Counter((e.method, e.workload[:2]) for e in plan)
        self.assertEqual(by[("full", "d7")], 50)
        self.assertEqual(by[("threshold", "d7")], 50)
        d7 = {(e.method, e.workload, e.seed) for e in plan if e.workload.startswith("d7")}
        self.assertIn(("threshold", "d7-u090", 9810), d7)
        for s in F.SCENARIOS:
            self.assertEqual(by[("full", s)], 3)
        self.assertIn("D-threshold-d7-u080-s9802", {e.episode_id for e in plan})
        self.assertEqual(plan, F.plan_fault_episodes(CFG, int(CFG["order_seed"])))

    def test_specs(self) -> None:
        d1 = F.fault_spec("d1", CFG)
        self.assertEqual((d1.window, d1.grace_s, d1.dst_level, d1.c1_level),
                         ((30.0, 90.0), 4.0, 0.6, 1.0))
        self.assertEqual(d1.actions[0].params["kind"], "freeze_telemetry")
        d3 = F.fault_spec("d3", CFG)
        self.assertEqual([(a.at_s, a.params["signal"]) for a in d3.actions],
                         [(30.0, "STOP"), (90.0, "CONT")])
        d4 = F.fault_spec("d4", CFG)
        self.assertEqual((d4.actions[0].params["then"], d4.actions[0].params["after_s"]),
                         ("CONT", 4.5))
        self.assertEqual(F.fault_spec("d6", CFG).actions[0].params["then"], "KILL")
        d7 = F.fault_spec("d7-u080", CFG)
        self.assertEqual((d7.dst_level, d7.actions), (0.8, ()))
        for bad in ("d8", "d7-u075", "stable"):
            with self.assertRaises(ValueError):
                F.fault_spec(bad, CFG)


class CheckTest(unittest.TestCase):
    start = T0.timestamp()

    def test_d1_pass_and_fail(self) -> None:
        spec = F.fault_spec("d1", CFG)
        ok = F.check_episode(spec, [], [rec(31, "COMMITTED"), rec(40, "INVALID_SNAPSHOT"),
                                        rec(95, "COMMITTED")], self.start)
        self.assertTrue(ok["pass"])          # 31 s is inside the grace, 95 s after the fault
        bad = F.check_episode(spec, [], [rec(40, "INVALID_SNAPSHOT"), rec(60, "COMMITTED")],
                              self.start)
        self.assertFalse(bad["pass"])
        none = F.check_episode(spec, [], [rec(40, "NO_RISK")], self.start)
        self.assertFalse(none["fault_took_effect"])
        self.assertFalse(none["pass"])

    def test_d3_never_to_the_stopped_controller(self) -> None:
        spec = F.fault_spec("d3", CFG)
        out = F.check_episode(spec, [], [rec(50, "INVALID_SNAPSHOT"),
                                         rec(120, "COMMITTED", "c3")], self.start)
        self.assertEqual(out["executed_to_c3"], 1)
        self.assertFalse(out["pass"])

    def test_d4_d6_and_d5(self) -> None:
        spec = F.fault_spec("d6", CFG)
        events: list[dict[str, Any]] = [
                  {"action": "migrate", "source": "c1", "http_status": 200, "status": "RESTORED"},
                  {"action": "post_check", "owner": "c1", "locked_switches": []}]
        self.assertTrue(F.check_episode(spec, events, [], self.start)["pass"])
        events[0].update(http_status=500, status=None)
        events[1]["locked_switches"] = ["s1"]
        self.assertFalse(F.check_episode(spec, events, [], self.start)["pass"])
        d5 = F.fault_spec("d5", CFG)
        pair: list[dict[str, Any]] = [{"action": "migrate", "http_status": 409}, {"action": "migrate", "http_status": 200}]
        self.assertTrue(F.check_episode(d5, pair, [], self.start)["pass"])
        pair[0]["http_status"] = 200
        self.assertFalse(F.check_episode(d5, pair, [], self.start)["pass"])

    def test_d7_overloaded_destination(self) -> None:
        spec = F.fault_spec("d7-u090", CFG)
        out = F.check_episode(spec, [], [rec(40, "COMMITTED", "c2", 1.04),
                                         rec(90, "NO_SAFE_ACTION")], self.start)
        self.assertEqual((out["overloaded_destinations"], out["max_u_dst_post"]), (1, 1.04))
        self.assertFalse(out["pass"])


class FakeSystem:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.lock = threading.Lock()
        self.busy = False

    def http(self, method: str, path: str, body: Any) -> tuple[int, dict]:
        with self.lock:
            self.calls.append((method, path))
            if path == "/api/v1/state":
                return 200, {"ownership": {"s10": "c1", "s2": "c1", "s1": "c2"},
                             "locked_switches": []}
            if path == "/api/v1/migrations":
                if self.busy:
                    return 409, {"detail": "s2 already has an active transaction"}
                self.busy = True
                return 200, {"status": "RESTORED", "failure_reason": "timeout"}
            return 200, {"kind": body["kind"]}

    def signal(self, cid: str, sig: str) -> list[int]:
        with self.lock:
            self.calls.append(("signal", f"{cid}:{sig}"))
        return [4242]


class RunnerTest(unittest.TestCase):
    def run_spec(self, workload: str) -> tuple[FakeSystem, list[dict]]:
        system = FakeSystem()
        log: list[dict[str, Any]] = []
        clock = [0.0]

        def sleep(s: float) -> None:
            clock[0] += s

        FaultRunner(F.fault_spec(workload, CFG), system.http, system.signal, log.append,
                    clock=lambda: clock[0], sleep=sleep).run(0.0)
        return system, log

    def test_pick_switch(self) -> None:
        state = {"ownership": {"s10": "c1", "s2": "c1", "s3": "c2"}, "locked_switches": ["s2"]}
        self.assertEqual(pick_switch(state, "c1"), "s10")
        self.assertEqual(owner_of(state, "s3"), "c2")
        with self.assertRaises(RuntimeError):
            pick_switch(state, "c4")

    def test_d3_signals_at_window_edges(self) -> None:
        system, log = self.run_spec("d3")
        self.assertEqual([(e["t"], e["signal"]) for e in log], [(30.0, "STOP"), (90.0, "CONT")])

    def test_d6_stop_migrate_kill_then_check(self) -> None:
        system, log = self.run_spec("d6")
        actions = [e["action"] + ":" + str(e.get("signal", e.get("status", ""))) for e in log]
        self.assertEqual(actions[0], "signal:STOP")
        self.assertIn("signal:KILL", actions)
        self.assertEqual(actions[-1], "post_check:")
        mig = [e for e in log if e["action"] == "migrate"][0]
        self.assertEqual((mig["switch_id"], mig["target"]), ("s2", "c3"))

    def test_d5_two_requests_one_conflict(self) -> None:
        system, log = self.run_spec("d5")
        codes = sorted(e["http_status"] for e in log if e["action"] == "migrate")
        self.assertEqual(codes, [200, 409])
        self.assertTrue(F.check_episode(F.fault_spec("d5", CFG), log, [], 0.0)["pass"])

    def test_d1_posts_the_telemetry_fault(self) -> None:
        system, log = self.run_spec("d1")
        self.assertEqual(log[0]["kind"], "freeze_telemetry")
        self.assertIn(("POST", "/api/v1/faults"), system.calls)


if __name__ == "__main__":
    unittest.main()
