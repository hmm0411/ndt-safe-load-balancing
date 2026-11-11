from __future__ import annotations

import unittest
from typing import Any

from scripts.decision import run_closed_loop as rcl


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RunTest(unittest.TestCase):
    def drive(self, answers: list[dict[str, Any]], **kw: Any) -> tuple[dict[str, Any], list[Any]]:
        calls: list[Any] = []

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            calls.append((path, dict(body)))
            return answers[min(len(calls) - 1, len(answers) - 1)]

        clock = FakeClock()
        out = rcl.run(post, kw.pop("scenario", "gradual"), kw.pop("mode", "dry_run"),
                      kw.pop("duration", 4.0), 1.0, clock=clock, sleep=clock.sleep,
                      log=lambda _: None, **kw)
        return out, calls

    def test_dry_run_counts_each_snapshot_once(self) -> None:
        answers: list[dict[str, Any]] = [
            {"decision_id": "d1", "status": "NO_RISK"},
                   {"decision_id": "d1", "status": "NO_RISK", "idempotent_replay": True},
                   {"decision_id": "d2", "status": "SELECTED"}]
        out, calls = self.drive(answers)
        self.assertEqual(out["counts"], {"gradual:NO_RISK": 1, "gradual:SELECTED": 1})
        self.assertEqual({path for path, _ in calls}, {"/api/v1/twin/decision/plan"})
        self.assertEqual(out["executed"], [])

    def test_execute_relabels_after_the_first_commit_and_fails_only_once(self) -> None:
        answers: list[dict[str, Any]] = [
            {"decision_id": "d1", "status": "RESTORED", "transaction": {"x": 1},
                    "selected": {"candidate_id": "c"}},
                   {"decision_id": "d2", "status": "DEFERRED"},
                   {"decision_id": "d3", "status": "DEFERRED"}]
        out, calls = self.drive(answers, mode="execute", scenario="verification_failure",
                                simulate_failure="flow_mod", after_commit="cooldown")
        self.assertEqual(out["counts"], {"cooldown:DEFERRED": 2,
                                         "verification_failure:RESTORED": 1})
        self.assertEqual(calls[0][1], {"scenario": "verification_failure",
                                       "simulate_failure": "flow_mod"})
        self.assertEqual(calls[1][1], {"scenario": "cooldown", "simulate_failure": "none"})
        self.assertEqual(len(out["executed"]), 1)


if __name__ == "__main__":
    unittest.main()
