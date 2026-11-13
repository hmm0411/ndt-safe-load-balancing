from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.decision import analyze_closed_loop as acl

TIMINGS = {"forecast": 1.0, "risk": 0.1, "candidate": 0.5, "outcome": 3.0, "safety": 0.4,
           "ranking": 0.2, "revalidation": 0.3}


def record(i: int, scenario: str, *, mode: str = "execute", status: str = "SELECTED",
           tx: str | None = None, reasons: dict[str, int] | None = None,
           actual_dst: float = 0.85) -> dict[str, Any]:
    selected = {"candidate_id": f"snap-{i}:s3:c1->c2", "predicted_src_utilization": 0.80,
                "predicted_dst_utilization": 0.83, "predicted_imbalance": 0.05,
                "utilization_margin": 0.037} if tx else None
    return {
        "decision_id": f"d{i}", "scenario": scenario, "mode": mode, "status": status,
        "snapshot_id": f"snap-{i}", "timings_ms": dict(TIMINGS), "collection_latency_ms": 12.0,
        "funnel": {"raw": 10, "eligible": 10, "evaluated": 10, "safe": 8},
        "reasons": reasons or {}, "selected": selected,
        "at_risk": [] if status in ("NO_RISK", "NOT_READY", "INVALID_SNAPSHOT") else ["c1"],
        "transaction": {"transaction_id": f"t{i}", "status": tx, "duration_s": 2.0} if tx else None,
        "actual_post": {"u_src": 0.78, "u_dst": actual_dst, "i": 0.04} if tx else None,
    }


ALL_PASS = [
    record(1, "stable", status="NO_RISK"),
    record(2, "gradual", tx="COMMITTED"),
    record(10, "sustained_overload", tx="COMMITTED"),
    record(3, "hot_switch", tx="COMMITTED", actual_dst=0.90),
    record(4, "verification_failure", tx="RESTORED"),
    record(5, "no_safe_action", status="NO_SAFE_ACTION", reasons={"DST_UTILIZATION_UNSAFE": 10}),
    record(6, "dst_near_capacity", status="NO_SAFE_ACTION",
           reasons={"DESTINATION_ALREADY_AT_RISK": 10}),
    record(7, "invalid_snapshot", status="INVALID_SNAPSHOT", reasons={"INVALID_SNAPSHOT": 10}),
    record(8, "cooldown", status="DEFERRED"),
    record(9, "conflict", mode="dry_run", status="DEFERRED"),
]


class AnalyzeTest(unittest.TestCase):
    def test_all_scenarios_pass_and_numbers_add_up(self) -> None:
        report = acl.analyze(ALL_PASS)
        self.assertTrue(report["all_gates_pass"], report["scenario_gates"])
        self.assertEqual(report["n_by_mode"], {"dry_run": 1, "execute": 9})
        total = report["latency_ms"]["decision_total"]
        self.assertEqual(total["n"], 9)
        self.assertAlmostEqual(total["p95"], sum(TIMINGS.values()))
        short = record(40, "stable", status="NO_RISK")
        short["timings_ms"] = {"forecast": 1.0, "risk": 0.1}
        both = acl.analyze(ALL_PASS + [short])["latency_ms"]
        self.assertEqual(both["decision_total"]["n"], 9)            # full pipeline only
        self.assertEqual(both["decision_total_all_records"]["n"], 10)
        pe = report["prediction_errors"]
        self.assertEqual(pe["n"], 3)
        self.assertEqual(pe["u_dst_upper_bound_coverage"], "2/3")  # 0.90 > 0.83 + 0.037
        h = report["horizon"]
        self.assertEqual(h["status"], "OK")
        self.assertEqual(h["migration_source"], "closed-loop runs")
        self.assertEqual(h["n_committed"], 3)
        self.assertEqual(h["including_rollbacks"]["n"], 4)
        self.assertEqual(len(pe["migrations"]), 3)
        self.assertFalse(any(m["dst_above_safe_limit"] for m in pe["migrations"]))
        # 12 ms + 5.5 ms + 2000 ms -> 3 steps of 1 s: h = 3 still enough
        self.assertEqual(h["horizon_steps"], 3)
        self.assertTrue(h["current_h_sufficient"])

    def test_missing_and_failing_scenarios(self) -> None:
        report = acl.analyze([record(1, "stable", tx="COMMITTED")])
        gates = report["scenario_gates"]
        self.assertEqual(gates["stable"]["result"], "FAIL")
        self.assertEqual(gates["gradual"]["result"], "MISSING")
        self.assertFalse(report["all_gates_pass"])

    def test_dry_run_only_uses_raw_v2_transaction_time(self) -> None:
        report = acl.analyze([record(1, "stable", mode="dry_run", status="NO_RISK")])
        self.assertEqual(report["latency_source"], "dry_run (no execute record yet)")
        self.assertIn("raw-v2", report["horizon"]["migration_source"])

    def test_workload_below_the_risk_threshold_is_not_triggered(self) -> None:
        quiet = [r for r in ALL_PASS if r["scenario"] != "gradual"]
        quiet += [record(20 + i, "gradual", status="NO_RISK") for i in range(3)]
        quiet += [record(25, "gradual", status="NOT_READY")]  # forecaster warm-up
        report = acl.analyze(quiet)
        self.assertEqual(report["scenario_gates"]["gradual"]["result"], "NOT_TRIGGERED")
        self.assertEqual(report["not_triggered"], ["gradual"])
        self.assertTrue(report["all_gates_pass"])
        dry = {**record(31, "gradual", mode="dry_run", status="SELECTED"), "at_risk": ["c1"]}
        self.assertEqual(acl.analyze(quiet + [dry])["scenario_gates"]["gradual"]["result"],
                         "NOT_TRIGGERED")  # the execute run never saw a risk
        mixed = quiet + [{**record(30, "gradual", status="NO_SAFE_ACTION"), "at_risk": ["c1"]}]
        self.assertEqual(acl.analyze(mixed)["scenario_gates"]["gradual"]["result"], "FAIL")

    def test_no_safe_action_ignores_warm_up_but_not_a_selected_action(self) -> None:
        base = [r for r in ALL_PASS if r["scenario"] != "no_safe_action"]
        risky = {**record(50, "no_safe_action", status="NO_SAFE_ACTION"), "at_risk": ["c1"]}
        warm = [record(51, "no_safe_action", status="NOT_READY"),
                record(52, "no_safe_action", status="NO_RISK")]
        ok = acl.analyze(base + [risky] + warm)["scenario_gates"]["no_safe_action"]
        self.assertEqual(ok["result"], "PASS")
        chosen = {**record(53, "no_safe_action", status="SELECTED"), "at_risk": ["c1"]}
        bad = acl.analyze(base + [risky, chosen])["scenario_gates"]["no_safe_action"]
        self.assertEqual(bad["result"], "FAIL")
        never = acl.analyze(base + warm)["scenario_gates"]["no_safe_action"]
        self.assertEqual(never["result"], "FAIL")  # no risk ever: the scenario did not happen

    def test_stats(self) -> None:
        self.assertEqual(acl.stats([]), {"n": 0})
        s = acl.stats([1.0, 2.0, 3.0, 4.0])
        self.assertEqual((s["n"], s["p50"], s["max"]), (4, 2.5, 4.0))


class MainTest(unittest.TestCase):
    def test_reads_records_and_refuses_incomplete_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "decision_runs"
            for r in ALL_PASS:
                (runs / r["decision_id"]).mkdir(parents=True)
                (runs / r["decision_id"] / "record.json").write_text(json.dumps(r))
            out = Path(tmp) / "analysis.json"
            self.assertEqual(acl.main(["--runs", str(runs), "--out", str(out)]), 0)
            self.assertTrue(json.loads(out.read_text())["all_gates_pass"])
            (runs / "bad").mkdir()
            (runs / "bad" / "record.json").write_text(json.dumps({"decision_id": "x"}))
            with self.assertRaises(ValueError):
                acl.main(["--runs", str(runs), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
