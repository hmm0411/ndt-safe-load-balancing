from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.decision import ranking_sensitivity as script
from scripts.experiments import benchmark_outcome_inference as bench
from src.experiments.migration import fixtures as fx
from src.twin.decision import ranking_sensitivity as rs
from src.twin.decision.ranking import NO_SAFE_ACTION, SELECTED
from src.twin.decision.ranking_policy import load_ranking_policy
from src.twin.decision.safety_gate import SafetyGate
from src.twin.decision.safety_policy import load_safety_policy

ARTIFACT = bench.ROOT / "data" / "models" / "outcome" / "post-migration-v1"
CAPACITY = bench.ROOT / "data" / "benchmarks" / "controller_capacity.json"
NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)


def states(util: dict[str, float]) -> list[dict[str, Any]]:
    return [fx.make_twin_state(util, hot_switch=("s3", 0.3), index=i) for i in range(5)]


def policies() -> dict[str, Any]:
    ranking = load_ranking_policy()
    return {name: ranking.with_weights(name, w) for name, w in ranking.sensitivity}


class HelpersTest(unittest.TestCase):
    def test_kendall_tau(self) -> None:
        self.assertEqual(rs.kendall_tau(["a", "b", "c"], ["a", "b", "c"]), 1.0)
        self.assertEqual(rs.kendall_tau(["a", "b", "c"], ["c", "b", "a"]), -1.0)
        self.assertIsNone(rs.kendall_tau(["a"], ["a"]))
        self.assertAlmostEqual(rs.kendall_tau(["a", "b", "c"], ["b", "a", "c"]) or 0, 1 / 3)

    def test_twin_round_trip(self) -> None:
        raw = states({"c1": 0.9, "c2": 0.4})[-1]
        twin = rs.twin_from_payload(raw)
        self.assertEqual(twin.snapshot_id, raw["snapshot_id"])
        self.assertEqual(rs.twin_from_payload(twin.to_dict()).to_dict(), twin.to_dict())


class EpochTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor, _ = bench.build_predictor(ARTIFACT, CAPACITY)
        self.gate = SafetyGate(load_safety_policy())

    def epoch(self, util: dict[str, float]) -> rs.Epoch:
        rows = states(util)
        self.predictor.history.clear()
        for row in rows:
            self.predictor.observe(row)
        return rs.rank_epoch("e1", rows[-1], self.predictor, self.gate, policies(), NOW)

    def test_loaded_source_gives_ranked_candidates(self) -> None:
        epoch = self.epoch({"c1": 0.95, "c2": 0.30})
        self.assertEqual(epoch.source, "c1")
        self.assertGreater(epoch.n_candidates, 1)
        base = epoch.results["base"]
        self.assertEqual(base.n_candidates, epoch.n_candidates)
        if base.ranked:
            self.assertIn(base.status, (SELECTED, "NO_BENEFICIAL_ACTION"))
        summary = rs.summarize([epoch])
        self.assertEqual(summary["n_epochs"], 1)
        self.assertEqual(set(summary["policies"]), {"base", "relief_heavy", "balance_heavy",
                                                    "qos_heavy"})

    def test_full_destination_gives_no_safe_action(self) -> None:
        epoch = self.epoch({"c1": 1.10, "c2": 0.99})
        self.assertEqual(epoch.n_safe, 0)
        self.assertEqual(epoch.results["base"].status, NO_SAFE_ACTION)


class ScriptTest(unittest.TestCase):
    def test_runs_over_attempt_directories_and_refuses_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "safety-stress-v1"
            for k, util in enumerate(({"c1": 0.95, "c2": 0.30}, {"c1": 0.92, "c2": 0.50})):
                attempt = root / f"mig-{k}"
                attempt.mkdir(parents=True)
                lines = [json.dumps(s) for s in states(util)]
                (attempt / "before_samples.jsonl").write_text("\n".join(lines) + "\n")
            (root / "mig-empty").mkdir()
            out = Path(tmp) / "out.json"
            self.assertEqual(script.main(["--runs", str(root), "--out", str(out)]), 0)
            report = json.loads(out.read_text())
            self.assertEqual(report["n_epochs"], 2)
            self.assertEqual(report["skipped"]["no_before_samples"], 1)
            self.assertTrue(out.with_suffix(".csv").exists())
            test_root = Path(tmp) / "outcome-test"
            test_root.mkdir()
            with self.assertRaises(SystemExit):
                script.main(["--runs", str(test_root), "--out", str(out)])


if __name__ == "__main__":
    unittest.main()
