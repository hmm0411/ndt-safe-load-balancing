from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.capacity import pin_pilot as pp

ROOT = Path(__file__).resolve().parents[2]
RULE = yaml.safe_load((ROOT / pp.CONFIG).read_text(encoding="utf-8"))["decision"]
BASE = {"c1": 9.0, "c2": 8.0, "c3": 8.5, "c4": 9.5}


def runs(factors: dict[float, dict[str, float]], n: int = 5) -> dict[float, list[dict[str, float]]]:
    return {rate: [{c: BASE[c] * f.get(c, 1.0) for c in BASE} for _ in range(n)]
            for rate, f in factors.items()}


class PinPilotTest(unittest.TestCase):
    def test_isolated_controllers_pass(self) -> None:
        result = pp.score(runs({30: {}, 90: {"c1": 1.4, "c2": 1.05}, 140: {"c1": 2.0}}), RULE)
        self.assertEqual(result["decision"], "remeasure_pinned")

    def test_shared_slowdown_like_day3_fails(self) -> None:
        day3 = {"c1": 1.374, "c2": 1.375, "c3": 1.2, "c4": 1.3}
        result = pp.score(runs({30: {}, 90: day3, 140: {"c1": 1.9}}), RULE)
        self.assertEqual(result["decision"], "keep_huong_2")
        self.assertFalse(result["rows"][0]["others_within_limit"])

    def test_target_must_rank_first(self) -> None:
        result = pp.score(runs({30: {}, 90: {"c1": 1.1, "c3": 1.15}, 140: {"c1": 2.0}}), RULE)
        self.assertFalse(result["rows"][0]["target_ranks_first"])
        self.assertEqual(result["decision"], "keep_huong_2")

    def test_missing_repeats_is_incomplete(self) -> None:
        data = runs({30: {}, 90: {"c1": 1.4}, 140: {"c1": 2.0}})
        data[140] = data[140][:4]
        self.assertEqual(pp.score(data, RULE)["decision"], "incomplete")

    def test_collect_reads_only_valid_runs_of_the_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, ok in (("probe-pin-pilot-4c40s-c1-30-r01", True),
                             ("probe-pin-pilot-4c40s-c1-30-r02", False),
                             ("probe-sweep-4c40s-c1-30-r01", True)):
                d = root / name
                d.mkdir()
                (d / "metadata.json").write_text(json.dumps({"target_new_flow_rate": 30}))
                (d / "validation.json").write_text(json.dumps({"valid": ok}))
                (d / "setup_probe_summary.json").write_text(json.dumps(
                    {c: {"setup_p95_ms": v} for c, v in BASE.items()}))
            got = pp.collect(root, "probe-pin-pilot-4c40s")
        self.assertEqual(len(got[30.0]), 1)


if __name__ == "__main__":
    unittest.main()
