from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from src.experiments.capacity import rules_v5 as rv
from src.experiments.common.config import load_yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_yaml(ROOT / "configs/experiments/capacity_v5_rules.yaml")


def v4_rules() -> rv.Rules:
    return rv.Rules.from_config(dict(CONFIG, experiment="capacity-2c20s-fine"))


def row(controller: str, rate: float, repeat: int, pps: float, setup: float,
        qos: float = 1.0) -> dict[str, str]:
    return {"run_id": f"capacity-4c40s-fine-{controller}-{rate:g}-r0{repeat}",
            "controller": controller, "target_new_flow_rate": str(rate),
            "processed_packet_in_rate_mean": str(pps + repeat), "qos_success_ratio": str(qos),
            "flow_setup_latency_p95_ms": str(setup), "valid": "True"}


class V4ReproductionTest(unittest.TestCase):
    def test_rules_give_back_capacity_v4(self) -> None:
        with (ROOT / "data/benchmarks/capacity_runs.csv").open(encoding="utf-8") as handle:
            out = rv.derive(list(csv.DictReader(handle)), v4_rules())
        v4 = json.loads((ROOT / "data/benchmarks/controller_capacity.json").read_text())
        self.assertEqual(out["method"]["safe_reference_rate_fps"], 90.0)
        for cid in ("c1", "c2"):
            got, want = out["controllers"][cid], v4["controllers"][cid]
            self.assertEqual((got["setup_knee_rate_fps"], got["throughput_knee_rate_fps"]),
                             (140.0, 160.0))
            for key in ("c_safe_pps", "c_setup_pps", "c_throughput_pps", "c_critical_pps"):
                self.assertAlmostEqual(got[key], want[key], places=9)
        # c2 alone would allow 100 fps; the common reference keeps 90 for both.
        self.assertEqual(out["controllers"]["c2"]["highest_healthy_rate_fps"], 100.0)


class RuleTest(unittest.TestCase):
    rules = rv.Rules.from_config(CONFIG)

    def sweep(self, setups: list[float], qos: list[float] | None = None) -> list[dict[str, str]]:
        rates = [30.0, 60.0, 90.0, 120.0][: len(setups)]
        qos = qos or [1.0] * len(setups)
        return [row("c1", r, k, r * 10, s, q)
                for r, s, q in zip(rates, setups, qos) for k in range(1, 6)]

    def test_healthy_stops_at_first_unhealthy_rate(self) -> None:
        sweep = rv.group_rows(self.sweep([10, 30, 15, 50]), self.rules)["c1"]
        self.assertEqual(rv.healthy_rates(sweep, self.rules), [30.0])  # 30 ms > 2 x 10 at 60

    def test_knees_jump_and_qos(self) -> None:
        sweep = rv.group_rows(self.sweep([10, 12, 30, 0], [1, 1, 1, 0.2]), self.rules)["c1"]
        self.assertEqual(rv.setup_knee(sweep, self.rules), 90.0)        # 30 > 2 x 12
        self.assertEqual(rv.throughput_knee(sweep, self.rules), 120.0)  # QoS 0.2 < 0.95

    def test_common_reference_is_the_lowest_highest_healthy(self) -> None:
        rows = self.sweep([10, 12, 15, 40]) + [
            dict(r, controller="c2", run_id=r["run_id"].replace("-c1-", "-c2-"))
            for r in self.sweep([10, 25, 30, 40])]
        out = rv.derive(rows, self.rules)
        self.assertEqual(out["method"]["safe_reference_rate_fps"], 30.0)
        self.assertEqual(out["controllers"]["c1"]["highest_healthy_rate_fps"], 90.0)

    def test_missing_repeat_is_an_error(self) -> None:
        rows = self.sweep([10, 12])[:-1]
        with self.assertRaisesRegex(ValueError, "4 valid repeats"):
            rv.derive(rows, self.rules)

    def test_other_experiments_and_invalid_rows_are_ignored(self) -> None:
        rows = self.sweep([10, 12]) + [dict(row("c1", 60, 9, 0, 999), valid="False")] + [
            dict(row("c1", 60, 1, 0, 999), run_id="capacity-2c20s-fine-c1-60-r01")]
        sweep = rv.group_rows(rows, self.rules)["c1"]
        self.assertEqual([len(s.processed) for s in sweep], [5, 5])

    def test_only_common_reference_is_defined(self) -> None:
        with self.assertRaises(ValueError):
            rv.Rules.from_config(dict(CONFIG, safe_reference="per_controller"))


if __name__ == "__main__":
    unittest.main()
