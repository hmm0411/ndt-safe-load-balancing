from __future__ import annotations

import copy
import unittest
from pathlib import Path
from typing import Any

from src.experiments.capacity.probe_acceptance import CONTROLLERS, judge
from src.experiments.common.config import load_yaml

ROOT = Path(__file__).resolve().parents[2]
RULES = load_yaml(ROOT / "configs/experiments/probe_acceptance_v1.yaml")
BASE = {"c1": 10.0, "c2": 11.0, "c3": 10.5, "c4": 9.5}


def cell(p95: float, timeouts: int = 0, samples: int = 3000) -> dict[str, Any]:
    return {"setup_p95_ms": {"mean": p95, "min": p95, "max": p95, "n": 5},
            "timeouts": timeouts, "samples": samples, "processed_pps_mean": 300.0}


def good_summary() -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for target in CONTROLLERS:
        curves[target] = {}
        for rate, factor in (("30", 1.0), ("90", 1.6), ("140", 4.0)):
            curves[target][rate] = {cid: cell(BASE[cid] * (factor if cid == target else 1.05))
                                    for cid in CONTROLLERS} | {"repeats": 5}
        curves[target]["30"] = {cid: cell(BASE[cid]) for cid in CONTROLLERS} | {"repeats": 5}
    load = {cid: {"delta_u": 0.01, "delta_pps": 9.0} for cid in CONTROLLERS}
    return {"curves": curves, "probe_load": load, "scale_pps": 933.0}


class ProbeAcceptanceTest(unittest.TestCase):
    def test_good_sweep_passes_all(self) -> None:
        result = judge(good_summary(), RULES)
        self.assertTrue(result["all_pass"], {k: v["pass"] for k, v in result["criteria"].items()})
        self.assertTrue(result["decision"].startswith("Huong 3"))

    def test_leak_to_another_controller_fails_isolation(self) -> None:
        summary = good_summary()
        summary["curves"]["c1"]["140"]["c2"] = cell(BASE["c2"] * 2.0)
        result = judge(summary, RULES)
        self.assertFalse(result["criteria"]["P1_isolation"]["pass"])
        self.assertEqual(result["decision"], "Huong 2")

    def test_flat_curve_fails_monotone_and_saturation(self) -> None:
        summary = good_summary()
        for rate in ("90", "140"):
            summary["curves"]["c3"][rate]["c3"] = cell(BASE["c3"])
        criteria = judge(summary, RULES)["criteria"]
        self.assertFalse(criteria["P2_monotone"]["pass"])
        self.assertFalse(criteria["P3_saturation_visible"]["pass"])

    def test_timeouts_alone_show_saturation(self) -> None:
        summary = good_summary()
        summary["curves"]["c4"]["140"]["c4"] = cell(BASE["c4"] * 1.2, timeouts=3)
        self.assertTrue(judge(summary, RULES)["criteria"]["P3_saturation_visible"]["pass"])

    def test_missing_probe_load_and_few_samples_fail(self) -> None:
        summary = copy.deepcopy(good_summary())
        summary["probe_load"]["c2"]["delta_u"] = None
        summary["curves"]["c1"]["90"]["c3"]["samples"] = 100
        criteria = judge(summary, RULES)["criteria"]
        self.assertFalse(criteria["P4_probe_load"]["pass"])
        self.assertFalse(criteria["P5_samples"]["pass"])


if __name__ == "__main__":
    unittest.main()
