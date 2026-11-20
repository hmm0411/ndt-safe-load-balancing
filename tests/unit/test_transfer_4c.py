from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from scripts.experiments import evaluate_transfer_4c as cli
from src.experiments.final_evaluation import transfer_4c as t4c

WORKLOADS = ["stable", "gradual", "burst", "oscillating", "hot_switch"]
MARGINS = {"global": 0.05, "controllers": {"c1": 0.05, "c2": 0.05}}


def forecast_rows(model_error: float, hot_error: float = 0.2) -> list[dict[str, Any]]:
    rows = []
    for w in WORKLOADS:
        for cid in ("c1", "c2", "c3", "c4"):
            for k in range(5):
                now, future = 0.5, 0.5 + (0.1 if k % 2 else -0.1)  # persistence error 0.1
                err = hot_error if w == "hot_switch" else model_error
                rows.append({"workload_type": w.replace("_", "-") if w == "hot_switch" else w,
                             "controller_id": cid, "utilization": now,
                             "future_utilization": future, "prediction": future - err})
    return rows


def outcome_rows(hybrid_under: float) -> list[dict[str, Any]]:
    return [{"workload": w, "u_dst_post": 0.6, "pred_analytical": 0.5,
             "pred_direct_xgb": 0.55, "pred_hybrid_xgb": 0.6 - hybrid_under}
            for w in WORKLOADS for _ in range(4)]


def rules() -> t4c.Rules:
    config = yaml.safe_load(Path(cli.ROOT / cli.CONFIG).read_text(encoding="utf-8"))
    return t4c.Rules.from_config(config)


class TransferTest(unittest.TestCase):
    def test_reuse_when_better_than_persistence_and_covered(self) -> None:
        f = t4c.forecast_transfer(forecast_rows(0.02), MARGINS, WORKLOADS)
        self.assertAlmostEqual(f["macro_rmse_ratio_seen"], 0.2)
        self.assertEqual(f["coverage_seen"], 1.0)
        self.assertEqual(f["per_workload"]["hot_switch"]["role"], "report_only")
        o = t4c.outcome_transfer(outcome_rows(0.01), 0.037, ["analytical", "direct_xgb",
                                                               "hybrid_xgb"])
        d = t4c.decide(f, o, rules())
        self.assertTrue(d["forecast_reuse"])
        self.assertTrue(d["outcome_reuse"])

    def test_hot_switch_never_decides(self) -> None:
        f = t4c.forecast_transfer(forecast_rows(0.02, hot_error=5.0), MARGINS, WORKLOADS)
        self.assertLess(f["macro_rmse_ratio_seen"], 1.0)
        self.assertEqual(f["per_workload"]["hot_switch"]["coverage"], 0.0)

    def test_retrain_when_margin_does_not_cover(self) -> None:
        f = t4c.forecast_transfer(forecast_rows(0.08), MARGINS, WORKLOADS)
        o = t4c.outcome_transfer(outcome_rows(0.08), 0.037, ["analytical", "direct_xgb",
                                                               "hybrid_xgb"])
        d = t4c.decide(f, o, rules())
        self.assertFalse(d["forecast_checks"]["coverage_at_least_min"])
        self.assertTrue(d["forecast_checks"]["macro_rmse_ratio_below_max"])
        self.assertFalse(d["outcome_reuse"])

    def test_missing_workload_is_an_error(self) -> None:
        rows = [r for r in forecast_rows(0.02) if r["workload_type"] != "burst"]
        with self.assertRaises(ValueError):
            t4c.forecast_transfer(rows, MARGINS, WORKLOADS)

    def test_cli_writes_report_with_frozen_margins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            f_path, o_path = base / "f.jsonl", base / "o.jsonl"
            f_path.write_text("\n".join(json.dumps(r) for r in forecast_rows(0.02)))
            o_path.write_text("\n".join(json.dumps(r) for r in outcome_rows(0.01)))
            out = base / "transfer.json"
            self.assertEqual(cli.main(["--forecast-rows", str(f_path), "--outcome-rows",
                                       str(o_path), "--out", str(out)]), 0)
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("forecast_reuse", report["decision"])
        self.assertAlmostEqual(report["outcome"]["margin"]["m_U"], 0.036629233239997006)


if __name__ == "__main__":
    unittest.main()
