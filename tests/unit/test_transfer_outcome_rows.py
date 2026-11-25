from __future__ import annotations

import unittest
from pathlib import Path

from src.experiments.final_evaluation import transfer_outcome_rows as tor

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data/models/outcome/post-migration-v1"


class TransferOutcomeRowsTest(unittest.TestCase):
    def test_rows_carry_truth_and_three_predictions(self) -> None:
        rows = [{"attempt_id": f"a{i}", "workload": "stable", "phase": "early",
                 "destination_load_level": 0.45, "direction": "c1->c2", "repeat": "r01",
                 "u_dst_post": 0.6, "analytical_u_dst_post": 0.5} for i in range(3)]
        fake = {"analytical": lambda rs: {"u_dst_post": [r["analytical_u_dst_post"] for r in rs]},
                "hybrid_xgb": lambda rs: {"u_dst_post": [0.58] * len(rs)}}
        out = tor.outcome_rows(rows, fake)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0]["pred_analytical"], 0.5)
        self.assertEqual(out[0]["pred_hybrid_xgb"], 0.58)
        self.assertEqual(out[0]["u_dst_post"], 0.6)

    def test_no_rows_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            tor.outcome_rows([], {})

    def test_frozen_artifact_loads_the_three_roles(self) -> None:
        predictors = tor.frozen_predictors(MODEL)
        self.assertEqual(sorted(predictors), ["analytical", "direct_xgb", "hybrid_xgb"])


if __name__ == "__main__":
    unittest.main()
