"""Residual targets and Hybrid features (Day 3): r = y - y_A, 24 + 4 features, no leakage."""
from __future__ import annotations

import math
import shutil
import tempfile
import unittest
from pathlib import Path

from outcome_testdata import write_raw_attempt
from src.experiments.migration import fixtures as fx
from src.twin.outcome import dataset as ds
from src.twin.outcome import residual
from src.twin.outcome.features import AUDIT_BY_DESIGN, CANDIDATE_FEATURES, is_forbidden
from src.twin.outcome.targets import BASE_TARGETS, OutcomeValues

CONFIG = ds.load_dataset_config(Path("configs/outcome/dataset.yaml"))
# The 24 model features of migration-outcome-processed-v1: the switch counts are constant on
# train (audit_only), the capacity columns are audit_only by design.
MODEL_FEATURES = [f for f in CANDIDATE_FEATURES
                  if f not in AUDIT_BY_DESIGN + ("src_switch_count", "dst_switch_count")]


def row(observed: OutcomeValues, analytical: OutcomeValues) -> dict[str, float]:
    return {**observed.to_dict(), **analytical.to_dict("analytical_")}


class ResidualTargetTest(unittest.TestCase):
    def test_residual_is_actual_minus_analytical(self) -> None:
        rows = [row(OutcomeValues(0.62, 0.81, 0.77, 0.79), OutcomeValues(0.55, 0.85, 0.76, 0.78)),
                row(OutcomeValues(0.40, 1.10, 0.70, 0.60), OutcomeValues(0.40, 1.25, 0.75, 0.80))]
        expected = {"u_src_post": [0.07, 0.0], "u_dst_post": [-0.04, -0.15],
                    "r_src95_post": [0.01, -0.05], "r_dst95_post": [0.01, -0.20]}
        for target in BASE_TARGETS:
            values = residual.residual_values(rows, target)
            for got, want in zip(values, expected[target]):
                self.assertAlmostEqual(float(got), want)

    def test_only_base_targets_and_finite(self) -> None:
        r = row(OutcomeValues(0.6, 0.8, 0.77, 0.78), OutcomeValues(0.5, 0.9, 0.77, 0.78))
        for target in ("i_post", "r95_post", "latency"):
            with self.assertRaises(ValueError):
                residual.residual(r, target)
        r["analytical_u_dst_post"] = math.nan
        with self.assertRaises(ValueError):
            residual.residual(r, "u_dst_post")

    def test_hybrid_has_28_features(self) -> None:
        self.assertEqual(len(MODEL_FEATURES), 24)
        features = residual.hybrid_features(MODEL_FEATURES)
        self.assertEqual(len(features), 28)
        self.assertEqual(features[:24], MODEL_FEATURES)
        self.assertEqual(tuple(features[24:]), tuple(f"analytical_{t}" for t in BASE_TARGETS))
        self.assertNotIn("analytical_r95_post", features)
        self.assertNotIn("analytical_i_post", features)

    def test_no_post_action_column(self) -> None:
        features = residual.hybrid_features(MODEL_FEATURES)
        self.assertEqual([f for f in features if is_forbidden(f)], [])
        for bad in ("u_dst_post", "saturated_after", "overload_after", "workload",
                    "analytical_u_dst_post"):
            with self.assertRaises(ValueError):
                residual.hybrid_features(MODEL_FEATURES + [bad])
        with self.assertRaises(ValueError):
            residual.hybrid_features(MODEL_FEATURES + MODEL_FEATURES[:1])
        with self.assertRaises(ValueError):
            residual.hybrid_features([])

    def test_analytical_features_do_not_depend_on_the_post_window(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        try:
            rows = []
            for after in ((0.64, 0.61), (0.30, 1.20)):
                raw = tmp / f"raw-{after[1]}"
                write_raw_attempt(raw, "mig-stable-middle-dst45-c1-c2-r01", "r01", after=after)
                built, _, _ = ds.build_rows(
                    raw, {"mig-stable-middle-dst45-c1-c2-r01": "train"},
                    {"c1": fx.SAFE_CAPACITY, "c2": fx.SAFE_CAPACITY}, CONFIG,
                    ds.saturation_flags(raw))
                rows.append(built[0])
            names = residual.ANALYTICAL_FEATURES
            self.assertEqual({k: rows[0][k] for k in names}, {k: rows[1][k] for k in names})
            self.assertNotEqual(residual.residual(rows[0], "u_dst_post"),
                                residual.residual(rows[1], "u_dst_post"))
        finally:
            fx.unlock_tree(tmp)
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
