"""outcome-features-v1: one shared builder, fail-closed, model features chosen on train."""
from __future__ import annotations

import unittest

from outcome_testdata import pre_window, views
from src.experiments.migration import fixtures as fx
from src.experiments.migration.schema import ControllerStateView
from src.twin.outcome.features import (
    AUDIT_BY_DESIGN,
    CANDIDATE_FEATURES,
    ActionSpec,
    build_features,
    select_model_features,
    slope,
)

ACTION = ActionSpec("s1", "c1", "c2")


class FeaturesTest(unittest.TestCase):
    def test_order_and_count(self) -> None:
        window = views(pre_window([0.70, 0.75, 0.80, 0.85, 0.90], [0.45] * 5, share=0.3))
        features = build_features(window, ACTION)
        self.assertEqual(list(features), list(CANDIDATE_FEATURES))
        self.assertEqual(len(CANDIDATE_FEATURES), 29)
        self.assertAlmostEqual(features["src_u_slope"], 0.05)
        self.assertAlmostEqual(features["src_u_mean"], 0.80)
        self.assertAlmostEqual(features["src_utilization"], 0.90)
        self.assertAlmostEqual(features["switch_control_load_share"], 0.3)
        self.assertAlmostEqual(features["imbalance_before"], (0.90 - 0.45) / 2)
        self.assertEqual(features["src_memory_rss_mb"], 56.8)

    def test_slope(self) -> None:
        self.assertAlmostEqual(slope([0.0, 1.0, 2.0], [1.0, 3.0, 5.0]), 2.0)
        with self.assertRaises(ValueError):
            slope([0.0], [1.0])
        with self.assertRaises(ValueError):
            slope([1.0, 1.0], [1.0, 2.0])

    def test_fail_closed(self) -> None:
        good = pre_window([0.8] * 5, [0.45] * 5)
        with self.assertRaises(ValueError):
            build_features(views(good[:4]), ACTION)
        invalid = [fx.make_twin_state({"c1": 0.8, "c2": 0.45}, index=i, valid=(i != 2))
                   for i in range(5)]
        with self.assertRaises(ValueError):
            build_features(views(invalid), ACTION)
        with self.assertRaises(ValueError):
            build_features(views(good), ActionSpec("s15", "c1", "c2"))  # s15 is owned by c2
        no_memory = [dict(s, controllers=[{k: v for k, v in c.items()
                                           if k != "process_memory_rss_mb"}
                                          for c in s["controllers"]]) for s in good]
        with self.assertRaises(ValueError):
            build_features(views(no_memory), ACTION)
        with self.assertRaises(ValueError):
            ActionSpec("s1", "c1", "c1")

    def test_model_features_from_train(self) -> None:
        rows = [build_features(views(pre_window([u] * 5, [0.45] * 5, share=s)), ACTION)
                for u, s in ((0.7, 0.1), (0.8, 0.2), (0.9, 0.3))]
        model, audit = select_model_features(rows)
        for name in AUDIT_BY_DESIGN:
            self.assertIn(name, audit)
            self.assertNotIn(name, model)
        self.assertIn("src_utilization", model)
        self.assertEqual(audit["dst_utilization"], "constant on train")
        with self.assertRaises(ValueError):
            select_model_features([])


class StateViewMemoryTest(unittest.TestCase):
    def test_optional_memory(self) -> None:
        raw = fx.make_twin_state({"c1": 0.5, "c2": 0.5})["controllers"][0]
        view = ControllerStateView.from_dict(raw)
        self.assertEqual(view.process_memory_rss_mb, 56.8)
        self.assertEqual(view.to_dict()["process_memory_rss_mb"], 56.8)
        old = {k: v for k, v in raw.items() if k != "process_memory_rss_mb"}
        view = ControllerStateView.from_dict(old)
        self.assertIsNone(view.process_memory_rss_mb)
        self.assertNotIn("process_memory_rss_mb", view.to_dict())


if __name__ == "__main__":
    unittest.main()
