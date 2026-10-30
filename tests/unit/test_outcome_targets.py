"""Outcome targets: post-window means, derived r95_post and i_post, no clamping."""
from __future__ import annotations

import math
import unittest

from outcome_testdata import views
from src.experiments.migration import fixtures as fx
from src.twin.outcome.targets import (
    ALL_TARGETS,
    BASE_TARGETS,
    OutcomeValues,
    derive,
    target_schema,
    targets_from_window,
)


class TargetsTest(unittest.TestCase):
    def test_post_window_mean(self) -> None:
        after = views(fx.make_window([{"c1": 0.6, "c2": 0.9}, {"c1": 0.7, "c2": 1.1}], 10))
        values = targets_from_window(after, "c1", "c2")
        self.assertAlmostEqual(values.u_src_post, 0.65)
        self.assertAlmostEqual(values.u_dst_post, 1.0)
        self.assertAlmostEqual(values.r_dst95_post, 0.7)

    def test_derived_targets(self) -> None:
        v = OutcomeValues(0.4, 1.2, 0.8, 0.6)
        self.assertEqual(v.r95_post, 0.8)
        self.assertAlmostEqual(v.i_post, 0.4)
        self.assertEqual(set(v.to_dict()), set(ALL_TARGETS))
        self.assertEqual(derive(v.to_dict()), {"r95_post": 0.8, "i_post": v.i_post})

    def test_utilization_above_one_is_kept(self) -> None:
        self.assertEqual(OutcomeValues(0.2, 1.35, 0.7, 0.7).u_dst_post, 1.35)

    def test_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            OutcomeValues(-0.1, 0.5, 0.7, 0.7)
        with self.assertRaises(ValueError):
            OutcomeValues(0.1, math.nan, 0.7, 0.7)
        with self.assertRaises(ValueError):
            targets_from_window([], "c1", "c2")

    def test_schema(self) -> None:
        schema = target_schema()
        self.assertEqual(schema["target_schema"], "outcome-targets-v1")
        self.assertEqual(schema["base_targets"], list(BASE_TARGETS))


if __name__ == "__main__":
    unittest.main()
