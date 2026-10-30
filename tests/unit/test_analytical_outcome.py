"""Analytical outcome model: load conservation, no clamping, response persistence."""
from __future__ import annotations

import unittest

from outcome_testdata import CAPACITY, pre_window, views
from src.twin.outcome import analytical
from src.twin.outcome.analytical import AnalyticalInputs
from src.twin.outcome.features import ActionSpec


def inputs(src: float = 840.0, dst: float = 500.0, switch: float = 224.1,
           c_src: float = 933.0, c_dst: float = 933.0) -> AnalyticalInputs:
    return AnalyticalInputs(src, dst, switch, c_src, c_dst, 0.75, 0.78)


class AnalyticalTest(unittest.TestCase):
    def test_zero_share_leaves_utilization_unchanged(self) -> None:
        pred = analytical.predict(inputs(switch=0.0))
        self.assertAlmostEqual(pred.u_src_post, 840.0 / 933.0)
        self.assertAlmostEqual(pred.u_dst_post, 500.0 / 933.0)

    def test_load_moves_from_source_to_destination(self) -> None:
        pred = analytical.predict(inputs())
        self.assertLess(pred.u_src_post, 840.0 / 933.0)
        self.assertGreater(pred.u_dst_post, 500.0 / 933.0)
        self.assertAlmostEqual(pred.u_src_post, (840.0 - 224.1) / 933.0)
        self.assertAlmostEqual(pred.u_dst_post, (500.0 + 224.1) / 933.0)

    def test_destination_above_one_is_not_clamped(self) -> None:
        pred = analytical.predict(inputs(dst=900.0, switch=300.0))
        self.assertAlmostEqual(pred.u_dst_post, 1200.0 / 933.0)
        self.assertGreater(pred.u_dst_post, 1.0)

    def test_source_rate_is_floored_at_zero(self) -> None:
        self.assertEqual(analytical.predict(inputs(src=100.0, switch=150.0)).u_src_post, 0.0)

    def test_response_is_persistence(self) -> None:
        pred = analytical.predict(inputs())
        self.assertEqual((pred.r_src95_post, pred.r_dst95_post), (0.75, 0.78))
        self.assertEqual(pred.r95_post, 0.78)

    def test_inputs_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            inputs(c_dst=0.0)
        with self.assertRaises(ValueError):
            inputs(switch=-1.0)
        with self.assertRaises(ValueError):
            inputs(src=float("inf"))

    def test_inputs_from_window(self) -> None:
        window = views(pre_window([0.8] * 5, [0.45] * 5, share=0.25))
        got = analytical.inputs_from_window(window, ActionSpec("s1", "c1", "c2"), "c1", "c2")
        self.assertAlmostEqual(got.src_rate_pps, 0.8 * CAPACITY)
        self.assertAlmostEqual(got.switch_rate_pps, 0.8 * CAPACITY * 0.25)
        self.assertEqual(got.dst_capacity_pps, CAPACITY)
        no_action = analytical.inputs_from_window(window, None, "c1", "c2")
        self.assertEqual(no_action.switch_rate_pps, 0.0)

    def test_short_window_is_refused(self) -> None:
        window = views(pre_window([0.8] * 3, [0.45] * 3))
        with self.assertRaises(ValueError):
            analytical.inputs_from_window(window, None, "c1", "c2")


if __name__ == "__main__":
    unittest.main()
