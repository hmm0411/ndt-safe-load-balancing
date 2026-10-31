"""i_post is derived from the predicted utilizations with the Week 11 definition."""
from __future__ import annotations

import unittest

from src.experiments.migration.metrics import imbalance
from src.twin.outcome import analytical
from src.twin.outcome.targets import OutcomeValues


class ImbalanceTest(unittest.TestCase):
    def test_two_controllers(self) -> None:
        v = OutcomeValues(0.9, 0.5, 0.7, 0.7)
        self.assertAlmostEqual(v.i_post, abs(0.9 - 0.5) / 2)
        self.assertAlmostEqual(v.i_post, imbalance({"c1": 0.9, "c2": 0.5}))

    def test_balanced_is_zero(self) -> None:
        self.assertEqual(OutcomeValues(0.6, 0.6, 0.7, 0.7).i_post, 0.0)

    def test_analytical_imbalance_follows_its_utilizations(self) -> None:
        pred = analytical.predict(analytical.AnalyticalInputs(
            1000.0, 500.0, 200.0, 1000.0, 1000.0, 0.7, 0.7))
        self.assertAlmostEqual(pred.i_post, abs(pred.u_src_post - pred.u_dst_post) / 2)


if __name__ == "__main__":
    unittest.main()
