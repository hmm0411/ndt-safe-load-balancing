import unittest
from unittest import mock
import os

from scripts.experiments.run_forecast_hot_switch_2c20s import (
    CONTROLLER_HOSTS,
    HOT_HOST,
    HOT_SHARE,
    build_distribution,
    hot_switch_level,
)
from src.experiments.workloads.hot_switch import (
    HOT_SWITCH_LEVELS,
    HotSwitchWorkload,
)

LEVELS = (("moderate", 0.25), ("high", 0.40), ("peak", 0.55))


class HotSwitchLevelTests(unittest.TestCase):
    def test_agreed_levels(self):
        self.assertEqual(HOT_SWITCH_LEVELS, {"moderate": 0.25, "high": 0.40, "peak": 0.55})

    def test_default_keeps_week10_behaviour(self):
        point = HotSwitchWorkload().target_at(1.0)
        self.assertEqual(point.phase, "hot_switch")
        self.assertEqual(point.hot_switch_share, 0.55)

    def test_level_sets_phase_and_share(self):
        for level, share in LEVELS:
            with self.subTest(level=level):
                point = HotSwitchWorkload(level=level).target_at(3.0)
                self.assertEqual(point.phase, level)
                self.assertEqual(point.hot_switch_share, share)

    def test_unknown_level_rejected(self):
        with self.assertRaises(ValueError):
            HotSwitchWorkload(level="extreme")

    def test_distribution_hot_fraction_matches_share(self):
        for _, share in LEVELS:
            for controller_id in ("c1", "c2"):
                with self.subTest(share=share, controller=controller_id):
                    rates = build_distribution(1433.0, controller_id, 1.03, share)
                    total = sum(rates.values())
                    hot = rates[HOT_HOST[controller_id]]
                    self.assertAlmostEqual(hot / total, share, places=9)
                    self.assertEqual(hot, max(rates.values()))
                    self.assertEqual(set(rates), set(CONTROLLER_HOSTS[controller_id]))

    def test_distribution_total_independent_of_share(self):
        totals = [
            sum(build_distribution(1433.0, "c1", 1.03, s).values())
            for _, s in LEVELS
        ]
        self.assertAlmostEqual(totals[0], totals[1], places=6)
        self.assertAlmostEqual(totals[1], totals[2], places=6)

    def test_distribution_default_is_week10(self):
        self.assertEqual(
            build_distribution(1433.0, "c1", 1.03),
            build_distribution(1433.0, "c1", 1.03, HOT_SHARE),
        )

    def test_env_level(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NDT_HOT_SWITCH_LEVEL", None)
            self.assertIsNone(hot_switch_level())
            os.environ["NDT_HOT_SWITCH_LEVEL"] = "high"
            self.assertEqual(hot_switch_level(), "high")
            os.environ["NDT_HOT_SWITCH_LEVEL"] = "bogus"
            with self.assertRaises(ValueError):
                hot_switch_level()


if __name__ == "__main__":
    unittest.main()
