import unittest

from scripts.experiments.run_forecast_hot_switch_2c20s import (
    CONTROLLER_HOSTS,
    build_distribution,
    check_achieved_utilization,
    hot_switch_controller_modes,
)


class HotSwitchProfileTests(unittest.TestCase):
    def test_no_profile_runs_hot_switch_on_both(self):
        self.assertEqual(hot_switch_controller_modes(None), {"c1": None, "c2": None})
        self.assertEqual(hot_switch_controller_modes({}), {"c1": None, "c2": None})

    def test_constant_destination(self):
        self.assertEqual(
            hot_switch_controller_modes({"c1": "hot-switch", "c2": "constant:0.45"}),
            {"c1": None, "c2": 0.45},
        )
        self.assertEqual(
            hot_switch_controller_modes({"c1": "constant:0.75", "c2": "hot-switch"}),
            {"c1": 0.75, "c2": None},
        )

    def test_invalid_profiles_rejected(self):
        bad = [
            {"c1": "constant:0.45", "c2": "constant:0.75"},
            {"c1": "gradual"},
            {"c1": "constant:abc"},
            {"c1": "constant:-1"},
            {"c1": "constant:0"},
            {"c3": "hot-switch"},
        ]
        for profiles in bad:
            with self.subTest(profiles=profiles):
                with self.assertRaises(ValueError):
                    hot_switch_controller_modes(profiles)

    def test_even_distribution_for_constant_level(self):
        for level in (0.45, 0.75):
            with self.subTest(level=level):
                rates = build_distribution(
                    1433.0, "c2", 1.03, 1.0 / len(CONTROLLER_HOSTS["c2"]), level
                )
                total = sum(rates.values())
                self.assertAlmostEqual(total * 1.03 / 1433.0, level, places=9)
                self.assertAlmostEqual(max(rates.values()), min(rates.values()), places=9)

    def test_default_target_unchanged(self):
        rates = build_distribution(1433.0, "c1", 1.03)
        self.assertAlmostEqual(sum(rates.values()) * 1.03 / 1433.0, 0.80, places=9)

    def test_achieved_check_uses_per_controller_targets(self):
        achieved = {"c1": 0.80, "c2": 0.44}
        self.assertEqual(
            check_achieved_utilization(achieved, {"c1": 0.80, "c2": 0.45}), []
        )
        self.assertEqual(len(check_achieved_utilization(achieved)), 1)


if __name__ == "__main__":
    unittest.main()
