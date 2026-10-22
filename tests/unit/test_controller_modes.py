import unittest

from src.experiments.workloads.profile_spec import controller_modes


class ControllerModesTests(unittest.TestCase):
    def test_no_profiles_means_all_controllers_run_the_scenario(self) -> None:
        self.assertEqual(
            controller_modes({}, "gradual", ("c1", "c2")), {"c1": None, "c2": None}
        )

    def test_source_scenario_destination_constant(self) -> None:
        got = controller_modes(
            {"c1": "gradual", "c2": "constant:0.45"}, "gradual", ("c1", "c2")
        )
        self.assertEqual(got, {"c1": None, "c2": 0.45})

    def test_rejects_other_scenario_unknown_controller_and_bad_level(self) -> None:
        for profiles in ({"c1": "burst"}, {"c3": "gradual"}, {"c2": "constant:x"}):
            with self.subTest(profiles=profiles):
                with self.assertRaises(ValueError):
                    controller_modes(profiles, "gradual", ("c1", "c2"))


if __name__ == "__main__":
    unittest.main()
