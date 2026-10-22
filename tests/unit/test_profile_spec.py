import unittest

from src.experiments.workloads.profile_spec import parse_profile_specs, stable_utilizations


class ParseProfileSpecsTests(unittest.TestCase):
    def test_none_and_empty_give_empty(self) -> None:
        self.assertEqual(parse_profile_specs(None), {})
        self.assertEqual(parse_profile_specs([]), {})

    def test_parses_pairs(self) -> None:
        self.assertEqual(
            parse_profile_specs(["c1=stable", "c2=constant:0.45"]),
            {"c1": "stable", "c2": "constant:0.45"},
        )

    def test_rejects_malformed_and_duplicate(self) -> None:
        for bad in (["c1"], ["=stable"], ["c1="], ["c1=stable", "c1=constant:0.5"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_profile_specs(bad)


class StableUtilizationsTests(unittest.TestCase):
    def test_no_profiles_keeps_old_behaviour(self) -> None:
        self.assertEqual(
            stable_utilizations({}, 0.60, ("c1", "c2")), {"c1": 0.60, "c2": 0.60}
        )

    def test_source_stable_destination_constant(self) -> None:
        got = stable_utilizations(
            {"c1": "stable", "c2": "constant:0.45"}, 0.60, ("c1", "c2")
        )
        self.assertEqual(got, {"c1": 0.60, "c2": 0.45})

    def test_rejects_unknown_controller_and_profile(self) -> None:
        for profiles in ({"c3": "stable"}, {"c1": "gradual"}, {"c2": "constant:abc"}):
            with self.subTest(profiles=profiles):
                with self.assertRaises(ValueError):
                    stable_utilizations(profiles, 0.60, ("c1", "c2"))


if __name__ == "__main__":
    unittest.main()
