import unittest

from src.experiments.workloads.burst import BurstWorkload
from src.experiments.workloads.gradual import GradualWorkload
from src.experiments.workloads.oscillating import OscillatingWorkload
from src.experiments.workloads.stable import StableWorkload


class StablePhaseTests(unittest.TestCase):
    def test_default_keeps_single_phase(self) -> None:
        profile = StableWorkload(0.6)
        self.assertEqual({profile.target_at(t).phase for t in (0, 30, 59)}, {"stable"})

    def test_three_phases_by_time(self) -> None:
        profile = StableWorkload(0.6, duration_seconds=60.0)
        got = [profile.target_at(t).phase for t in (0, 19, 20, 39, 40, 59)]
        self.assertEqual(got, ["early", "early", "middle", "middle", "late", "late"])

    def test_rejects_non_positive_duration(self) -> None:
        with self.assertRaises(ValueError):
            StableWorkload(0.6, duration_seconds=0.0)


class OscillatingPhaseTests(unittest.TestCase):
    def test_default_keeps_single_phase(self) -> None:
        self.assertEqual(OscillatingWorkload().target_at(3.0).phase, "oscillating")

    def test_phases_follow_the_wave(self) -> None:
        profile = OscillatingWorkload(period_seconds=20.0, phased=True)
        self.assertEqual(profile.target_at(1.0).phase, "rising")
        self.assertEqual(profile.target_at(5.0).phase, "peak")
        self.assertEqual(profile.target_at(8.0).phase, "falling")
        self.assertEqual(profile.target_at(15.0).phase, "falling")
        self.assertEqual(profile.target_at(19.0).phase, "rising")


class ExistingPhaseMappingTests(unittest.TestCase):
    def test_gradual_phases(self) -> None:
        profile = GradualWorkload()
        self.assertEqual(profile.target_at(5.0).phase, "u40")
        self.assertEqual(profile.target_at(25.0).phase, "u60")
        self.assertEqual(profile.target_at(55.0).phase, "u95")

    def test_burst_phases(self) -> None:
        profile = BurstWorkload()
        self.assertEqual(profile.target_at(5.0).phase, "baseline_1")
        self.assertEqual(profile.target_at(20.0).phase, "burst_1")
        self.assertEqual(profile.target_at(30.0).phase, "recovery")


if __name__ == "__main__":
    unittest.main()
