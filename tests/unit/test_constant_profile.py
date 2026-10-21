import unittest

from src.experiments.workloads.constant import ConstantProfile


class ConstantProfileTests(unittest.TestCase):
    def test_returns_same_level_and_hold_phase(self) -> None:
        profile = ConstantProfile(0.45)
        for t in (0.0, 10.5, 120.0):
            point = profile.target_at(t)
            self.assertEqual(point.target_utilization, 0.45)
            self.assertEqual(point.phase, "hold")
            self.assertEqual(point.elapsed_seconds, t)

    def test_rejects_invalid_target(self) -> None:
        for bad in (-0.1, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    ConstantProfile(bad)


if __name__ == "__main__":
    unittest.main()
