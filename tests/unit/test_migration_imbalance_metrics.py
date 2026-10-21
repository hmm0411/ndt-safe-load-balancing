"""imbalance / max_utilization of the Week 11 outcome."""
from __future__ import annotations

import math
import unittest

from src.experiments.migration.metrics import imbalance, max_utilization


class ImbalanceTest(unittest.TestCase):
    def test_two_controllers_is_half_the_gap(self) -> None:
        self.assertAlmostEqual(imbalance({"c1": 0.8, "c2": 0.4}), 0.2)

    def test_balanced_is_zero(self) -> None:
        self.assertEqual(imbalance({"c1": 0.6, "c2": 0.6}), 0.0)

    def test_four_controllers(self) -> None:
        u = {"c1": 0.9, "c2": 0.5, "c3": 0.5, "c4": 0.5}
        mean = 0.6
        expected = math.sqrt(((0.3) ** 2 + 3 * (0.1) ** 2) / 4)
        self.assertAlmostEqual(sum(u.values()) / 4, mean)
        self.assertAlmostEqual(imbalance(u), expected)

    def test_ids_do_not_matter(self) -> None:
        self.assertAlmostEqual(imbalance({"a": 0.8, "b": 0.4}), imbalance({"c2": 0.4, "c1": 0.8}))

    def test_overload_is_not_clamped(self) -> None:
        self.assertAlmostEqual(max_utilization({"c1": 1.1, "c2": 0.4}), 1.1)
        self.assertAlmostEqual(imbalance({"c1": 1.2, "c2": 0.4}), 0.4)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            imbalance({})
        with self.assertRaises(ValueError):
            imbalance({"c1": float("nan"), "c2": 0.4})
        with self.assertRaises(ValueError):
            max_utilization({"c1": float("inf")})


if __name__ == "__main__":
    unittest.main()
