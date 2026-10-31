"""Outcome metrics and the validation selection score."""
from __future__ import annotations

import math
import unittest

from src.twin.outcome.metrics import (
    DEFAULT_WEIGHTS,
    normalized_rmse,
    quantile,
    regression_metrics,
    selection_score,
)


class MetricsTest(unittest.TestCase):
    def test_regression_metrics(self) -> None:
        m = regression_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 6.0])
        self.assertEqual(m["mae"], 0.5)
        self.assertEqual(m["rmse"], 1.0)
        self.assertEqual(m["bias"], 0.5)
        self.assertAlmostEqual(m["p95_abs_error"], 1.7)
        self.assertAlmostEqual(m["r2"], 1 - 4.0 / 5.0)

    def test_constant_truth_has_no_r2(self) -> None:
        self.assertTrue(math.isnan(regression_metrics([1.0, 1.0], [1.0, 2.0])["r2"]))

    def test_quantile_linear(self) -> None:
        self.assertEqual(quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.95), 4.8)
        with self.assertRaises(ValueError):
            quantile([], 0.5)

    def test_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            regression_metrics([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            regression_metrics([math.nan], [1.0])

    def test_selection_score(self) -> None:
        truth = {t: [0.1 * i for i in range(21)] for t in DEFAULT_WEIGHTS}
        perfect = selection_score(truth, truth)
        self.assertEqual(perfect["score"], 0.0)
        shifted = {t: [v + 0.1 for v in values] for t, values in truth.items()}
        score = selection_score(truth, shifted)
        expected = normalized_rmse(0.1, truth["u_dst_post"])
        self.assertAlmostEqual(score["n_u_dst_post"], expected)
        self.assertAlmostEqual(score["score"], expected)
        with self.assertRaises(ValueError):
            selection_score(truth, truth, {"u_dst_post": 0.5})


if __name__ == "__main__":
    unittest.main()
