from __future__ import annotations

import math
import unittest

import numpy as np

from src.twin.forecasting.margin import (
    coverage,
    coverage_by_group,
    error_margins,
    finite_sample_quantile,
    margin_for,
    overload_confusion,
)
from src.twin.forecasting.ood import FeatureRanges, OODGate, gate_mask


class FiniteSampleQuantileTest(unittest.TestCase):
    def test_n19_returns_the_largest_element(self) -> None:
        values = np.arange(1, 20, dtype=float)  # ceil(20 * 0.95) = 19
        self.assertEqual(finite_sample_quantile(values), 19.0)

    def test_n99_returns_the_95th_element(self) -> None:
        values = np.random.default_rng(0).permutation(np.arange(1, 100, dtype=float))
        self.assertEqual(finite_sample_quantile(values), 95.0)  # ceil(100 * 0.95)

    def test_small_n_is_clipped_to_the_maximum(self) -> None:
        self.assertEqual(finite_sample_quantile([0.3, 0.1, 0.2]), 0.3)

    def test_invalid_input_raises(self) -> None:
        for bad in ([], [1.0, float("nan")]):
            with self.assertRaises(ValueError):
                finite_sample_quantile(bad)
        with self.assertRaises(ValueError):
            finite_sample_quantile([1.0], level=1.0)

    def test_coverage_on_exchangeable_data_is_at_least_95_percent(self) -> None:
        rng = np.random.default_rng(42)
        results = []
        for _ in range(300):
            errors = np.abs(rng.standard_t(df=4, size=219))
            m = finite_sample_quantile(errors[:119])
            results.append(np.mean(errors[119:] <= m))
        self.assertGreaterEqual(float(np.mean(results)), 0.95)


class ErrorMarginsTest(unittest.TestCase):
    def test_contract_fields_and_per_controller_margin(self) -> None:
        y_true = [0.5] * 40
        y_pred = [0.5 - 0.01] * 20 + [0.5 + 0.05] * 20
        ids = ["c1"] * 20 + ["c2"] * 20
        margins = error_margins(y_true, y_pred, ids)
        self.assertEqual(margins["method"], "validation_abs_error_finite_sample_q95")
        self.assertEqual(margins["alpha"], 0.05)
        self.assertEqual(margins["n_validation"], 40)
        self.assertAlmostEqual(margins["controllers"]["c1"], 0.01)
        self.assertAlmostEqual(margins["controllers"]["c2"], 0.05)
        self.assertAlmostEqual(margins["global"], 0.05)
        # Signed U - U_hat: c1 under-forecasts by 0.01, c2 over-forecasts.
        self.assertAlmostEqual(margins["one_sided_q95"], 0.01)
        self.assertAlmostEqual(margin_for(margins, "c3"), margins["global"])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            error_margins([0.1, 0.2], [0.1], ["c1", "c1"])
        with self.assertRaises(ValueError):
            error_margins([0.1, 0.2], [0.1, 0.2], ["c1"])


class CoverageTest(unittest.TestCase):
    def test_coverage_counts_errors_within_margin(self) -> None:
        self.assertAlmostEqual(coverage([1.0, 1.0, 1.0, 1.0], [1.0, 1.05, 1.2, 0.9], 0.1), 0.75)

    def test_coverage_by_group(self) -> None:
        table = coverage_by_group([0, 0, 0], [0.0, 0.5, 0.0], [0.1, 0.1, 0.1], ["a", "a", "b"])
        self.assertEqual(table["a"], {"coverage": 0.5, "samples": 2.0})
        self.assertEqual(table["b"]["coverage"], 1.0)

    def test_overload_confusion(self) -> None:
        stats = overload_confusion([1.1, 1.2, 0.5, 0.5], [1.05, 0.9, 1.0, 0.2])
        self.assertEqual(stats["missed_overload"], 1.0)
        self.assertEqual(stats["false_alarm"], 1.0)
        self.assertAlmostEqual(stats["missed_overload_rate"], 0.5)
        self.assertTrue(math.isnan(overload_confusion([0.1], [0.1])["missed_overload_rate"]))


class FeatureRangesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ranges = FeatureRanges.fit([[0.0, 10.0], [1.0, 20.0]], ["a", "b"])

    def test_tolerance_widens_by_ten_percent_of_span(self) -> None:
        self.assertEqual(self.ranges.violations([1.1, 21.0]), ())
        self.assertEqual(self.ranges.violations([1.11, 21.0]), ("a",))
        self.assertEqual(self.ranges.violations([-0.2, 8.0]), ("a", "b"))

    def test_nan_is_out_of_range(self) -> None:
        self.assertEqual(self.ranges.violations([float("nan"), 15.0]), ("a",))

    def test_round_trip_through_contract_dict(self) -> None:
        data = self.ranges.to_dict()
        self.assertEqual(data["method"], "train_min_max")
        self.assertEqual(data["features"]["b"], {"min": 10.0, "max": 20.0})
        self.assertEqual(FeatureRanges.from_dict(data, ["a", "b"]), self.ranges)

    def test_order_comes_from_feature_schema_not_json_keys(self) -> None:
        # The file is written with sort_keys=True, so JSON order is alphabetical.
        data = FeatureRanges.fit([[0.0, 10.0], [1.0, 20.0]], ["z", "a"]).to_dict()
        data["features"] = dict(sorted(data["features"].items()))
        ranges = FeatureRanges.from_dict(data, ["z", "a"])
        self.assertEqual(ranges.violations([0.5, 15.0]), ())
        with self.assertRaises(ValueError):
            FeatureRanges.from_dict(data, ["z", "b"])

    def test_matrix_out_of_range(self) -> None:
        flags = self.ranges.out_of_range([[0.5, 15.0], [5.0, 15.0]])
        self.assertEqual(flags.tolist(), [False, True])


class OODGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = OODGate(FeatureRanges.fit([[0.0], [1.0]], ["u"]))

    def test_gate_turns_on_only_at_third_consecutive_sample(self) -> None:
        decisions = [self.gate.update("c1", [5.0]) for _ in range(4)]
        self.assertEqual([d.active for d in decisions], [False, False, True, True])
        self.assertEqual(decisions[0].features, ("u",))

    def test_gate_turns_off_when_back_in_range(self) -> None:
        for _ in range(3):
            self.gate.update("c1", [5.0])
        self.assertFalse(self.gate.update("c1", [0.5]).active)
        self.assertFalse(self.gate.update("c1", [5.0]).active)

    def test_controllers_have_independent_counters(self) -> None:
        for _ in range(2):
            self.gate.update("c1", [5.0])
        self.assertFalse(self.gate.update("c2", [5.0]).active)
        self.assertTrue(self.gate.update("c1", [5.0]).active)

    def test_offline_mask_restarts_on_new_sequence(self) -> None:
        flags = [True, True, True, True, True, False, True]
        sequences = ["a", "a", "a", "b", "b", "b", "b"]
        self.assertEqual(
            gate_mask(flags, sequences).tolist(),
            [False, False, True, False, False, False, False],
        )


if __name__ == "__main__":
    unittest.main()
