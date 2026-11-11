from __future__ import annotations

import json
import unittest

from src.twin.decision.safety_metrics import (
    ReplayRecord,
    compute_metrics,
    make_rate,
    wilson,
)


def rec(name: str, accepted: bool, u: float, r: float | None = None) -> ReplayRecord:
    return ReplayRecord(name, accepted, u, r)


def sample() -> list[ReplayRecord]:
    unsafe = [(1.20, False), (1.10, False), (1.50, False), (1.01, True)]
    safe = [(0.90, False)] + [(0.80 + 0.02 * i, True) for i in range(5)]
    rows = [rec(f"u{i}", acc, u) for i, (u, acc) in enumerate(unsafe)]
    return rows + [rec(f"s{i}", acc, u) for i, (u, acc) in enumerate(safe)]


class WilsonTest(unittest.TestCase):
    def test_known_intervals(self) -> None:
        low, high = wilson(0, 3)
        self.assertAlmostEqual(low, 0.0, places=9)
        self.assertAlmostEqual(high, 0.5615, places=3)
        low, high = wilson(3, 3)
        self.assertAlmostEqual(low, 0.4385, places=3)
        self.assertAlmostEqual(high, 1.0, places=9)
        low, high = wilson(1, 2)
        self.assertAlmostEqual(low, 0.0945, places=3)
        self.assertAlmostEqual(high, 0.9055, places=3)

    def test_bad_arguments(self) -> None:
        for args in ((1, 0), (-1, 5), (6, 5)):
            with self.assertRaises(ValueError):
                wilson(*args)

    def test_empty_denominator_has_no_value(self) -> None:
        rate = make_rate(0, 0)
        self.assertEqual((rate.value, rate.low, rate.high), (None, None, None))


class CountsTest(unittest.TestCase):
    def test_confusion_counts_and_rates(self) -> None:
        m = compute_metrics(sample(), u_limit=1.0)
        self.assertEqual((m.records, m.unsafe, m.accepted), (10, 4, 6))
        self.assertEqual((m.unsafe_rejected, m.unsafe_accepted), (3, 1))
        self.assertEqual((m.safe_rejected, m.safe_accepted), (1, 5))
        self.assertAlmostEqual(m.ucrr.value or -1, 3 / 4)
        self.assertAlmostEqual(m.uar.value or -1, 1 / 4)
        self.assertAlmostEqual(m.frr.value or -1, 1 / 6)
        self.assertAlmostEqual(m.sap.value or -1, 5 / 6)
        self.assertAlmostEqual(m.doar.value or -1, 1 / 6)

    def test_ucrr_and_uar_add_up_to_one(self) -> None:
        m = compute_metrics(sample(), u_limit=1.0)
        self.assertAlmostEqual((m.ucrr.value or 0) + (m.uar.value or 0), 1.0)

    def test_without_response_limit_doar_is_one_minus_sap(self) -> None:
        m = compute_metrics(sample(), u_limit=1.0)
        self.assertAlmostEqual((m.doar.value or 0) + (m.sap.value or 0), 1.0)

    def test_boundary_is_inclusive(self) -> None:
        rows = [rec("a", True, 1.0), rec("b", True, 1.0 + 1e-12), rec("c", True, 1.001)]
        m = compute_metrics(rows, u_limit=1.0)
        self.assertEqual((m.safe_accepted, m.unsafe_accepted), (2, 1))

    def test_intervals_come_with_the_counts(self) -> None:
        m = compute_metrics(sample(), u_limit=1.0)
        self.assertEqual((m.uar.numerator, m.uar.denominator), (1, 4))
        self.assertLess(m.uar.low or 1, m.uar.value or 0)
        self.assertGreater(m.uar.high or 0, m.uar.value or 1)

    def test_no_unsafe_case_leaves_unsafe_rates_empty(self) -> None:
        m = compute_metrics([rec("a", True, 0.5), rec("b", False, 0.6)], u_limit=1.0)
        self.assertIsNone(m.ucrr.value)
        self.assertIsNone(m.uar.value)
        self.assertAlmostEqual(m.frr.value or -1, 0.5)

    def test_empty_input_does_not_invent_zeros(self) -> None:
        m = compute_metrics([], u_limit=1.0)
        self.assertEqual(m.records, 0)
        for rate in (m.ucrr, m.uar, m.frr, m.sap, m.doar):
            self.assertIsNone(rate.value)


class ResponseTest(unittest.TestCase):
    def test_not_configured_without_r95_max(self) -> None:
        m = compute_metrics(sample(), u_limit=1.0)
        self.assertIsNone(m.rvr)
        self.assertEqual(m.rvr_status, "NOT_CONFIGURED")

    def test_response_violation_makes_a_safe_u_unsafe(self) -> None:
        rows = [rec("a", True, 0.9, 2.0), rec("b", True, 0.9, 0.5), rec("c", False, 0.9, 3.0)]
        m = compute_metrics(rows, u_limit=1.0, r_limit=1.0)
        self.assertEqual((m.unsafe, m.unsafe_accepted, m.unsafe_rejected), (2, 1, 1))
        assert m.rvr is not None
        self.assertEqual((m.rvr.numerator, m.rvr.denominator), (1, 2))
        self.assertEqual(m.doar.numerator, 0)

    def test_missing_observed_response_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            compute_metrics([rec("a", True, 0.9)], u_limit=1.0, r_limit=1.0)


class RecordTest(unittest.TestCase):
    def test_invalid_records_are_refused(self) -> None:
        for args in (("", True, 0.5), ("a", True, float("nan")), ("a", True, -0.1)):
            with self.assertRaises(ValueError):
                ReplayRecord(*args)
        with self.assertRaises(ValueError):
            ReplayRecord("a", True, 0.5, float("inf"))

    def test_to_dict_is_json_serialisable(self) -> None:
        data = compute_metrics(sample(), u_limit=1.0).to_dict()
        self.assertEqual(json.loads(json.dumps(data))["unsafe"], 4)


if __name__ == "__main__":
    unittest.main()
