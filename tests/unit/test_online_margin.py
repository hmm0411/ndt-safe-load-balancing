from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from typing import Any

from src.twin.calibration.feedback import ForecastFeedback, OutcomeMargin
from src.twin.calibration.online_margin import MarginTracker, OnlineMargin, TrackerConfig
from test_decision_service import FakePredictor
from decision_testdata import make_twin
from src.experiments.migration.schema import MigrationAction


class OnlineMarginTest(unittest.TestCase):
    def test_step_sizes(self) -> None:
        m = OnlineMargin(0.10, TrackerConfig(alpha=0.05, gamma=0.01))
        self.assertTrue(m.update(0.05))            # covered: shrinks by gamma * alpha
        self.assertAlmostEqual(m.value, 0.10 - 0.0005)
        self.assertFalse(m.update(0.50))           # missed: grows by gamma * (1 - alpha)
        self.assertAlmostEqual(m.value, 0.0995 + 0.0095)
        self.assertEqual((m.updates, m.misses), (2, 1))

    def test_miss_rate_tends_to_alpha(self) -> None:
        rng = random.Random(3)
        m = OnlineMargin(0.0, TrackerConfig(alpha=0.05, gamma=0.005))
        for _ in range(20000):
            m.update(rng.gauss(0.0, 0.1))
        self.assertAlmostEqual(m.misses / m.updates, 0.05, delta=0.01)
        self.assertAlmostEqual(m.value, 0.1645, delta=0.02)   # z_0.95 * sigma

    def test_disabled_never_moves_and_floor_cap(self) -> None:
        off = OnlineMargin(0.2, enabled=False)
        off.update(5.0)
        self.assertEqual((off.value, off.misses), (0.2, 1))
        low = OnlineMargin(0.0, TrackerConfig(gamma=1.0, floor=0.0))
        low.update(-1.0)
        self.assertEqual(low.value, 0.0)
        capped = OnlineMargin(0.1, TrackerConfig(gamma=1.0, cap=0.3))
        capped.update(9.0)
        self.assertEqual(capped.value, 0.3)

    def test_bad_input(self) -> None:
        for make in (lambda: TrackerConfig(alpha=0.0), lambda: TrackerConfig(gamma=0.0),
                     lambda: TrackerConfig(floor=-1.0), lambda: TrackerConfig(floor=0.5, cap=0.1)):
            with self.assertRaises(ValueError):
                make()
        with self.assertRaises(ValueError):
            OnlineMargin(float("nan"))
        with self.assertRaises(ValueError):
            OnlineMargin(0.1).update(float("inf"))

    def test_tracker_default_for_unknown_key(self) -> None:
        t = MarginTracker({"c1": 0.1}, default=0.2)
        self.assertEqual((t.value("c1"), t.value("c9")), (0.1, 0.2))


def pred(cid: str, current: float | None, predicted: float | None,
         status: str = "READY") -> Any:
    return SimpleNamespace(controller_id=cid, status=status, current_utilization=current,
                           predicted_utilization=predicted)


class ForecastFeedbackTest(unittest.TestCase):
    def test_error_is_matched_h_steps_later(self) -> None:
        fb = ForecastFeedback({"c1": 0.1}, 0.1, horizon_steps=2, mode="adaptive",
                              config=TrackerConfig(gamma=0.01))
        fb.observe([pred("c1", 0.5, 0.9)])          # forecast for step +2: 0.9
        fb.observe([pred("c1", 0.5, 0.5)])
        self.assertEqual(fb.tracker._get("c1").updates, 0)
        fb.observe([pred("c1", 1.2, 0.5)])          # actual 1.2 vs 0.9: error 0.3 > 0.1
        margin = fb.tracker._get("c1")
        self.assertEqual((margin.updates, margin.misses), (1, 1))
        self.assertAlmostEqual(fb.margin("c1"), 0.1 + 0.0095)

    def test_gap_drops_pending_forecasts(self) -> None:
        fb = ForecastFeedback({"c1": 0.1}, 0.1, 1, "adaptive", TrackerConfig())
        fb.observe([pred("c1", 0.5, 0.9)])
        fb.observe([pred("c1", None, None, status="NOT_READY")])
        fb.observe([pred("c1", 0.5, 0.5)])
        self.assertEqual(fb.tracker._get("c1").updates, 0)

    def test_fixed_and_zero(self) -> None:
        fixed = ForecastFeedback({"c1": 0.1}, 0.1, 1, "fixed", TrackerConfig(gamma=0.5))
        for _ in range(3):
            fixed.observe([pred("c1", 2.0, 0.0)])
        self.assertEqual(fixed.margin("c1"), 0.1)
        self.assertEqual(ForecastFeedback({}, 0.1, 1, "zero", TrackerConfig()).margin("c1"), 0.0)
        with self.assertRaises(ValueError):
            ForecastFeedback({}, 0.1, 1, "loose", TrackerConfig())


class OutcomeMarginTest(unittest.TestCase):
    def predict(self, wrapper: OutcomeMargin) -> Any:
        twin = make_twin().to_dict()
        return wrapper.predict_batch(twin, [MigrationAction("MIGRATE", "c1", "c2", "s1")],
                                     observe=False)[0]

    def test_modes(self) -> None:
        fixed = OutcomeMargin(FakePredictor(), "fixed", TrackerConfig())
        self.assertEqual(self.predict(fixed).dst_utilization_margin, 0.037)
        zero = OutcomeMargin(FakePredictor(), "zero", TrackerConfig())
        p = self.predict(zero)
        self.assertEqual((p.dst_utilization_margin, p.response_margin_ms), (0.0, 0.0))
        adaptive = OutcomeMargin(FakePredictor(), "adaptive", TrackerConfig(gamma=0.02))
        self.assertIsNone(adaptive.observe_outcome(0.5, 0.9))    # no prediction seen yet
        self.assertEqual(self.predict(adaptive).dst_utilization_margin, 0.037)
        self.assertFalse(adaptive.observe_outcome(0.5, 0.9))     # error 0.4 > 0.037
        self.assertAlmostEqual(self.predict(adaptive).dst_utilization_margin, 0.037 + 0.019)
        self.assertIsNone(fixed.observe_outcome(0.5, 0.9))

    def test_other_attributes_pass_through(self) -> None:
        inner = FakePredictor()
        self.assertEqual(OutcomeMargin(inner, "fixed", TrackerConfig()).dst, inner.dst)


if __name__ == "__main__":
    unittest.main()
