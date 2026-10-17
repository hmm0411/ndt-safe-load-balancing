"""RuntimeForecaster (V2): readiness, gaps, OOD gate, overload rule."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.schemas.forecasting import (
    HISTORY_GAP,
    INSUFFICIENT_HISTORY,
    INVALID_SNAPSHOT,
    MISSING_CAPACITY,
)
from src.twin.forecasting.artifact import build_feature_schema
from src.twin.forecasting.features import feature_names
from src.twin.forecasting.fixtures import MARGINS, twin_state, write_persistence_artifact
from src.twin.forecasting.ood import MONITOR_ONLY_FEATURES
from src.twin.forecasting.predictor import ForecastArtifactError, RuntimeForecaster

FORECAST_V1 = Path("data/models/forecasting/forecast-v1")


class PersistenceForecasterTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "forecast-test"
        self.dir.mkdir()
        write_persistence_artifact(self.dir)
        self.fc = RuntimeForecaster(self.dir)

    def feed(self, seconds: range | list[float], **kwargs: Any) -> list[Any]:
        result: list[Any] = []
        for s in seconds:
            result = self.fc.update(twin_state(float(s), **kwargs))
        return result

    def by_cid(self, predictions: list[Any]) -> dict[str, Any]:
        return {p.controller_id: p for p in predictions}

    def test_invalid_snapshot_is_not_ready_without_values(self) -> None:
        p = self.by_cid(self.fc.update(twin_state(0, valid=False)))["c1"]
        self.assertEqual((p.status, p.reason), ("NOT_READY", INVALID_SNAPSHOT))
        self.assertIsNone(p.predicted_utilization)
        self.assertIsNone(p.overload_risk)

    def test_ready_only_at_the_eleventh_sample(self) -> None:
        tenth = self.by_cid(self.feed(range(10)))["c1"]
        self.assertEqual((tenth.status, tenth.reason), ("NOT_READY", INSUFFICIENT_HISTORY))
        eleventh = self.by_cid(self.fc.update(twin_state(10)))["c1"]
        self.assertEqual(eleventh.status, "READY")
        self.assertAlmostEqual(eleventh.predicted_utilization, 0.6)
        self.assertAlmostEqual(eleventh.predicted_packet_in_rate, 0.6 * 1433.0)
        self.assertAlmostEqual(eleventh.error_margin_utilization, MARGINS["c1"])
        self.assertAlmostEqual(eleventh.upper_utilization_bound, 0.6 + MARGINS["c1"])
        self.assertFalse(eleventh.overload_risk)
        self.assertEqual(eleventh.horizon_seconds, 3.0)
        self.assertEqual(eleventh.quality_score, 0.97)

    def test_gap_reports_history_gap_until_refilled(self) -> None:
        self.feed(range(11))
        after_gap = self.by_cid(self.fc.update(twin_state(14)))["c1"]  # 4 s gap
        self.assertEqual(after_gap.reason, HISTORY_GAP)
        self.assertEqual(self.by_cid(self.feed(range(15, 24)))["c1"].reason, HISTORY_GAP)
        self.assertEqual(self.by_cid(self.fc.update(twin_state(24)))["c1"].status, "READY")

    def test_margin_pushes_risk_and_measured_overload_is_always_risk(self) -> None:
        near = self.by_cid(self.feed(range(11), u=(0.96, 0.5)))["c1"]
        self.assertTrue(near.overload_risk)  # 0.96 + 0.048 >= 1
        fc = RuntimeForecaster(self.dir, threshold=2.0)
        over = None
        for s in range(11):
            over = {p.controller_id: p for p in fc.update(twin_state(float(s), u=(2.1, 0.5)))}
        assert over is not None
        self.assertTrue(over["c1"].overload_risk)

    def test_hot_switch_share_turns_on_the_ood_gate_at_the_third_sample(self) -> None:
        self.feed(range(11))
        flags = [self.by_cid(self.fc.update(twin_state(float(s), share=0.54)))["c1"]
                 for s in range(11, 14)]
        self.assertEqual([p.ood_fallback for p in flags], [False, False, True])
        self.assertIn("max_switch_control_load_share", flags[-1].ood_features)

    def test_switch_count_change_after_migration_turns_on_the_gate(self) -> None:
        self.feed(range(11))
        flags = [self.by_cid(self.fc.update(twin_state(float(s), switch_count=9)))["c2"]
                 for s in range(11, 14)]
        self.assertTrue(flags[-1].ood_fallback)
        self.assertIn("managed_switch_count", flags[-1].ood_features)

    def test_missing_capacity(self) -> None:
        p = self.by_cid(self.feed(range(11), capacity=0.0))["c1"]
        self.assertEqual((p.status, p.reason), ("NOT_READY", MISSING_CAPACITY))

    def test_duplicate_snapshot_is_not_counted_twice(self) -> None:
        self.feed(range(10))
        state = twin_state(10)
        self.assertEqual(self.by_cid(self.fc.update(state))["c1"].status, "READY")
        self.assertEqual(self.by_cid(self.fc.update(state))["c1"].reason, "DUPLICATE_SNAPSHOT")

    def test_feature_hook_gets_27_values_in_schema_order(self) -> None:
        seen: list[tuple[str, list[float]]] = []
        fc = RuntimeForecaster(self.dir, on_features=lambda cid, at, x: seen.append((cid, x)))
        for s in range(11):
            fc.update(twin_state(float(s)))
        self.assertEqual({cid for cid, _ in seen}, {"c1", "c2"})
        self.assertEqual(len(seen[0][1]), 27)
        u_index = list(feature_names()).index("utilization")
        self.assertAlmostEqual(seen[0][1][u_index], 0.6)

    def test_wrong_feature_order_is_rejected(self) -> None:
        schema = build_feature_schema(feature_names())
        schema["feature_names"] = list(reversed(schema["feature_names"]))
        (self.dir / "feature_schema.json").write_text(json.dumps(schema), encoding="utf-8")
        with self.assertRaises(ForecastArtifactError):
            RuntimeForecaster(self.dir)

    def test_monitor_only_columns_are_the_audit_columns(self) -> None:
        assert self.fc.ood is not None
        self.assertEqual(self.fc.ood.ranges.monitor_only, MONITOR_ONLY_FEATURES)


@unittest.skipUnless((FORECAST_V1 / "model_card.json").is_file(), "forecast-v1 not present")
class ForecastV1ArtifactTest(unittest.TestCase):
    def test_real_artifact_is_ready_and_single_threaded(self) -> None:
        fc = RuntimeForecaster(FORECAST_V1)
        self.assertEqual(fc.model_name, "xgboost")
        self.assertEqual(fc.model.get_params()["n_jobs"], 1)
        predictions: list[Any] = []
        for s in range(11):
            predictions = fc.update(twin_state(float(s), u=(0.58, 0.58)))
        for p in predictions:
            self.assertEqual(p.status, "READY")
            assert p.predicted_utilization is not None
            self.assertTrue(math.isfinite(p.predicted_utilization))
            self.assertLess(abs(p.predicted_utilization - 0.58), 0.3)
            self.assertFalse(p.ood_fallback)


if __name__ == "__main__":
    unittest.main()
