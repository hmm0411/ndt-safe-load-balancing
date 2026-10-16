"""Quality Score (V1): piecewise components, weights, config, assessor wiring."""
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.schemas.snapshot import NetworkSnapshot, OwnershipState, SnapshotQuality
from src.schemas.telemetry import ControllerTelemetry
from src.twin.state.quality import TwinQualityAssessor
from src.twin.state.quality_score import (
    COMPONENTS,
    load_quality_config,
    piecewise,
    quality_components,
    quality_score,
    validate_config,
)

CONFIG = load_quality_config(Path("configs/quality.yaml"))
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


class PiecewiseTest(unittest.TestCase):
    def test_shape(self) -> None:
        self.assertEqual(piecewise(100, 200, 1000), 1.0)
        self.assertEqual(piecewise(200, 200, 1000), 1.0)
        self.assertAlmostEqual(piecewise(600, 200, 1000), 0.5)
        self.assertEqual(piecewise(1000, 200, 1000), 0.0)
        self.assertEqual(piecewise(math.nan, 200, 1000), 0.0)
        self.assertEqual(piecewise(math.inf, 200, 1000), 0.0)


class ScoreTest(unittest.TestCase):
    def components(self, **overrides: float) -> dict[str, float]:
        values = dict(consistency_score=1.0, completeness_ratio=1.0, age_ms=0.0,
                      jitter_ms=0.0, collection_ms=0.0, twinning_rate=1.0)
        values.update(overrides)
        return quality_components(thresholds=CONFIG["thresholds"], **values)

    def test_perfect_state_scores_one(self) -> None:
        self.assertAlmostEqual(quality_score(self.components(), CONFIG["weights"]), 1.0)

    def test_score_stays_in_unit_interval(self) -> None:
        worst = self.components(consistency_score=-1, completeness_ratio=2, age_ms=1e9,
                                jitter_ms=1e9, collection_ms=1e9, twinning_rate=5)
        score = quality_score(worst, CONFIG["weights"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_one_wrong_role_costs_weight_times_share(self) -> None:
        score = quality_score(self.components(consistency_score=0.975), CONFIG["weights"])
        self.assertAlmostEqual(score, 1.0 - CONFIG["weights"]["consistency"] * 0.025)

    def test_config_must_have_all_weights_summing_to_one(self) -> None:
        raw = {"weights": {**CONFIG["weights"], "consistency": 0.5},
               "thresholds": CONFIG["thresholds"]}
        with self.assertRaises(ValueError):
            validate_config(raw)
        raw = {"weights": {k: v for k, v in CONFIG["weights"].items() if k != "jitter"},
               "thresholds": CONFIG["thresholds"]}
        with self.assertRaises(ValueError):
            validate_config(raw)
        self.assertEqual(set(CONFIG["weights"]), set(COMPONENTS))


def controller(cid: str, lag_ms: float = 0.0) -> ControllerTelemetry:
    return ControllerTelemetry(
        controller_id=cid, observed_at=T0, ingested_at=T0 + timedelta(milliseconds=lag_ms),
        packet_in_total=0, packet_in_rate=0.0, processed_packet_in_total=0,
        processed_packet_in_rate=0.0, flow_mod_total=0, flow_mod_rate=0.0,
        process_cpu_percent=0.0, process_memory_rss_mb=0.0, response_mean_ms=0.0,
        response_p95_ms=0.0, managed_switch_count=1,
    )


def snapshot(valid: bool = True, consistency: float = 1.0, lag_ms: float = 0.0) -> NetworkSnapshot:
    return NetworkSnapshot(
        snapshot_id="snap", created_at=T0, topology_version=1, ownership_version=1,
        controllers=[controller("c1", lag_ms), controller("c2")],
        switches=[],
        ownership=[OwnershipState("s1", "c1", "MASTER", "MASTER", 1, 1)],
        quality=SnapshotQuality(fresh=True, complete=True, consistent=valid, valid=valid,
                                consistency_score=consistency),
    )


class AssessorTest(unittest.TestCase):
    def assessor(self, config: dict | None = CONFIG) -> TwinQualityAssessor:
        return TwinQualityAssessor({"c1", "c2"}, {"s1"}, 2500, 0.0, 0.0, lambda: 1.0,
                                   quality_config=config)

    def test_without_config_the_score_is_absent(self) -> None:
        q = self.assessor(None).assess(snapshot())
        self.assertIsNone(q.quality_score)
        self.assertEqual(q.quality_components, {})

    def test_score_and_components_are_reported(self) -> None:
        # The fixture has no switch telemetry, so completeness < 1 in both cases;
        # only the collection component differs between baseline and slow.
        baseline = self.assessor().assess(snapshot())
        slow = self.assessor().assess(
            snapshot(lag_ms=CONFIG["thresholds"]["collection_ms"]["hard"])
        )
        self.assertEqual(set(slow.quality_components), set(COMPONENTS))
        self.assertEqual(baseline.quality_components["collection"], 1.0)
        self.assertEqual(slow.quality_components["collection"], 0.0)
        assert baseline.quality_score is not None and slow.quality_score is not None
        self.assertAlmostEqual(
            baseline.quality_score - slow.quality_score,
            CONFIG["weights"]["collection"],
            places=5,
        )

    def test_high_score_never_makes_an_invalid_snapshot_valid(self) -> None:
        q = self.assessor().assess(snapshot(valid=False, consistency=0.975))
        self.assertFalse(q.valid)
        assert q.quality_score is not None
        self.assertGreater(q.quality_score, 0.9)
        self.assertEqual(q.consistency_score, 0.975)


if __name__ == "__main__":
    unittest.main()
