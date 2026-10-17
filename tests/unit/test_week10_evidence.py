"""Week 10 evidence: Quality Score replay keeps the Hard Gate independent of the weights."""
import unittest
from datetime import datetime, timedelta, timezone

from scripts.analysis.week10_evidence import QUALITY_VARIANTS, _histogram, replay_quality

T0 = datetime(2026, 10, 16, tzinfo=timezone.utc)
THRESHOLDS = {"age_ms": {"good": 1000.0, "hard": 2500.0}, "jitter_ms": {"good": 630.0, "hard": 1000.0},
              "collection_ms": {"good": 568.579, "hard": 1000.0}}
TELEMETRY = {"freshness_max_age_ms": 2500,
             "twin_quality": {"twinning_window_seconds": 60, "min_twinning_rate": 0.95,
                              "min_completeness_ratio": 1.0}}


def snapshot(second: int, *, valid: bool = True, consistency: float = 1.0, lag_ms: int = 700) -> dict:
    t = T0 + timedelta(seconds=second)
    controllers = [{"controller_id": c, "observed_at": (t - timedelta(milliseconds=lag_ms)).isoformat(),
                    "ingested_at": (t - timedelta(milliseconds=200)).isoformat()} for c in ("c1", "c2")]
    return {"snapshot_id": f"s{second}", "created_at": t.isoformat(), "controllers": controllers,
            "switches": [{"switch_id": "s1"}], "ownership": [{"switch_id": "s1"}],
            "quality": {"valid": valid, "consistency_score": consistency}}


class QualityReplayTest(unittest.TestCase):
    def test_hard_gate_does_not_depend_on_weights(self) -> None:
        snaps = [snapshot(s) for s in range(30)]
        snaps += [snapshot(30, valid=False, consistency=0.975), snapshot(31, lag_ms=3000)]
        rows = replay_quality(snaps, QUALITY_VARIANTS, THRESHOLDS, TELEMETRY, {"c1", "c2"}, {"s1"})
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(r["hard_valid"] for r in rows[:30]))
        self.assertFalse(rows[30]["hard_valid"])  # role inconsistency: invalid whatever the score
        self.assertFalse(rows[31]["hard_valid"])  # stale telemetry, and the twinning rate dropped
        self.assertGreater(rows[30]["base"], 0.9)
        # the variants rank the same snapshot differently but never touch hard_valid
        self.assertNotEqual(rows[30]["consistency_heavy"], rows[30]["freshness_heavy"])

    def test_startup_snapshots_do_not_count_against_the_twinning_window(self) -> None:
        snaps = [snapshot(s, valid=False) for s in range(5)] + [snapshot(s) for s in range(5, 40)]
        rows = replay_quality(snaps, QUALITY_VARIANTS, THRESHOLDS, TELEMETRY, {"c1", "c2"}, {"s1"})
        self.assertFalse(any(r["hard_valid"] for r in rows[:5]))
        self.assertTrue(all(r["hard_valid"] for r in rows[5:]))

    def test_histogram_counts_every_value(self) -> None:
        counts, edges = _histogram([0.0, 0.5, 1.0, 1.0], 4)
        self.assertEqual(sum(counts), 4)
        self.assertEqual(len(edges), 5)


if __name__ == "__main__":
    unittest.main()
