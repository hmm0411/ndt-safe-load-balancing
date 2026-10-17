"""consistency_score (V1): graded role consistency over 20 switches x 2 controllers."""
import unittest
from datetime import datetime, timezone

from src.schemas.telemetry import ControllerTelemetry
from src.telemetry.snapshot_validator import SnapshotValidator

SWITCHES = [f"s{i}" for i in range(1, 21)]
OWNER = {s: ("c1" if i < 10 else "c2") for i, s in enumerate(SWITCHES)}


def sample(controller_id: str) -> ControllerTelemetry:
    now = datetime.now(timezone.utc)
    return ControllerTelemetry(
        controller_id=controller_id, observed_at=now, ingested_at=now,
        packet_in_total=0, packet_in_rate=0.0, processed_packet_in_total=0,
        processed_packet_in_rate=0.0, flow_mod_total=0, flow_mod_rate=0.0,
        process_cpu_percent=0.0, process_memory_rss_mb=0.0, response_mean_ms=0.0,
        response_p95_ms=0.0, managed_switch_count=10,
    )


def roles() -> dict:
    return {
        (s, c): {"role": "MASTER" if OWNER[s] == c else "SLAVE", "connected": True}
        for s in SWITCHES for c in ("c1", "c2")
    }


class ConsistencyScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = SnapshotValidator({"c1", "c2"}, set(SWITCHES), 2500)

    def validate(self, role_matrix, ownership=None):
        return self.validator.validate([sample("c1"), sample("c2")], [],
                                       ownership if ownership is not None else dict(OWNER),
                                       role_matrix=role_matrix)

    def test_all_pairs_consistent(self) -> None:
        q = self.validate(roles())
        self.assertEqual((q.consistent_role_pairs, q.expected_role_pairs), (40, 40))
        self.assertEqual(q.consistency_score, 1.0)
        self.assertTrue(q.valid)

    def test_one_wrong_role_gives_39_of_40_and_invalid(self) -> None:
        r = roles()
        r[("s1", "c2")] = {"role": "MASTER", "connected": True}
        q = self.validate(r)
        self.assertAlmostEqual(q.consistency_score, 0.975)
        self.assertFalse(q.consistent)
        self.assertFalse(q.valid)  # Hard Gate unchanged

    def test_disconnected_pair_is_not_consistent(self) -> None:
        r = roles()
        r[("s5", "c1")] = {"role": "MASTER", "connected": False}
        self.assertAlmostEqual(self.validate(r).consistency_score, 39 / 40)

    def test_missing_role_state(self) -> None:
        r = roles()
        del r[("s7", "c2")]
        self.assertAlmostEqual(self.validate(r).consistency_score, 39 / 40)

    def test_switch_without_ownership_loses_both_pairs(self) -> None:
        ownership = dict(OWNER)
        del ownership["s20"]
        q = self.validate(roles(), ownership)
        self.assertAlmostEqual(q.consistency_score, 38 / 40)
        self.assertFalse(q.complete)


if __name__ == "__main__":
    unittest.main()
