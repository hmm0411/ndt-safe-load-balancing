import unittest
from datetime import datetime, timedelta, timezone

from src.schemas.telemetry import ControllerTelemetry
from src.telemetry.snapshot_validator import SnapshotValidator


def sample(controller_id: str, observed_at=None):
    now = observed_at or datetime.now(timezone.utc)
    return ControllerTelemetry(
        controller_id=controller_id,
        observed_at=now,
        ingested_at=now,
        packet_in_total=0,
        packet_in_rate=0.0,
        processed_packet_in_total=0,
        processed_packet_in_rate=0.0,
        flow_mod_total=0,
        flow_mod_rate=0.0,
        process_cpu_percent=0.0,
        process_memory_rss_mb=0.0,
        response_mean_ms=0.0,
        response_p95_ms=0.0,
        managed_switch_count=0,
    )


def correct_roles():
    return {
        ("s1", "c1"): {"role": "MASTER", "connected": True},
        ("s1", "c2"): {"role": "SLAVE", "connected": True},
        ("s2", "c1"): {"role": "SLAVE", "connected": True},
        ("s2", "c2"): {"role": "MASTER", "connected": True},
    }


class SnapshotValidatorTests(unittest.TestCase):
    def setUp(self):
        self.validator = SnapshotValidator({"c1", "c2"}, {"s1", "s2"}, 2500)

    def test_valid_quiet_state(self):
        quality = self.validator.validate(
            [sample("c1"), sample("c2")],
            [],
            {"s1": "c1", "s2": "c2"},
            role_matrix=correct_roles(),
        )
        self.assertTrue(quality.valid)

    def test_missing_controller_is_invalid(self):
        quality = self.validator.validate(
            [sample("c1")],
            [],
            {"s1": "c1", "s2": "c2"},
            role_matrix=correct_roles(),
        )
        self.assertFalse(quality.complete)
        self.assertFalse(quality.valid)

    def test_stale_controller_is_invalid(self):
        old = datetime.now(timezone.utc) - timedelta(seconds=10)
        quality = self.validator.validate(
            [sample("c1", old), sample("c2")],
            [],
            {"s1": "c1", "s2": "c2"},
            role_matrix=correct_roles(),
        )
        self.assertFalse(quality.fresh)
        self.assertFalse(quality.valid)

    def test_role_ownership_conflict_is_invalid(self):
        roles = correct_roles()
        roles[("s1", "c1")] = {"role": "SLAVE", "connected": True}
        roles[("s1", "c2")] = {"role": "MASTER", "connected": True}
        quality = self.validator.validate(
            [sample("c1"), sample("c2")],
            [],
            {"s1": "c1", "s2": "c2"},
            role_matrix=roles,
        )
        self.assertFalse(quality.consistent)
        self.assertFalse(quality.valid)


if __name__ == "__main__":
    unittest.main()
