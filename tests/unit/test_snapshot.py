import unittest
from datetime import datetime, timezone
from src.schemas.telemetry import ControllerTelemetry
from src.telemetry.snapshot_validator import SnapshotValidator

def sample(cid):
    now = datetime.now(timezone.utc)
    return ControllerTelemetry(
        cid, now, now, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    )

class SnapshotTests(unittest.TestCase):
    def test_valid_quiet_state(self):
        validator = SnapshotValidator({"c1", "c2"}, {"s1", "s2"}, 5000)
        quality = validator.validate(
            [sample("c1"), sample("c2")],
            [],
            {"s1": "c1", "s2": "c2"},
        )
        self.assertTrue(quality.valid)

if __name__ == "__main__":
    unittest.main()
