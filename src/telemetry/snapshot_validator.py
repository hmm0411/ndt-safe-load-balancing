from datetime import datetime, timezone
from src.schemas.snapshot import SnapshotQuality

class SnapshotValidator:
    def __init__(self, expected_controller_ids: set[str], expected_switch_ids: set[str],
                 freshness_max_age_ms: float):
        self.expected_controller_ids = expected_controller_ids
        self.expected_switch_ids = expected_switch_ids
        self.freshness_max_age_ms = freshness_max_age_ms

    def validate(self, controllers, switches, ownership_mapping, now=None):
        now = now or datetime.now(timezone.utc)
        missing, stale, conflicts = [], [], []

        controller_ids = {x.controller_id for x in controllers}
        for cid in sorted(self.expected_controller_ids - controller_ids):
            missing.append(f"controller:{cid}:telemetry")

        for sid in self.expected_switch_ids:
            if sid not in ownership_mapping:
                missing.append(f"switch:{sid}:ownership")

        max_age = 0.0
        for sample in controllers:
            age = max(0.0, (now - sample.observed_at).total_seconds() * 1000)
            max_age = max(max_age, age)
            if age > self.freshness_max_age_ms:
                stale.append(f"controller:{sample.controller_id}")

        for sid, owner in ownership_mapping.items():
            if owner not in self.expected_controller_ids:
                conflicts.append(f"invalid_owner:{sid}:{owner}")

        fresh = not stale
        complete = not missing
        consistent = not conflicts
        return SnapshotQuality(
            fresh=fresh,
            complete=complete,
            consistent=consistent,
            valid=fresh and complete and consistent,
            missing_fields=missing,
            stale_sources=stale,
            conflicts=conflicts,
            max_data_age_ms=max_age,
        )
