from __future__ import annotations

from datetime import datetime, timezone

from src.schemas.snapshot import SnapshotQuality


class SnapshotValidator:
    def __init__(
        self,
        expected_controller_ids: set[str],
        expected_switch_ids: set[str],
        freshness_max_age_ms: float,
    ):
        self.expected_controller_ids = expected_controller_ids
        self.expected_switch_ids = expected_switch_ids
        self.freshness_max_age_ms = float(freshness_max_age_ms)

    def validate(
        self,
        controllers,
        switches,
        ownership_mapping,
        role_matrix=None,
        now=None,
    ):
        now = now or datetime.now(timezone.utc)
        role_matrix = role_matrix or {}
        missing: list[str] = []
        stale: list[str] = []
        conflicts: list[str] = []

        controller_ids = {item.controller_id for item in controllers}
        for controller_id in sorted(self.expected_controller_ids - controller_ids):
            missing.append(f"controller:{controller_id}:telemetry")

        for switch_id in sorted(self.expected_switch_ids):
            if switch_id not in ownership_mapping:
                missing.append(f"switch:{switch_id}:ownership")

        max_age_ms = 0.0
        for sample in controllers:
            age_ms = max(0.0, (now - sample.observed_at).total_seconds() * 1000.0)
            max_age_ms = max(max_age_ms, age_ms)
            if age_ms > self.freshness_max_age_ms:
                stale.append(f"controller:{sample.controller_id}")

        for switch_id, owner in ownership_mapping.items():
            if switch_id not in self.expected_switch_ids:
                conflicts.append(f"unknown_switch:{switch_id}")
                continue
            if owner not in self.expected_controller_ids:
                conflicts.append(f"invalid_owner:{switch_id}:{owner}")
                continue

            # In the current 2-controller MVP every switch must stay connected to both
            # controllers; the owner must be MASTER and every peer must be SLAVE.
            for controller_id in sorted(self.expected_controller_ids):
                observed = role_matrix.get((switch_id, controller_id))
                if observed is None:
                    conflicts.append(f"role_state_missing:{switch_id}:{controller_id}")
                    continue
                if not observed.get("connected", False):
                    conflicts.append(f"disconnected:{switch_id}:{controller_id}")

                actual_role = str(observed.get("role", "UNKNOWN"))
                expected_role = "MASTER" if controller_id == owner else "SLAVE"
                if actual_role != expected_role:
                    conflicts.append(
                        f"role_mismatch:{switch_id}:{controller_id}:"
                        f"expected_{expected_role}:actual_{actual_role}"
                    )

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
            max_data_age_ms=max_age_ms,
        )
