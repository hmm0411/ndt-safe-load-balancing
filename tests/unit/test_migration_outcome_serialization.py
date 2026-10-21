"""MigrationOutcomeRecord round-trips through JSON without losing fields."""
from __future__ import annotations

import json
import unittest

from src.experiments.migration import fixtures as fx
from src.experiments.migration.metrics import compute_outcome, summarize_quality
from src.experiments.migration.schema import (
    SCHEMA_VERSION,
    AttemptMetadata,
    MigrationAction,
    MigrationOutcomeRecord,
    MigrationTransactionView,
    StateSnapshotView,
)


def build_record(action_type: str = "MIGRATE") -> MigrationOutcomeRecord:
    before = [
        StateSnapshotView.from_twin_state(x)
        for x in fx.make_window([{"c1": 0.8, "c2": 0.45}] * 5, consistency_score=1.0)
    ]
    after = [
        StateSnapshotView.from_twin_state(x)
        for x in fx.make_window([{"c1": 0.72, "c2": 0.53}] * 5, 10)
    ]
    if action_type == "MIGRATE":
        action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1", selection_rule="random")
        tx: MigrationTransactionView | None = MigrationTransactionView.from_response(
            fx.committed_response()
        )
    else:
        action = MigrationAction("NO_OP", "c1", "c2")
        tx = None
    metadata = AttemptMetadata(
        attempt_id="mig-stable-middle-dst45-c1-c2-r01",
        workload="stable",
        phase="middle",
        destination_load_level=0.45,
        source_controller="c1",
        target_controller="c2",
        repeat="r01",
        seed=9201,
        topology="2C20S",
        git_commit="0" * 40,
        config_sha256="f" * 64,
        capacity_version="capacity-v3",
        quality_version="quality-score-v1",
        started_at=fx.T0.isoformat(),
    )
    return MigrationOutcomeRecord(
        metadata=metadata,
        before_state=tuple(before),
        action=action,
        transaction=tx,
        after_state=tuple(after),
        outcome=compute_outcome(before, after, action, tx),
        quality=summarize_quality(before, after),
    )


class SerializationTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        for action_type in ("MIGRATE", "NO_OP"):
            record = build_record(action_type)
            text = json.dumps(record.to_dict(), sort_keys=True)
            again = MigrationOutcomeRecord.from_dict(json.loads(text))
            self.assertEqual(again, record)
            self.assertEqual(json.dumps(again.to_dict(), sort_keys=True), text)

    def test_quality_fields_are_serialized(self) -> None:
        data = build_record().to_dict()
        quality = data["before_state"][0]["quality"]
        self.assertIn("consistency_score", quality)
        self.assertIn("quality_score", quality)
        self.assertEqual(data["quality"]["required_window"], 5)
        self.assertTrue(data["quality"]["post_window_complete"])

    def test_no_week12_prediction_fields(self) -> None:
        text = json.dumps(build_record().to_dict()).lower()
        for word in ("predict", "candidate_score", "safety_score", "upper_bound"):
            self.assertNotIn(word, text)

    def test_schema_version_is_enforced(self) -> None:
        data = build_record().to_dict()
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)
        data["schema_version"] = "migration-outcome-v0"
        with self.assertRaises(ValueError):
            MigrationOutcomeRecord.from_dict(data)

    def test_migrate_needs_transaction_and_noop_has_none(self) -> None:
        record = build_record()
        data = record.to_dict()
        data["transaction"] = None
        with self.assertRaises(ValueError):
            MigrationOutcomeRecord.from_dict(data)


if __name__ == "__main__":
    unittest.main()
