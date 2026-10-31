"""Fail-closed front of the post-migration outcome predictor (Week 12, Day 2)."""

from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.experiments.migration.schema import MigrationAction
from src.twin.outcome import predictor as p

EXPECTED_FIELDS = [
    "snapshot_id", "switch_id", "source_controller", "target_controller",
    "model_name", "model_version",
    "predicted_src_utilization", "predicted_dst_utilization",
    "predicted_src_response_p95_ms", "predicted_dst_response_p95_ms",
    "predicted_worst_response_p95_ms", "predicted_imbalance",
    "dst_utilization_margin", "response_margin_ms",
    "inference_time_ms", "quality_score", "consistency_score", "status", "reason",
]  # fmt: skip


def _state() -> dict[str, Any]:
    return {
        "snapshot_id": "snap-1",
        "quality": {"valid": True, "quality_score": 0.9, "consistency_score": 1.0},
        "controllers": [
            {"controller_id": "c1", "safe_capacity_pps": 932.9},
            {"controller_id": "c2", "safe_capacity_pps": 933.1},
        ],
        "switches": [{"switch_id": "s7", "controller_id": "c1", "control_load_share": 0.3}],
        "ownership": [{"switch_id": "s7", "owner_controller_id": "c1"}],
    }


def _action() -> MigrationAction:
    return MigrationAction(
        action_type="MIGRATE", source_controller="c1", target_controller="c2", switch_id="s7"
    )


def _fake_artifact(root: Path, capacity_version: str = "cap@x") -> Path:
    for entry in p.ARTIFACT_ENTRIES:
        path = root / entry
        if entry.endswith(".json"):
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir()
    manifest = {
        "model_version": p.MODEL_VERSION,
        "feature_schema": p.FEATURE_SCHEMA,
        "target_schema": p.TARGET_SCHEMA,
        "capacity_version": capacity_version,
    }
    (root / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class OutcomePredictionContractTest(unittest.TestCase):
    def test_fields_follow_plan_section_50_and_have_no_safe(self) -> None:
        names = [f.name for f in dataclasses.fields(p.PostMigrationOutcomePrediction)]
        self.assertEqual(names, EXPECTED_FIELDS)
        self.assertNotIn("safe", names)

    def test_prediction_is_frozen(self) -> None:
        pred = p.PostMigrationOutcomePredictor().predict(_state(), _action())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pred.status = p.READY  # type: ignore[misc]


class NotReadyReasonTest(unittest.TestCase):
    def _reason(
        self,
        state: dict[str, Any],
        action: MigrationAction | None = None,
        history: int = 5,
        predictor: p.PostMigrationOutcomePredictor | None = None,
    ) -> str:
        pred = (predictor or p.PostMigrationOutcomePredictor()).predict(
            state, action or _action(), history_size=history
        )
        self.assertEqual(pred.status, p.NOT_READY)
        self.assertIsNotNone(pred.reason)
        return str(pred.reason)

    def test_numbers_are_none_not_zero(self) -> None:
        state = _state()
        state["quality"]["valid"] = False
        pred = p.PostMigrationOutcomePredictor().predict(state, _action(), history_size=5)
        for name in EXPECTED_FIELDS[6:15]:
            self.assertIsNone(getattr(pred, name), name)

    def test_invalid_snapshot(self) -> None:
        state = _state()
        state["quality"]["valid"] = False
        self.assertEqual(self._reason(state), p.INVALID_SNAPSHOT)

    def test_missing_quality_is_invalid(self) -> None:
        state = _state()
        del state["quality"]
        self.assertEqual(self._reason(state), p.INVALID_SNAPSHOT)

    def test_no_op_is_not_a_what_if(self) -> None:
        action = MigrationAction(
            action_type="NO_OP", source_controller="c1", target_controller="c2"
        )
        self.assertEqual(self._reason(_state(), action), p.INVALID_ACTION)

    def test_history_shorter_than_five(self) -> None:
        self.assertEqual(self._reason(_state(), history=4), p.INSUFFICIENT_HISTORY)
        self.assertEqual(self._reason(_state(), history=0), p.INSUFFICIENT_HISTORY)

    def test_switch_not_owned_by_source(self) -> None:
        state = _state()
        state["ownership"][0]["owner_controller_id"] = "c2"
        self.assertEqual(self._reason(state), p.SWITCH_NOT_OWNED_BY_SOURCE)

    def test_switch_without_ownership_row(self) -> None:
        state = _state()
        state["ownership"] = []
        self.assertEqual(self._reason(state), p.SWITCH_NOT_OWNED_BY_SOURCE)

    def test_missing_switch_telemetry(self) -> None:
        state = _state()
        state["switches"] = []
        self.assertEqual(self._reason(state), p.MISSING_SWITCH_TELEMETRY)

    def test_switch_telemetry_without_number(self) -> None:
        state = _state()
        state["switches"][0]["control_load_share"] = None
        self.assertEqual(self._reason(state), p.MISSING_SWITCH_TELEMETRY)

    def test_missing_safe_capacity(self) -> None:
        for bad in (None, 0, -1.0, float("nan")):
            state = _state()
            state["controllers"][1]["safe_capacity_pps"] = bad
            self.assertEqual(self._reason(state), p.MISSING_CAPACITY, bad)

    def test_missing_controller_row(self) -> None:
        state = _state()
        state["controllers"] = state["controllers"][:1]
        self.assertEqual(self._reason(state), p.MISSING_CAPACITY)

    def test_no_artifact(self) -> None:
        self.assertEqual(self._reason(_state()), p.ARTIFACT_MISSING)

    def test_artifact_entry_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_artifact(Path(tmp))
            (root / "model_card.json").unlink()
            predictor = p.PostMigrationOutcomePredictor(root, "cap@x")
            reason = self._reason(_state(), predictor=predictor)
            self.assertEqual(reason, p.ARTIFACT_SCHEMA_MISMATCH)

    def test_artifact_schema_name_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_artifact(Path(tmp))
            manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
            manifest["feature_schema"] = "outcome-features-v0"
            (root / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            predictor = p.PostMigrationOutcomePredictor(root, "cap@x")
            reason = self._reason(_state(), predictor=predictor)
            self.assertEqual(reason, p.ARTIFACT_SCHEMA_MISMATCH)

    def test_capacity_version_differs_from_running_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_artifact(Path(tmp), capacity_version="cap@old")
            predictor = p.PostMigrationOutcomePredictor(root, "cap@new")
            reason = self._reason(_state(), predictor=predictor)
            self.assertEqual(reason, p.CAPACITY_VERSION_MISMATCH)

    def test_unknown_running_capacity_version_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_artifact(Path(tmp))
            predictor = p.PostMigrationOutcomePredictor(root, None)
            reason = self._reason(_state(), predictor=predictor)
            self.assertEqual(reason, p.CAPACITY_VERSION_MISMATCH)

    def test_clean_case_stays_closed_until_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fake_artifact(Path(tmp))
            predictor = p.PostMigrationOutcomePredictor(root, "cap@x")
            self.assertEqual(self._reason(_state(), predictor=predictor), p.MODEL_NOT_LOADED)

    def test_prediction_carries_ids_and_quality(self) -> None:
        pred = p.PostMigrationOutcomePredictor().predict(_state(), _action(), history_size=5)
        self.assertEqual(
            (pred.snapshot_id, pred.switch_id, pred.source_controller, pred.target_controller),
            ("snap-1", "s7", "c1", "c2"),
        )
        self.assertEqual((pred.quality_score, pred.consistency_score), (0.9, 1.0))

    def test_predict_does_not_modify_inputs(self) -> None:
        state = _state()
        before = copy.deepcopy(state)
        p.PostMigrationOutcomePredictor().predict(state, _action(), history_size=5)
        self.assertEqual(state, before)


if __name__ == "__main__":
    unittest.main()
