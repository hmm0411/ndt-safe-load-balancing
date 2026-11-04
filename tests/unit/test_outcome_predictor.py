"""Post-migration outcome predictor: fail-closed front and the numeric path (Week 12, Day 3)."""

from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.experiments.migration import fixtures as fx
from src.experiments.migration.schema import MigrationAction, StateSnapshotView
from src.twin.outcome import analytical, random_forest
from src.twin.outcome import predictor as p
from src.twin.outcome.features import ActionSpec, build_features
from src.twin.outcome.residual import hybrid_features
from src.twin.outcome.targets import BASE_TARGETS

EXPECTED_FIELDS = [
    "snapshot_id", "switch_id", "source_controller", "target_controller",
    "model_name", "model_version",
    "predicted_src_utilization", "predicted_dst_utilization",
    "predicted_src_response_p95_ms", "predicted_dst_response_p95_ms",
    "predicted_worst_response_p95_ms", "predicted_imbalance",
    "dst_utilization_margin", "response_margin_ms",
    "inference_time_ms", "quality_score", "consistency_score", "status", "reason",
]  # fmt: skip

MODEL_FEATURES = ["src_utilization", "dst_utilization", "switch_control_load_share", "src_u_mean"]
CAP = "cap@x"
M_U, M_R = 0.07, 0.11


def _states(n: int = 5, **kwargs: Any) -> list[dict[str, Any]]:
    kwargs.setdefault("hot_switch", ("s1", 0.2))
    return [fx.make_twin_state({"c1": 0.8, "c2": 0.3}, index=i, **kwargs) for i in range(n)]


def _action(switch: str = "s1") -> MigrationAction:
    return MigrationAction(
        action_type="MIGRATE", source_controller="c1", target_controller="c2", switch_id=switch
    )


def _write_artifact(
    root: Path, capacity_version: str = CAP, constant: float = 0.0, family: str = "random_forest"
) -> Path:
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
    (root / "feature_schema.json").write_text(
        json.dumps({"feature_schema": p.FEATURE_SCHEMA, "model_features": MODEL_FEATURES}),
        encoding="utf-8",
    )
    (root / "model_card.json").write_text(json.dumps({"hybrid_family": family}), encoding="utf-8")
    (root / "uncertainty_margin.json").write_text(
        json.dumps({"m_U": M_U, "m_R": M_R}), encoding="utf-8"
    )
    width = len(hybrid_features(MODEL_FEATURES))
    x = np.zeros((4, width))
    for target in BASE_TARGETS:
        model = RandomForestRegressor(n_estimators=2, random_state=0)
        model.fit(x, np.full(4, constant))
        random_forest.save(model, root / "residual_models" / f"residual_rf_{target}")
    return root


def _ready_predictor(root: Path, **kwargs: Any) -> p.PostMigrationOutcomePredictor:
    predictor = p.PostMigrationOutcomePredictor(root, CAP, **kwargs)
    for state in _states()[:-1]:
        predictor.observe(state)
    return predictor


class OutcomePredictionContractTest(unittest.TestCase):
    def test_fields_follow_plan_section_50_and_have_no_safe(self) -> None:
        names = [f.name for f in dataclasses.fields(p.PostMigrationOutcomePrediction)]
        self.assertEqual(names, EXPECTED_FIELDS)
        self.assertNotIn("safe", names)

    def test_prediction_is_frozen(self) -> None:
        pred = p.PostMigrationOutcomePredictor().predict(_states(1)[0], _action())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pred.status = p.READY  # type: ignore[misc]


class NotReadyReasonTest(unittest.TestCase):
    def _reason(
        self,
        state: dict[str, Any],
        action: MigrationAction | None = None,
        predictor: p.PostMigrationOutcomePredictor | None = None,
        warm: int = 4,
    ) -> str:
        predictor = predictor or p.PostMigrationOutcomePredictor()
        for earlier in _states()[:warm]:
            predictor.observe(earlier)
        pred = predictor.predict(state, action or _action())
        self.assertEqual(pred.status, p.NOT_READY)
        self.assertIsNotNone(pred.reason)
        return str(pred.reason)

    def test_numbers_are_none_not_zero(self) -> None:
        state = _states()[4]
        state["quality"]["valid"] = False
        pred = p.PostMigrationOutcomePredictor().predict(state, _action())
        for name in EXPECTED_FIELDS[6:15]:
            self.assertIsNone(getattr(pred, name), name)

    def test_invalid_snapshot(self) -> None:
        state = _states()[4]
        state["quality"]["valid"] = False
        self.assertEqual(self._reason(state), p.INVALID_SNAPSHOT)

    def test_malformed_snapshot_is_invalid(self) -> None:
        state = _states()[4]
        del state["created_at"]
        self.assertEqual(self._reason(state), p.INVALID_SNAPSHOT)

    def test_no_op_is_not_a_what_if(self) -> None:
        action = MigrationAction(
            action_type="NO_OP", source_controller="c1", target_controller="c2"
        )
        self.assertEqual(self._reason(_states()[4], action), p.INVALID_ACTION)

    def test_history_shorter_than_five(self) -> None:
        self.assertEqual(self._reason(_states()[3], warm=3), p.INSUFFICIENT_HISTORY)
        self.assertEqual(self._reason(_states()[0], warm=0), p.INSUFFICIENT_HISTORY)

    def test_gap_in_history_is_insufficient(self) -> None:
        late = fx.make_twin_state({"c1": 0.8, "c2": 0.3}, index=30, hot_switch=("s1", 0.2))
        self.assertEqual(self._reason(late), p.INSUFFICIENT_HISTORY)

    def test_switch_not_owned_by_source(self) -> None:
        self.assertEqual(self._reason(_states()[4], _action("s15")), p.SWITCH_NOT_OWNED_BY_SOURCE)

    def test_missing_switch_telemetry(self) -> None:
        state = _states()[4]
        state["switches"] = [s for s in state["switches"] if s["switch_id"] != "s1"]
        self.assertEqual(self._reason(state), p.MISSING_SWITCH_TELEMETRY)

    def test_missing_safe_capacity(self) -> None:
        state = _states()[4]
        state["controllers"][1]["safe_capacity_pps"] = 0.0
        self.assertEqual(self._reason(state), p.MISSING_CAPACITY)

    def test_no_artifact(self) -> None:
        self.assertEqual(self._reason(_states()[4]), p.ARTIFACT_MISSING)

    def test_artifact_entry_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            (root / "model_card.json").unlink()
            predictor = p.PostMigrationOutcomePredictor(root, CAP)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.ARTIFACT_SCHEMA_MISMATCH)

    def test_artifact_schema_name_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            manifest = json.loads((root / "training_manifest.json").read_text(encoding="utf-8"))
            manifest["feature_schema"] = "outcome-features-v0"
            (root / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            predictor = p.PostMigrationOutcomePredictor(root, CAP)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.ARTIFACT_SCHEMA_MISMATCH)

    def test_capacity_version_differs_from_running_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp), capacity_version="cap@old")
            predictor = p.PostMigrationOutcomePredictor(root, "cap@new")
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.CAPACITY_VERSION_MISMATCH)

    def test_unknown_running_capacity_version_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            predictor = p.PostMigrationOutcomePredictor(root, None)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.CAPACITY_VERSION_MISMATCH)

    def test_broken_residual_model_is_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            (root / "residual_models" / "residual_rf_u_src_post.joblib").unlink()
            predictor = p.PostMigrationOutcomePredictor(root, CAP)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.ARTIFACT_SCHEMA_MISMATCH)

    def test_model_feature_not_built_by_build_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            (root / "feature_schema.json").write_text(
                json.dumps({"feature_schema": p.FEATURE_SCHEMA, "model_features": ["u_dst_post"]}),
                encoding="utf-8",
            )
            predictor = p.PostMigrationOutcomePredictor(root, CAP)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.ARTIFACT_SCHEMA_MISMATCH)

    def test_negative_margin_is_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            (root / "uncertainty_margin.json").write_text(
                json.dumps({"m_U": -0.1, "m_R": 0.1}), encoding="utf-8"
            )
            predictor = p.PostMigrationOutcomePredictor(root, CAP)
            self.assertEqual(self._reason(_states()[4], predictor=predictor),
                             p.ARTIFACT_SCHEMA_MISMATCH)

    def test_prediction_carries_ids_and_quality(self) -> None:
        pred = p.PostMigrationOutcomePredictor().predict(_states()[0], _action())
        self.assertEqual(
            (pred.snapshot_id, pred.switch_id, pred.source_controller, pred.target_controller),
            ("snap-0000", "s1", "c1", "c2"),
        )
        self.assertEqual((pred.quality_score, pred.consistency_score), (0.98, 1.0))

    def test_predict_does_not_modify_inputs(self) -> None:
        state = _states()[4]
        before = copy.deepcopy(state)
        p.PostMigrationOutcomePredictor().predict(state, _action())
        self.assertEqual(state, before)


class NumericPathTest(unittest.TestCase):
    def test_zero_residual_equals_analytical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp))
            predictor = _ready_predictor(root)
            pred = predictor.predict(_states()[4], _action())
            self.assertEqual((pred.status, pred.reason), (p.READY, None))
            window = [StateSnapshotView.from_twin_state(s) for s in _states()]
            spec = ActionSpec("s1", "c1", "c2")
            base = analytical.predict(analytical.inputs_from_window(window, spec, "c1", "c2"))
            self.assertAlmostEqual(pred.predicted_src_utilization or -1, base.u_src_post)
            self.assertAlmostEqual(pred.predicted_dst_utilization or -1, base.u_dst_post)
            self.assertAlmostEqual(pred.predicted_src_response_p95_ms or -1, base.r_src95_post)
            self.assertAlmostEqual(pred.predicted_dst_response_p95_ms or -1, base.r_dst95_post)
            self.assertAlmostEqual(pred.predicted_worst_response_p95_ms or -1, base.r95_post)
            self.assertAlmostEqual(pred.predicted_imbalance or -1, base.i_post)
            self.assertEqual((pred.dst_utilization_margin, pred.response_margin_ms), (M_U, M_R))
            self.assertEqual((pred.model_name, pred.model_version),
                             (p.MODEL_NAME, p.MODEL_VERSION))
            self.assertGreater(pred.inference_time_ms or 0.0, 0.0)

    def test_load_moves_from_source_to_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictor = _ready_predictor(_write_artifact(Path(tmp)))
            pred = predictor.predict(_states()[4], _action())
            # 0.2 share of c1 (0.8) moves: src 0.8 -> 0.64, dst 0.3 -> 0.3 + 0.16
            self.assertAlmostEqual(pred.predicted_src_utilization or -1, 0.64, places=6)
            self.assertAlmostEqual(pred.predicted_dst_utilization or -1, 0.46, places=6)

    def test_residual_is_added_and_floored_at_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            up = _ready_predictor(_write_artifact(Path(tmp), constant=0.1))
            pred = up.predict(_states()[4], _action())
            self.assertAlmostEqual(pred.predicted_dst_utilization or -1, 0.56, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            down = _ready_predictor(_write_artifact(Path(tmp), constant=-5.0))
            pred = down.predict(_states()[4], _action())
            self.assertEqual(pred.predicted_src_utilization, 0.0)
            self.assertEqual(pred.predicted_dst_utilization, 0.0)

    def test_features_come_from_the_five_snapshot_window(self) -> None:
        window = [StateSnapshotView.from_twin_state(s) for s in _states()]
        features = build_features(window, ActionSpec("s1", "c1", "c2"))
        self.assertIn("src_u_mean", features)
        self.assertAlmostEqual(features["src_u_mean"], 0.8)

    def test_predict_is_repeatable_and_has_no_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictor = _ready_predictor(_write_artifact(Path(tmp)))
            first = predictor.predict(_states()[4], _action())
            second = predictor.predict(_states()[4], _action())
            self.assertEqual(predictor.history.size(), 5)
            self.assertEqual(first.predicted_dst_utilization, second.predicted_dst_utilization)

    def test_other_family_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_artifact(Path(tmp), family="lightgbm")
            predictor = _ready_predictor(root)
            pred = predictor.predict(_states()[4], _action())
            self.assertEqual((pred.status, pred.reason), (p.NOT_READY, p.ARTIFACT_SCHEMA_MISMATCH))

    def test_fail_closed_after_artifact_is_fine_but_state_breaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            predictor = _ready_predictor(_write_artifact(Path(tmp)))
            state = _states()[4]
            state["switches"] = [
                s for s in state["switches"] if s["switch_id"] != "s1"
            ]
            pred = predictor.predict(state, _action())
            self.assertEqual((pred.status, pred.reason), (p.NOT_READY, p.MISSING_SWITCH_TELEMETRY))
            self.assertIsNone(pred.predicted_dst_utilization)


if __name__ == "__main__":
    unittest.main()
