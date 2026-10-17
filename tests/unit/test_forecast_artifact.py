"""Contract C1 checks of a forecast artifact directory (artifact.py)."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.twin.forecasting.artifact import (
    HISTORY_SAMPLES_REQUIRED,
    MAX_MODEL_BYTES,
    NOT_A_TRIGGER,
    build_feature_schema,
    build_model_card,
    summarize_metrics,
    validate_artifact,
)
from src.twin.forecasting.features import feature_names


def _card(model_name: str = "xgboost", model_version: str = "forecast-v1") -> dict[str, Any]:
    return build_model_card(
        model_name=model_name,
        model_version=model_version,
        dataset_version="forecast-dataset-v2-h3-provisional",
        horizon_steps=3,
        sampling_interval_seconds=1.0,
        mlflow_run_id="run",
        registered_model="ndt-forecast-xgboost",
        registered_version="2",
        git_commit="abc",
        dataset_sha256="0" * 64,
        selection_score=0.3,
    )


def _margin(model_name: str = "xgboost") -> dict[str, Any]:
    return {
        "method": "validation_abs_error_finite_sample_q95",
        "alpha": 0.05,
        "global": 0.05,
        "controllers": {"c1": 0.048, "c2": 0.055},
        "n_validation": 1336,
        "one_sided_q95": 0.031,
        "model_name": model_name,
    }


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


class ArtifactTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name) / "forecast-v1"
        self.dir.mkdir()

    def make_valid(self, model_name: str = "xgboost") -> None:
        card = _card(model_name)
        _write(self.dir / "model_card.json", card)
        _write(self.dir / "feature_schema.json", build_feature_schema(feature_names()))
        _write(self.dir / "error_margin.json", _margin(model_name))
        for key in ("model_file", "scaler_file"):
            if card[key]:
                (self.dir / card[key]).write_bytes(b"model")

    def errors(self) -> list[str]:
        return validate_artifact(self.dir)


class ValidateArtifactTest(ArtifactTestCase):
    def test_valid_artifact_has_no_errors(self) -> None:
        self.make_valid()
        self.assertEqual(self.errors(), [])

    def test_valid_linear_regression_needs_its_scaler(self) -> None:
        self.make_valid("linear_regression")
        self.assertEqual(self.errors(), [])
        (self.dir / "scaler.joblib").unlink()
        self.assertIn("missing scaler.joblib (model_card.scaler_file)", self.errors())

    def test_missing_runtime_file(self) -> None:
        self.make_valid()
        (self.dir / "error_margin.json").unlink()
        self.assertEqual(self.errors(), ["missing error_margin.json"])

    def test_missing_model_file(self) -> None:
        self.make_valid()
        (self.dir / "xgboost.json").unlink()
        self.assertIn("missing xgboost.json (model_card.model_file)", self.errors())

    def test_history_of_ten_samples_is_rejected(self) -> None:
        self.make_valid()
        schema = build_feature_schema(feature_names())
        schema["history_samples_required"] = 10
        _write(self.dir / "feature_schema.json", schema)
        self.assertTrue(any("history_samples_required" in e for e in self.errors()))

    def test_feature_order_must_match_features_module(self) -> None:
        self.make_valid()
        schema = build_feature_schema(feature_names())
        schema["feature_names"] = list(reversed(schema["feature_names"]))
        _write(self.dir / "feature_schema.json", schema)
        self.assertTrue(any("feature_names differ" in e for e in self.errors()))

    def test_limitations_must_forbid_direct_migration(self) -> None:
        self.make_valid()
        card = _card()
        card["limitations"] = [x for x in card["limitations"] if x != NOT_A_TRIGGER]
        _write(self.dir / "model_card.json", card)
        self.assertTrue(any("limitations must include" in e for e in self.errors()))

    def test_wrong_output_kind(self) -> None:
        self.make_valid()
        card = _card()
        card["output"] = "level"
        _write(self.dir / "model_card.json", card)
        self.assertIn("model_card.json: output must be 'delta'", self.errors())

    def test_model_larger_than_25_mb(self) -> None:
        self.make_valid()
        # Sparse file: no real disk space is used.
        os.truncate(self.dir / "xgboost.json", MAX_MODEL_BYTES + 1)
        self.assertIn("xgboost.json is larger than 25 MB", self.errors())

    def test_directory_name_must_equal_model_version(self) -> None:
        self.make_valid()
        renamed = self.dir.with_name("forecast-v2")
        self.dir.rename(renamed)
        self.dir = renamed
        self.assertIn(
            "model_card.json: model_version differs from the directory name", self.errors()
        )

    def test_margin_of_another_model_is_rejected(self) -> None:
        self.make_valid()
        _write(self.dir / "error_margin.json", _margin("random_forest"))
        self.assertIn(
            "error_margin.json: computed for a different model than model_card", self.errors()
        )

    def test_margin_missing_key(self) -> None:
        self.make_valid()
        margin = _margin()
        del margin["controllers"]
        _write(self.dir / "error_margin.json", margin)
        self.assertIn("error_margin.json: missing controllers", self.errors())


class BuildersTest(unittest.TestCase):
    def test_moving_average_is_not_deployable(self) -> None:
        with self.assertRaises(ValueError):
            _card("moving_average_w3")

    def test_feature_schema_requires_the_training_order(self) -> None:
        schema = build_feature_schema(feature_names())
        self.assertEqual(schema["history_samples_required"], HISTORY_SAMPLES_REQUIRED)
        self.assertEqual(HISTORY_SAMPLES_REQUIRED, 11)
        self.assertEqual(schema["feature_count"], 27)
        with self.assertRaises(ValueError):
            build_feature_schema(list(reversed(feature_names())))

    def test_card_fields_for_xgboost(self) -> None:
        card = _card()
        self.assertEqual(card["model_file"], "xgboost.json")
        self.assertIsNone(card["scaler_file"])
        self.assertEqual(card["output"], "delta")
        self.assertEqual(card["horizon_seconds"], 3.0)
        self.assertIn(NOT_A_TRIGGER, card["limitations"])


class SummarizeMetricsTest(unittest.TestCase):
    def test_ratio_and_coverage_per_split(self) -> None:
        reports = {
            "test": {
                "xgboost": {"rmse": 0.04, "mae": 0.015, "p95_absolute_error": 0.05},
                "persistence": {"rmse": 0.1, "mae": 0.05, "p95_absolute_error": 0.2},
            },
            "unseen": {"persistence": {"rmse": 0.03, "mae": 0.01, "p95_absolute_error": 0.05}},
        }
        coverage = {"splits": {"test": {"without_gate": {"coverage": 0.957},
                                        "with_gate": {"coverage": 0.957}}}}
        out = summarize_metrics("xgboost", reports, coverage)
        row = out["splits"]["test"]
        self.assertAlmostEqual(row["rmse_ratio_vs_persistence"], 0.4)
        self.assertEqual(row["coverage95"], 0.957)
        self.assertNotIn("unseen", out["splits"])  # no xgboost row there


if __name__ == "__main__":
    unittest.main()
