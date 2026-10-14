"""forecast-v1 artifact: build and check the files of contract C1.

See docs/forecast_artifact_contract.md. The runtime predictor reads only
model_card.json, feature_schema.json, error_margin.json and (optionally)
feature_ranges.json, plus the model file named in the card.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.twin.forecasting.features import LAG_STEPS, feature_names as expected_feature_names

# Current sample + every lag the features use (utilization_lag_1..10).
HISTORY_SAMPLES_REQUIRED = max(LAG_STEPS) + 1

MODEL_FILES: dict[str, str | None] = {
    "xgboost": "xgboost.json",
    "random_forest": "random_forest.joblib",
    "linear_regression": "linear_regression.joblib",
    "persistence": None,
}
SCALER_FILES: dict[str, str | None] = {"linear_regression": "scaler.joblib"}
# Tree models learn U(t+h) - U(t); Linear Regression learns the level.
OUTPUTS = {
    "xgboost": "delta",
    "random_forest": "delta",
    "linear_regression": "level",
    "persistence": "delta",
}
NOT_A_TRIGGER = "not authorized to trigger migration directly"
LIMITATIONS = (
    NOT_A_TRIGGER,
    "trained on 2 controllers x 20 switches with fixed ownership; "
    "migration and hot-switch were never in the training data",
    "hot-switch is unseen in training; tree models were worse than persistence "
    "there in forecast-models-v2, so the OOD gate (feature_ranges.json) falls back "
    "to persistence",
    "horizon of 3 s is provisional until decision latency is measured (horizon_v1.json)",
)
MAX_MODEL_BYTES = 25 * 1024 * 1024

RUNTIME_FILES = ("model_card.json", "feature_schema.json", "error_margin.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_feature_schema(feature_names: Sequence[str]) -> dict[str, Any]:
    names = list(feature_names)
    if tuple(names) != expected_feature_names():
        raise ValueError("feature names differ from features.feature_names()")
    return {
        "feature_count": len(names),
        "feature_names": names,
        "history_samples_required": HISTORY_SAMPLES_REQUIRED,
        "target": "future_utilization",
    }


def build_model_card(
    *,
    model_name: str,
    model_version: str,
    dataset_version: str,
    horizon_steps: int,
    sampling_interval_seconds: float,
    mlflow_run_id: str | None,
    registered_model: str | None,
    registered_version: str | None,
    git_commit: str,
    dataset_sha256: str,
    selection_score: float | None,
) -> dict[str, Any]:
    if model_name not in MODEL_FILES:
        raise ValueError(
            f"{model_name!r} is not deployable under contract C1 "
            f"(choose one of {sorted(MODEL_FILES)})"
        )
    return {
        "model_version": model_version,
        "model_name": model_name,
        "model_file": MODEL_FILES[model_name],
        "scaler_file": SCALER_FILES.get(model_name),
        "output": OUTPUTS[model_name],
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "horizon_steps": horizon_steps,
        "sampling_interval_seconds": sampling_interval_seconds,
        "horizon_seconds": horizon_steps * sampling_interval_seconds,
        "selection_score": selection_score,
        "mlflow_run_id": mlflow_run_id,
        "registered_model": registered_model,
        "registered_version": registered_version,
        "git_commit": git_commit,
        "purpose": (
            "Forecast controller utilization U(t+h) from TwinState history; "
            "U_hat + m_f >= 1 marks an overload risk for the planner."
        ),
        "limitations": list(LIMITATIONS),
    }


def _read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def validate_artifact(model_dir: Path) -> list[str]:
    """Contract C1 violations of one artifact directory (empty = valid)."""
    errors: list[str] = []
    for file_name in RUNTIME_FILES:
        if not (model_dir / file_name).is_file():
            errors.append(f"missing {file_name}")
    if errors:
        return errors

    card = _read(model_dir / "model_card.json")
    for key in ("model_version", "model_name", "output", "dataset_version",
                "horizon_steps", "sampling_interval_seconds", "mlflow_run_id",
                "registered_model", "registered_version", "git_commit",
                "purpose", "limitations"):
        if key not in card:
            errors.append(f"model_card.json: missing {key}")
    name = card.get("model_name")
    if name not in MODEL_FILES:
        errors.append(f"model_card.json: unknown model_name {name!r}")
    else:
        if card.get("model_file") != MODEL_FILES[name]:
            errors.append(f"model_card.json: model_file must be {MODEL_FILES[name]!r}")
        if card.get("scaler_file") != SCALER_FILES.get(name):
            errors.append(f"model_card.json: scaler_file must be {SCALER_FILES.get(name)!r}")
        if card.get("output") != OUTPUTS[name]:
            errors.append(f"model_card.json: output must be {OUTPUTS[name]!r}")
    if NOT_A_TRIGGER not in card.get("limitations", []):
        errors.append(f"model_card.json: limitations must include {NOT_A_TRIGGER!r}")
    if model_dir.name != card.get("model_version"):
        errors.append("model_card.json: model_version differs from the directory name")
    for key in ("model_file", "scaler_file"):
        referenced = card.get(key)
        if referenced:
            path = model_dir / str(referenced)
            if not path.is_file():
                errors.append(f"missing {referenced} (model_card.{key})")
            elif path.stat().st_size > MAX_MODEL_BYTES:
                errors.append(f"{referenced} is larger than 25 MB")

    schema = _read(model_dir / "feature_schema.json")
    if tuple(schema.get("feature_names", ())) != expected_feature_names():
        errors.append("feature_schema.json: feature_names differ from features.feature_names()")
    if schema.get("history_samples_required") != HISTORY_SAMPLES_REQUIRED:
        errors.append(
            f"feature_schema.json: history_samples_required must be {HISTORY_SAMPLES_REQUIRED}"
        )
    if schema.get("target") != "future_utilization":
        errors.append("feature_schema.json: target must be 'future_utilization'")

    margin = _read(model_dir / "error_margin.json")
    if margin.get("method") != "validation_abs_error_finite_sample_q95":
        errors.append("error_margin.json: unexpected method")
    for key in ("alpha", "global", "controllers", "n_validation", "one_sided_q95"):
        if key not in margin:
            errors.append(f"error_margin.json: missing {key}")
    if margin.get("model_name", name) != name:
        errors.append("error_margin.json: computed for a different model than model_card")
    return errors


def summarize_metrics(
    model_name: str,
    reports: Mapping[str, Mapping[str, Any]],
    coverage_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """metrics.json: RMSE, MAE, P95 and Coverage95 of the deployed model."""
    out: dict[str, Any] = {"model_name": model_name, "splits": {}}
    for split_name, report in reports.items():
        metrics = report.get(model_name)
        if metrics is None:
            continue
        row = {
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "p95_absolute_error": metrics["p95_absolute_error"],
        }
        persistence = report.get("persistence")
        if persistence and persistence.get("rmse"):
            row["rmse_ratio_vs_persistence"] = metrics["rmse"] / persistence["rmse"]
        if coverage_report is not None:
            split = coverage_report.get("splits", {}).get(split_name)
            if split is not None:
                row["coverage95"] = split["without_gate"]["coverage"]
                row["coverage95_with_ood_gate"] = split["with_gate"]["coverage"]
        out["splits"][split_name] = row
    return out
