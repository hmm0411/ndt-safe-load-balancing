#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow

from src.twin.forecasting.baselines import (
    moving_average_predict,
    persistence_predict,
)
from src.twin.forecasting.evaluator import (
    evaluate_by_workload,
    evaluate_predictor,
)
from src.twin.forecasting.features import (
    feature_names as expected_feature_names,
)
from src.twin.forecasting.linear_model import (
    feature_shift_report,
    fit_linear_regression,
    near_constant_features,
)
from src.twin.forecasting.splitter import (
    split_records,
)


DATASET_VERSION = (
    "forecast-dataset-v1-h3-provisional"
)

MODEL_VERSION = "baseline-v1"

DATA_ROOT = Path(
    "data/forecasting/processed"
) / DATASET_VERSION

MODEL_ROOT = Path(
    "data/models/forecasting"
) / MODEL_VERSION

DATASET_PATH = (
    DATA_ROOT / "forecast_dataset.csv"
)

DATASET_MANIFEST_PATH = (
    DATA_ROOT
    / "forecast_dataset_manifest.json"
)

SPLIT_MANIFEST_PATH = (
    DATA_ROOT / "split_manifest.json"
)

MA_WINDOWS = (3, 5, 10)


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return [
            dict(row)
            for row in csv.DictReader(handle)
        ]


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def log_metrics(
    prefix: str,
    metrics: dict[str, float],
) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(
            f"{prefix}_{name}",
            float(value),
        )


def main() -> None:
    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = load_records(
        DATASET_PATH
    )

    splits = split_records(records)

    manifest = json.loads(
        DATASET_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    feature_names = list(
        manifest["feature_names"]
    )

    horizon = manifest["horizon"]

    horizon_status = str(
        horizon["status"]
    )

    horizon_steps = int(
        horizon["steps"]
    )

    horizon_effective_seconds = float(
        horizon["effective_seconds"]
    )

    if horizon_steps <= 0:
        raise RuntimeError(
            "dataset horizon steps must be positive"
        )

    if horizon_effective_seconds <= 0:
        raise RuntimeError(
            "dataset effective horizon must be positive"
        )

    # The dataset must have been built with the current feature schema;
    # a stale processed CSV would silently train on dropped columns.
    if feature_names != list(expected_feature_names()):
        raise RuntimeError(
            "processed dataset feature schema is stale; "
            "rebuild it with build_forecast_dataset.py"
        )

    constant_features = near_constant_features(
        splits["train"],
        feature_names,
    )

    if constant_features:
        print(
            "WARNING near-constant training features: "
            + ", ".join(constant_features)
        )

    # Distribution shift of every evaluation split against TRAIN.
    # Written before any model is fitted so it cannot be skipped.
    shift_report = {
        split_name: feature_shift_report(
            splits["train"],
            splits[split_name],
            feature_names,
        )
        for split_name in ("validation", "test", "unseen")
        if splits[split_name]
    }

    write_json(
        MODEL_ROOT / "feature_shift_report.json",
        {
            "near_constant_training_features": constant_features,
            "splits": shift_report,
        },
    )

    for split_name, report in shift_report.items():
        shifted = sorted(
            name
            for name, item in report.items()
            if item["max_abs_z"] > 10.0
        )

        if shifted:
            print(
                f"WARNING {split_name}: features far outside the "
                "training distribution: " + ", ".join(shifted)
            )

    validation_results: dict[
        str,
        dict[str, float],
    ] = {}

    # -------------------------------------------------
    # Persistence: validation only at selection stage.
    # -------------------------------------------------
    persistence_val, _ = (
        evaluate_predictor(
            splits["validation"],
            persistence_predict,
        )
    )

    validation_results[
        "persistence"
    ] = persistence_val

    # -------------------------------------------------
    # Moving Average candidate selection.
    # Validation RMSE only.
    # -------------------------------------------------
    for window in MA_WINDOWS:
        name = f"moving_average_w{window}"

        metrics, _ = evaluate_predictor(
            splits["validation"],
            lambda rows, w=window:
                moving_average_predict(
                    rows,
                    window=w,
                ),
        )

        validation_results[name] = metrics

    selected_ma_name = min(
        (
            f"moving_average_w{window}"
            for window in MA_WINDOWS
        ),
        key=lambda name:
            validation_results[name][
                "rmse"
            ],
    )

    selected_ma_window = int(
        selected_ma_name.rsplit(
            "w",
            1,
        )[1]
    )

    # -------------------------------------------------
    # Linear Regression.
    # Scaler is fitted inside this function using
    # TRAIN ONLY.
    # -------------------------------------------------
    linear = fit_linear_regression(
        splits["train"],
        feature_names,
    )

    linear_val, _ = evaluate_predictor(
        splits["validation"],
        linear.predict,
    )

    validation_results[
        "linear_regression"
    ] = linear_val

    # -------------------------------------------------
    # Configuration is now frozen.
    # Only now evaluate test + unseen.
    # -------------------------------------------------
    final_results: dict[
        str,
        dict[str, dict[str, float]],
    ] = {}

    final_predictions: dict[
        str,
        dict[str, Any],
    ] = {}

    predictors = {
        "persistence":
            persistence_predict,
        selected_ma_name:
            lambda rows:
                moving_average_predict(
                    rows,
                    window=selected_ma_window,
                ),
        "linear_regression":
            linear.predict,
    }

    for model_name, predictor in (
        predictors.items()
    ):
        final_results[model_name] = {}
        final_predictions[model_name] = {}

        for split_name in (
            "test",
            "unseen",
        ):
            metrics, predictions = (
                evaluate_predictor(
                    splits[split_name],
                    predictor,
                )
            )

            final_results[
                model_name
            ][split_name] = metrics

            final_predictions[
                model_name
            ][split_name] = (
                evaluate_by_workload(
                    splits[split_name],
                    predictions,
                )
            )

    # -------------------------------------------------
    # Official artifacts.
    # -------------------------------------------------
    joblib.dump(
        linear.model,
        MODEL_ROOT
        / "linear_regression.joblib",
    )

    joblib.dump(
        linear.scaler,
        MODEL_ROOT / "scaler.joblib",
    )

    write_json(
        MODEL_ROOT / "feature_schema.json",
        {
            "feature_count":
                len(feature_names),
            "feature_names":
                feature_names,
            "target":
                "future_utilization",
        },
    )

    split_manifest = json.loads(
        SPLIT_MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    write_json(
        MODEL_ROOT / "split_manifest.json",
        split_manifest,
    )

    write_json(
        MODEL_ROOT
        / "validation_metrics.json",
        validation_results,
    )

    write_json(
        MODEL_ROOT
        / "test_metrics.json",
        {
            name: result["test"]
            for name, result
            in final_results.items()
        },
    )

    write_json(
        MODEL_ROOT
        / "unseen_metrics.json",
        {
            name: result["unseen"]
            for name, result
            in final_results.items()
        },
    )

    write_json(
        MODEL_ROOT
        / "per_workload_metrics.json",
        final_predictions,
    )

    training_manifest = {
        "schema_version": "1.0",
        "model_version":
            MODEL_VERSION,
        "dataset_version":
            DATASET_VERSION,
        "horizon_status":
            horizon_status,
        "horizon_steps":
            horizon_steps,
        "horizon_effective_seconds":
            horizon_effective_seconds,
        "feature_count":
            len(feature_names),
        "ma_candidates":
            list(MA_WINDOWS),
        "ma_selection_metric":
            "validation_rmse",
        "selected_ma_window":
            selected_ma_window,
        "scaler_fit_scope":
            "train_only",
        "random_shuffle":
            False,
        "test_used_for_selection":
            False,
        "unseen_used_for_selection":
            False,
    }

    write_json(
        MODEL_ROOT
        / "training_manifest.json",
        training_manifest,
    )

    # -------------------------------------------------
    # MLflow tracking.
    # Local artifacts remain authoritative.
    # -------------------------------------------------
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    mlflow.set_experiment(
        "ndt-forecasting"
    )

    with mlflow.start_run(
        run_name=(
            "week9-baseline-v1-"
            "h3-provisional"
        )
    ):
        mlflow.log_params({
            "model_version":
                MODEL_VERSION,
            "dataset_version":
                DATASET_VERSION,
            "horizon_steps":
                horizon_steps,
            "horizon_status":
                horizon_status,
            "horizon_effective_seconds":
                horizon_effective_seconds,
            "feature_count":
                len(feature_names),
            "selected_ma_window":
                selected_ma_window,
            "ma_selection_metric":
                "validation_rmse",
            "scaler_fit_scope":
                "train_only",
            "random_shuffle":
                False,
        })

        for model_name, metrics in (
            validation_results.items()
        ):
            log_metrics(
                f"validation_{model_name}",
                metrics,
            )

        for model_name, result in (
            final_results.items()
        ):
            log_metrics(
                f"test_{model_name}",
                result["test"],
            )

            log_metrics(
                f"unseen_{model_name}",
                result["unseen"],
            )

        mlflow.log_artifacts(
            str(MODEL_ROOT),
            artifact_path="baseline-v1",
        )

    print("=" * 72)
    print("WEEK 9 FORECAST BASELINES")
    print("=" * 72)

    print()
    print("Validation RMSE:")
    for name, metrics in sorted(
        validation_results.items()
    ):
        print(
            f"  {name:22} "
            f"{metrics['rmse']:.6f}"
        )

    print()
    print(
        "Selected MA window :",
        selected_ma_window,
    )

    print()
    print("TEST:")
    for name, result in (
        final_results.items()
    ):
        metrics = result["test"]
        print(
            f"  {name:22} "
            f"MAE={metrics['mae']:.6f} "
            f"RMSE={metrics['rmse']:.6f} "
            f"R2={metrics['r2']:.6f}"
        )

    print()
    print("UNSEEN:")
    for name, result in (
        final_results.items()
    ):
        metrics = result["unseen"]
        print(
            f"  {name:22} "
            f"MAE={metrics['mae']:.6f} "
            f"RMSE={metrics['rmse']:.6f} "
            f"R2={metrics['r2']:.6f}"
        )

    print()
    print("Artifacts:", MODEL_ROOT)
    print("MLflow    : sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
