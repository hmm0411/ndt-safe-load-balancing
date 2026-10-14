#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import argparse
import functools

import joblib
import yaml

from src.twin.forecasting.versions import MODEL_VERSION as VERSIONED_MODEL
from src.twin.forecasting.versions import dataset_version
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
from sklearn.pipeline import make_pipeline

from src.twin.forecasting.linear_model import (
    records_to_matrix,
    feature_shift_report,
    fit_linear_regression,
    near_constant_features,
)
from src.twin.forecasting.splitter import (
    split_records,
)
from src.twin.forecasting.selection import (
    DEFAULT_WEIGHTS,
    measure_latency_ms,
    select_model,
    selection_score,
    single_sample_predictors,
)
from src.twin.forecasting.tracking import Tracker
from src.twin.forecasting.tree_models import (
    fit_random_forest,
    fit_xgboost,
)


DATASET_VERSION = dataset_version()

MODEL_VERSION = VERSIONED_MODEL

DATA_ROOT = Path(
    "data/forecasting/processed"
) / DATASET_VERSION

MODELS_ROOT = Path("data/models/forecasting")

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

MODEL_CONFIG_PATH = Path("configs/forecasting/models.yaml")


def load_model_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("mlflow", "moving_average", "random_forest", "xgboost"):
        if section not in config:
            raise ValueError(f"{path}: missing section {section!r}")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train persistence, moving average, Linear Regression, "
        "Random Forest and XGBoost forecasters."
    )
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG_PATH)
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (default: <model_version>-<dataset_version>)",
    )
    parser.add_argument(
        "--note",
        default="",
        help="free-text note stored as the MLflow run description",
    )
    parser.add_argument(
        "--model-version",
        default=MODEL_VERSION,
        help="artifact directory under data/models/forecasting/ "
        f"(default: {MODEL_VERSION}). Use a new name so a frozen version "
        "is never overwritten, e.g. forecast-v1.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--register",
        action="store_true",
        help="register the selected tree models in the MLflow Model Registry",
    )
    return parser.parse_args()


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


def records_frame(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> Any:
    """Numeric feature/target view of a split, for MLflow dataset tracking."""
    import pandas as pd

    frame = pd.DataFrame(records)
    keep = [c for c in ("run_id", "workload_type", "controller_id") if c in frame]
    numeric = frame[[*feature_names, "future_utilization"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return pd.concat([frame[keep], numeric], axis=1)


def log_figures(
    tracker: Tracker,
    importances: dict[str, dict[str, float]],
    skill: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Two summary figures attached to the MLflow run (not written locally)."""
    if tracker.run_id is None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for axis, (name, values) in zip(axes, importances.items(), strict=False):
        top = list(values.items())[:15][::-1]
        axis.barh([k for k, _ in top], [v for _, v in top], color="#4C72B0")
        axis.set_title(f"{name}: top-15 feature importance")
    tracker.log_figure(fig, "figures/feature_importance.png")
    plt.close(fig)

    test = skill.get("test", {})
    workloads = list(test)
    models = sorted({m for ratios in test.values() for m in ratios})
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        axis.bar(
            [w + i * width for w in range(len(workloads))],
            [test[w].get(model, float("nan")) for w in workloads],
            width=width,
            label=model,
        )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks([w + 0.4 - width / 2 for w in range(len(workloads))], workloads)
    axis.set_ylabel("RMSE / RMSE persistence (test)")
    axis.set_title("Below 1 = better than persistence")
    axis.legend()
    tracker.log_figure(fig, "figures/skill_vs_persistence_test.png")
    plt.close(fig)


def select_by_score(
    selection_config: dict[str, Any],
    feature_names: list[str],
    validation: list[dict[str, Any]],
    validation_results: dict[str, dict[str, float]],
    *,
    selected_ma_window: int,
    linear: Any,
    random_forest: Any,
    xgboost_model: Any,
) -> dict[str, Any]:
    """S_m per model family on VALIDATION; the lowest score is deployed."""
    weights = dict(selection_config.get("weights", DEFAULT_WEIGHTS))
    tie_tolerance = float(selection_config.get("tie_tolerance", 0.01))
    warmup = int(selection_config.get("latency_warmup", 50))
    repeats = int(selection_config.get("latency_repeats", 200))

    row = records_to_matrix(validation[:1], feature_names)
    predictors = single_sample_predictors(
        feature_names,
        row,
        ma_window=selected_ma_window,
        linear_model=linear.model,
        scaler=linear.scaler,
        random_forest=random_forest.model,
        xgboost=xgboost_model.model,
    )
    baseline = validation_results["persistence"]
    rows: list[dict[str, Any]] = []
    scores: dict[str, float] = {}
    for name, predict_one in predictors.items():
        latency = measure_latency_ms(predict_one, warmup=warmup, repeats=repeats)
        metrics = {**validation_results[name], "inference_p95_ms": latency["p95_ms"]}
        scores[name] = selection_score(metrics, baseline, weights)
        rows.append({
            "model": name,
            "selection_score": scores[name],
            "rmse": metrics["rmse"],
            "p95_absolute_error": metrics["p95_absolute_error"],
            "mae": metrics["mae"],
            "rmse_ratio_vs_persistence": metrics["rmse"] / baseline["rmse"],
            "inference_p50_ms": latency["p50_ms"],
            "inference_p95_ms": latency["p95_ms"],
        })
    result = select_model(scores, tie_tolerance=tie_tolerance)
    rows.sort(key=lambda item: item["selection_score"])
    return {
        "method": "selection_score_validation",
        "split": "validation",
        "baseline": "persistence",
        "weights": weights,
        "latency_cap_ms": 1000.0,
        "latency_warmup": warmup,
        "latency_repeats": repeats,
        "tie_tolerance": tie_tolerance,
        "selected": result.selected,
        "ranking": result.ranking,
        "tree_gap": result.tree_gap,
        "tree_near_tie": result.tree_near_tie,
        "models": rows,
        "test_used_for_selection": False,
        "unseen_used_for_selection": False,
    }


def _train(
    config: dict[str, Any],
    tracker: Tracker,
    model_version: str,
) -> None:
    model_root = MODELS_ROOT / model_version
    model_root.mkdir(
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

    ma_windows: list[int] = [int(w) for w in config["moving_average"]["windows"]]
    rf_config = config["random_forest"]
    xgb_config = config["xgboost"]

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

    # Dataset lineage: name, digest, schema and row count of every split
    # appear under "Datasets used" for this run.
    for split_name, context in (
        ("train", "training"),
        ("validation", "validation"),
        ("test", "testing"),
        ("unseen", "ood_testing"),
    ):
        if splits[split_name]:
            tracker.log_dataset(
                records_frame(splits[split_name], feature_names),
                name=f"{DATASET_VERSION}/{split_name}",
                context=context,
                source=str(DATASET_PATH),
                targets="future_utilization",
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
        model_root / "feature_shift_report.json",
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
    for window in ma_windows:
        name = f"moving_average_w{window}"

        metrics, _ = evaluate_predictor(
            splits["validation"],
            functools.partial(
                moving_average_predict,
                window=window,
            ),
        )

        validation_results[name] = metrics

    selected_ma_name = min(
        (
            f"moving_average_w{window}"
            for window in ma_windows
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
    # Random Forest and XGBoost.
    # Target = U(t+h) - U(t); grids searched on
    # VALIDATION only (lowest RMSE of the forecast).
    # -------------------------------------------------
    print("Fitting Random Forest grid ...", flush=True)
    random_forest, random_forest_grid = fit_random_forest(
        splits["train"],
        splits["validation"],
        feature_names,
        grid=rf_config["grid"],
        fixed=rf_config["fixed"],
        on_candidate=tracker.log_candidate,
    )
    # Runtime predicts one row per controller per second. With n_jobs=-1
    # every predict call starts a thread pool, which costs ~20 ms for one row
    # (measured in M1); single-threaded prediction is far faster for one row.
    random_forest.model.set_params(n_jobs=1)
    print("Fitting XGBoost grid ...", flush=True)
    xgboost_model, xgboost_grid = fit_xgboost(
        splits["train"],
        splits["validation"],
        feature_names,
        grid=xgb_config["grid"],
        fixed=xgb_config["fixed"],
        on_candidate=tracker.log_candidate,
    )

    tracker.log_curve(
        "xgboost_learning_curve_validation_rmse",
        xgboost_model.learning_curve(),
    )

    for tree_model in (random_forest, xgboost_model):
        tree_val, _ = evaluate_predictor(
            splits["validation"],
            tree_model.predict,
        )
        validation_results[tree_model.name] = tree_val

    # -------------------------------------------------
    # Model family selection S_m (validation only).
    # -------------------------------------------------
    model_selection = select_by_score(
        config.get("selection", {}),
        feature_names,
        splits["validation"],
        validation_results,
        selected_ma_window=selected_ma_window,
        linear=linear,
        random_forest=random_forest,
        xgboost_model=xgboost_model,
    )
    write_json(model_root / "model_selection.json", model_selection)

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
        "random_forest":
            random_forest.predict,
        "xgboost":
            xgboost_model.predict,
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
        model_root
        / "linear_regression.joblib",
    )

    joblib.dump(
        linear.scaler,
        model_root / "scaler.joblib",
    )
    # compress=3: the 600-tree forest is several times smaller on disk and
    # loads just as fast (contract C1 keeps model files to a few MB).
    joblib.dump(
        random_forest.model,
        model_root / "random_forest.joblib",
        compress=3,
    )
    xgboost_model.model.save_model(
        str(model_root / "xgboost.json")
    )
    write_json(
        model_root / "hyperparameter_search.json",
        {
            "selection_split": "validation",
            "selection_metric": "rmse",
            "random_forest": {
                "fixed": rf_config["fixed"],
                "grid": rf_config["grid"],
                "selected": random_forest.params,
                "candidates": random_forest_grid,
            },
            "xgboost": {
                "fixed": xgb_config["fixed"],
                "grid": xgb_config["grid"],
                "selected": xgboost_model.params,
                "best_iteration": xgboost_model.extra.get("best_iteration"),
                "candidates": xgboost_grid,
            },
        },
    )
    write_json(
        model_root / "feature_importance.json",
        {
            "random_forest": random_forest.feature_importance(),
            "xgboost": xgboost_model.feature_importance(),
        },
    )

    write_json(
        model_root / "feature_schema.json",
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
        model_root / "split_manifest.json",
        split_manifest,
    )

    write_json(
        model_root
        / "validation_metrics.json",
        validation_results,
    )

    write_json(
        model_root
        / "test_metrics.json",
        {
            name: result["test"]
            for name, result
            in final_results.items()
        },
    )

    write_json(
        model_root
        / "unseen_metrics.json",
        {
            name: result["unseen"]
            for name, result
            in final_results.items()
        },
    )

    write_json(
        model_root
        / "per_workload_metrics.json",
        final_predictions,
    )

    # RMSE of every model divided by persistence RMSE, per split and
    # workload: the number the thesis reports (< 1 means the model beats
    # simply repeating the current load).
    skill_vs_persistence: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in ("test", "unseen"):
        reference = final_predictions["persistence"][split_name]
        skill_vs_persistence[split_name] = {
            workload: {
                model_name: (
                    final_predictions[model_name][split_name][workload]["rmse"]
                    / reference[workload]["rmse"]
                    if reference[workload]["rmse"] > 0
                    else float("nan")
                )
                for model_name in final_predictions
                if model_name != "persistence"
            }
            for workload in sorted(reference)
        }
    write_json(
        model_root / "skill_vs_persistence.json",
        skill_vs_persistence,
    )

    training_manifest = {
        "schema_version": "1.0",
        "model_version":
            model_version,
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
            list(ma_windows),
        "ma_selection_metric":
            "validation_rmse",
        "selected_ma_window":
            selected_ma_window,
        "scaler_fit_scope":
            "train_only",
        "tree_target":
            "future_utilization - utilization",
        "tree_selection":
            "validation_rmse",
        "random_forest_params":
            random_forest.params,
        "xgboost_params":
            xgboost_model.params,
        "xgboost_best_iteration":
            xgboost_model.extra.get("best_iteration"),
        "family_selection":
            "selection_score_validation",
        "selected_model":
            model_selection["selected"],
        "random_shuffle":
            False,
        "test_used_for_selection":
            False,
        "unseen_used_for_selection":
            False,
    }

    write_json(
        model_root
        / "training_manifest.json",
        training_manifest,
    )

    # -------------------------------------------------
    # MLflow tracking (no-op with --no-mlflow).
    # Local artifacts remain authoritative.
    # -------------------------------------------------
    tracker.log_params({
        "model_version": model_version,
        "dataset_version": DATASET_VERSION,
        "horizon_steps": horizon_steps,
        "horizon_status": horizon_status,
        "horizon_effective_seconds": horizon_effective_seconds,
        "feature_count": len(feature_names),
        "selected_ma_window": selected_ma_window,
        "tree_target": "future_utilization - utilization",
        "selection_metric": "validation_rmse",
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
        "unseen_rows": len(splits["unseen"]),
    })
    tracker.log_params(random_forest.params, prefix="random_forest.")
    tracker.log_params(rf_config["fixed"], prefix="random_forest.fixed.")
    tracker.log_params(xgboost_model.params, prefix="xgboost.")
    tracker.log_params(xgb_config["fixed"], prefix="xgboost.fixed.")
    tracker.log_metrics(
        {"xgboost_best_iteration": xgboost_model.extra.get("best_iteration", -1)}
    )
    tracker.log_params({
        "selected_model": model_selection["selected"],
        "selection_tree_near_tie": model_selection["tree_near_tie"],
    })
    for item in model_selection["models"]:
        tracker.log_metrics(
            {
                "selection_score": item["selection_score"],
                "inference_p95_ms": item["inference_p95_ms"],
            },
            prefix=f"{item['model']}_",
        )
    tracker.log_table(model_selection["models"], "tables/model_selection.json")

    for model_name, metrics in validation_results.items():
        tracker.log_metrics(metrics, prefix=f"validation_{model_name}_")

    for model_name, result in final_results.items():
        tracker.log_metrics(result["test"], prefix=f"test_{model_name}_")
        tracker.log_metrics(result["unseen"], prefix=f"unseen_{model_name}_")

    for model_name, by_split in final_predictions.items():
        for split_name, by_workload in by_split.items():
            for workload, metrics in by_workload.items():
                tracker.log_metrics(
                    {"rmse": metrics["rmse"], "mae": metrics["mae"]},
                    prefix=f"{split_name}_{workload}_{model_name}_",
                )

    for split_name, by_workload in skill_vs_persistence.items():
        for workload, ratios in by_workload.items():
            tracker.log_metrics(
                ratios, prefix=f"skill_{split_name}_{workload}_"
            )

    tracker.log_table(
        {
            "model": [m for m, by in final_predictions.items() for s in by for _ in by[s]],
            "split": [s for by in final_predictions.values() for s in by for _ in by[s]],
            "workload": [w for by in final_predictions.values() for s in by for w in by[s]],
            "rmse": [by[s][w]["rmse"] for by in final_predictions.values() for s in by for w in by[s]],
            "mae": [by[s][w]["mae"] for by in final_predictions.values() for s in by for w in by[s]],
        },
        "tables/per_workload_metrics.json",
    )
    for name, table in (("random_forest", random_forest_grid), ("xgboost", xgboost_grid)):
        tracker.log_table(
            [{**{f"param_{k}": str(v) for k, v in row["params"].items()},
              **{k: v for k, v in row.items() if k != "params"}} for row in table],
            f"tables/grid_{name}.json",
        )
    log_figures(
        tracker,
        {
            "random_forest": random_forest.feature_importance(),
            "xgboost": xgboost_model.feature_importance(),
        },
        skill_vs_persistence,
    )

    def version_tags(model_name: str) -> dict[str, Any]:
        return {
            "dataset_version": DATASET_VERSION,
            "model_version": model_version,
            "horizon_steps": horizon_steps,
            "target": "future_utilization - utilization",
            "validation_rmse": validation_results[model_name]["rmse"],
            "test_rmse": final_results[model_name]["test"]["rmse"],
            "unseen_rmse": final_results[model_name]["unseen"]["rmse"],
        }

    example = records_to_matrix(splits["validation"][:5], feature_names)
    tracker.log_model(
        "sklearn",
        random_forest.model,
        "random_forest",
        example,
        # Our own freshly trained forest, not a downloaded file.
        trusted_types=["sklearn.tree._tree.Tree"],
        version_tags=version_tags("random_forest"),
    )
    tracker.log_model(
        "xgboost",
        xgboost_model.model,
        "xgboost",
        example,
        version_tags=version_tags("xgboost"),
    )
    tracker.log_model(
        "sklearn",
        make_pipeline(linear.scaler, linear.model),
        "linear_regression",
        example,
    )
    tracker.write_pointer(model_root / "mlflow_run.json")
    tracker.log_artifacts(model_root, artifact_path=model_version)

    print("=" * 72)
    print(f"FORECAST MODELS {model_version} on {DATASET_VERSION}")
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
    print("Random Forest      :", random_forest.params)
    print(
        "XGBoost            :",
        xgboost_model.params,
        "best_iteration =",
        xgboost_model.extra.get("best_iteration"),
    )

    print()
    print("Selection score S_m on validation (lower = better):")
    for item in model_selection["models"]:
        print(
            f"  {item['model']:22} S_m={item['selection_score']:.4f} "
            f"latency_p95={item['inference_p95_ms']:.3f} ms"
        )
    print("Selected model     :", model_selection["selected"])
    if model_selection["tree_near_tie"]:
        print(
            "NOTE random_forest and xgboost differ by "
            f"{model_selection['tree_gap']:.4f} < {model_selection['tie_tolerance']}: "
            "report them as comparable, not as a clear winner."
        )

    print()
    print("TEST, RMSE relative to persistence per workload (< 1 = better):")
    for workload, ratios in skill_vs_persistence["test"].items():
        print(
            f"  {workload:12} "
            + "  ".join(f"{name}={ratio:.3f}" for name, ratio in ratios.items())
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
            f"R2={metrics['r2']:.6f} "
            f"P95={metrics['p95_absolute_error']:.6f}"
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
            f"R2={metrics['r2']:.6f} "
            f"P95={metrics['p95_absolute_error']:.6f}"
        )

    print()
    print("Artifacts:", model_root)
    if tracker.run_id:
        print(f"MLflow    : {tracker.tracking_uri}  run_id={tracker.run_id}")
        for name, version in tracker.registered_versions.items():
            print(f"            registered {name} version {version} (alias {tracker.candidate_alias})")


def main() -> None:
    args = parse_args()
    config = load_model_config(args.config)
    mlflow_config = config["mlflow"]

    tracker = Tracker(
        enabled=bool(mlflow_config.get("enabled", True)) and not args.no_mlflow,
        tracking_uri=str(mlflow_config["tracking_uri"]),
        experiment=str(mlflow_config["experiment"]),
        run_name=args.run_name or f"{args.model_version}-{DATASET_VERSION}",
        register_models=args.register or bool(mlflow_config.get("register_models")),
        registered_names=mlflow_config.get("registered_names", {}),
        system_metrics=bool(mlflow_config.get("system_metrics", False)),
        system_metrics_interval=float(
            mlflow_config.get("system_metrics_interval_seconds", 5.0)
        ),
        candidate_alias=str(mlflow_config.get("candidate_alias", "challenger")),
    )

    # The run opens before any fitting so progress (grid candidates) shows up
    # in the MLflow UI while training is still running.
    tracker.start(
        {
            "model_version": args.model_version,
            "dataset_version": DATASET_VERSION,
            "model_config": str(args.config),
            "role": "training",
            # The VM is shared under one Linux user, so mlflow.user is the
            # same for everyone; filter runs with tags.owner = "my" / "vy".
            "owner": os.getenv("NDT_OWNER", "unknown"),
            "mlflow.note.content": args.note,
        }
    )
    try:
        _train(config, tracker, args.model_version)
    except BaseException:
        tracker.end(status="FAILED")
        raise
    tracker.end()


if __name__ == "__main__":
    main()
