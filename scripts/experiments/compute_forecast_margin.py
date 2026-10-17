#!/usr/bin/env python3
"""Error margin m_f, Coverage95 and the OOD gate for one trained model (M2).

Reads the model chosen by S_m (model_selection.json) from
data/models/forecasting/<model_version>/ and writes, next to it:

  error_margin.json     [runtime] finite-sample q95 of |e| on VALIDATION
  feature_ranges.json   [runtime] train min/max for the OOD gate (model
                        features + monitor-only columns, see ood.py)
  coverage.json         Coverage95 per split / controller / workload,
                        with and without the OOD gate, + alarm confusion
  ood_degradation.json  D_OOD and RMSE with / without the gate

The model is never refit here: m_f is the error of the deployed model on
data it has not seen. Test and hot-switch are only evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np

from src.twin.forecasting.baselines import moving_average_predict, persistence_predict
from src.twin.forecasting.linear_model import records_to_matrix
from src.twin.forecasting.margin import (
    coverage,
    coverage_by_group,
    error_margins,
    margin_for,
    overload_confusion,
)
from src.twin.forecasting.metrics import evaluate_regression
from src.twin.forecasting.ood import (
    MONITOR_ONLY_FEATURES,
    FeatureRanges,
    gate_mask,
    monitored_names,
)
from src.twin.forecasting.splitter import split_records
from src.twin.forecasting.tracking import log_to_pointed_run
from src.twin.forecasting.versions import MODEL_VERSION, MODELS_ROOT, PROCESSED_ROOT, dataset_version

Predictor = Callable[[list[dict[str, Any]]], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument(
        "--model",
        default=None,
        help="model to evaluate (default: 'selected' in model_selection.json)",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--consecutive-steps", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--model-features-only",
        action="store_true",
        help="gate on the 27 model features only (the first forecast-v1 gate; "
        "kept to report the before/after comparison)",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_predictor(model_dir: Path, model_name: str, feature_names: list[str]) -> Predictor:
    """Batch U(t+h) forecaster for one model family, from saved files only."""
    if model_name == "persistence":
        return persistence_predict
    if model_name.startswith("moving_average_w"):
        window = int(model_name.rsplit("w", 1)[1])
        return lambda rows: moving_average_predict(rows, window=window)
    if model_name == "linear_regression":
        model = joblib.load(model_dir / "linear_regression.joblib")
        scaler = joblib.load(model_dir / "scaler.joblib")
        return lambda rows: np.asarray(
            model.predict(scaler.transform(records_to_matrix(rows, feature_names))), dtype=float
        )
    if model_name == "random_forest":
        tree = joblib.load(model_dir / "random_forest.joblib")
    elif model_name == "xgboost":
        from xgboost import XGBRegressor

        tree = XGBRegressor()
        tree.load_model(str(model_dir / "xgboost.json"))
    else:
        raise ValueError(f"unknown model {model_name!r}")

    def predict_delta(rows: list[dict[str, Any]]) -> np.ndarray:
        current = persistence_predict(rows)
        return current + np.asarray(tree.predict(records_to_matrix(rows, feature_names)))

    return predict_delta


def time_ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The OOD counter needs consecutive samples of one sequence in order."""
    return sorted(rows, key=lambda r: (str(r["sequence_id"]), str(r["observed_at"])))


def evaluate_split(
    rows: list[dict[str, Any]],
    predictor: Predictor,
    feature_names: list[str],
    margins: dict[str, Any],
    ranges: FeatureRanges,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = time_ordered(rows)
    y_true = np.asarray([float(r["future_utilization"]) for r in rows])
    current = persistence_predict(rows)
    model_pred = predictor(rows)
    controllers = [str(r["controller_id"]) for r in rows]
    workloads = [str(r["workload_type"]) for r in rows]
    m = np.asarray([margin_for(margins, cid) for cid in controllers])

    flags = ranges.out_of_range(records_to_matrix(rows, list(ranges.feature_names)))
    active = gate_mask(flags, [str(r["sequence_id"]) for r in rows], ranges.consecutive_steps)
    gated_pred = np.where(active, current, model_pred)

    cov: dict[str, Any] = {}
    for label, pred in (("without_gate", model_pred), ("with_gate", gated_pred)):
        cov[label] = {
            "coverage": coverage(y_true, pred, m),
            "by_controller": coverage_by_group(y_true, pred, m, controllers),
            "by_workload": coverage_by_group(y_true, pred, m, workloads),
            "overload": overload_confusion(y_true, pred + m, threshold),
        }
    cov["samples"] = len(rows)

    violations: dict[str, int] = {}
    for row in records_to_matrix(rows, list(ranges.feature_names)):
        for name in ranges.violations(row):
            violations[name] = violations.get(name, 0) + 1
    degradation = {
        "samples": len(rows),
        "out_of_range_rate": float(flags.mean()),
        "out_of_range_by_feature": {
            name: count / len(rows) for name, count in sorted(violations.items())
        },
        "d_ood": float(active.mean()),
        "rmse_without_gate": evaluate_regression(y_true, model_pred)["rmse"],
        "rmse_with_gate": evaluate_regression(y_true, gated_pred)["rmse"],
        "rmse_persistence": evaluate_regression(y_true, current)["rmse"],
        "d_ood_by_workload": {
            w: float(active[np.asarray(workloads) == w].mean()) for w in sorted(set(workloads))
        },
    }
    return cov, degradation


def main() -> None:
    args = parse_args()
    model_dir = MODELS_ROOT / args.model_version
    feature_names = list(read_json(model_dir / "feature_schema.json")["feature_names"])
    model_name = args.model or read_json(model_dir / "model_selection.json")["selected"]

    dataset = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
    with dataset.open(encoding="utf-8", newline="") as handle:
        splits = split_records([dict(row) for row in csv.DictReader(handle)])

    predictor = load_predictor(model_dir, model_name, feature_names)

    validation = splits["validation"]
    margins = error_margins(
        [float(r["future_utilization"]) for r in validation],
        predictor(validation),
        [str(r["controller_id"]) for r in validation],
        alpha=args.alpha,
    )
    margins["model_name"] = model_name
    margins["model_version"] = args.model_version
    write_json(model_dir / "error_margin.json", margins)

    monitor_only = () if args.model_features_only else monitored_names(
        feature_names, MONITOR_ONLY_FEATURES
    )[len(feature_names):]
    watched = list(feature_names) + list(monitor_only)
    ranges = FeatureRanges.fit(
        records_to_matrix(splits["train"], watched),
        watched,
        tolerance_fraction=args.tolerance_fraction,
        consecutive_steps=args.consecutive_steps,
        monitor_only=monitor_only,
    )
    write_json(model_dir / "feature_ranges.json", ranges.to_dict())

    coverage_report: dict[str, Any] = {
        "model_name": model_name,
        "model_version": args.model_version,
        "margin_source": "validation",
        "threshold": args.threshold,
        "splits": {},
    }
    degradation_report: dict[str, Any] = {
        "model_name": model_name,
        "model_version": args.model_version,
        "gate": {k: v for k, v in ranges.to_dict().items() if k != "features"},
        "splits": {},
    }
    for split_name in ("validation", "test", "unseen"):
        cov, deg = evaluate_split(
            splits[split_name], predictor, feature_names, margins, ranges, args.threshold
        )
        coverage_report["splits"][split_name] = cov
        degradation_report["splits"][split_name] = deg
    test_rmse = degradation_report["splits"]["test"]["rmse_without_gate"]
    unseen_rmse = degradation_report["splits"]["unseen"]["rmse_without_gate"]
    degradation_report["ood_to_id_rmse_ratio"] = unseen_rmse / test_rmse if test_rmse else None
    write_json(model_dir / "coverage.json", coverage_report)
    write_json(model_dir / "ood_degradation.json", degradation_report)

    print(f"model={model_name}  m_f global={margins['global']:.4f}  "
          f"controllers={ {k: round(v, 4) for k, v in margins['controllers'].items()} }  "
          f"n={margins['n_validation']}  one_sided_q95={margins['one_sided_q95']:.4f}")
    print(f"{'split':10} {'cov':>7} {'cov+gate':>9} {'D_OOD':>7} "
          f"{'RMSE':>8} {'RMSE+gate':>10} {'persist':>8}")
    for split_name in ("validation", "test", "unseen"):
        cov = coverage_report["splits"][split_name]
        deg = degradation_report["splits"][split_name]
        print(f"{split_name:10} {cov['without_gate']['coverage']:7.3f} "
              f"{cov['with_gate']['coverage']:9.3f} {deg['d_ood']:7.3f} "
              f"{deg['rmse_without_gate']:8.4f} {deg['rmse_with_gate']:10.4f} "
              f"{deg['rmse_persistence']:8.4f}")

    if not args.no_mlflow:
        metrics: dict[str, Any] = {
            "m_f_global": margins["global"],
            "m_f_one_sided_q95": margins["one_sided_q95"],
            **{f"m_f_{cid}": v for cid, v in margins["controllers"].items()},
        }
        for split_name in ("test", "unseen"):
            cov = coverage_report["splits"][split_name]
            deg = degradation_report["splits"][split_name]
            metrics[f"coverage95_{split_name}"] = cov["without_gate"]["coverage"]
            metrics[f"coverage95_{split_name}_gated"] = cov["with_gate"]["coverage"]
            metrics[f"d_ood_{split_name}"] = deg["d_ood"]
            metrics[f"rmse_{split_name}_gated"] = deg["rmse_with_gate"]
        run_id = log_to_pointed_run(
            model_dir / "mlflow_run.json",
            metrics=metrics,
            artifacts=[
                (model_dir / name, "margin")
                for name in ("error_margin.json", "feature_ranges.json",
                             "coverage.json", "ood_degradation.json")
            ],
        )
        if run_id:
            print("MLflow run", run_id)


if __name__ == "__main__":
    main()
