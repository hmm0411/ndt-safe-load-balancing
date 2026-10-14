#!/usr/bin/env python3
"""Single-sample inference latency of every forecaster (Week 10, M1 step 4).

Measures the runtime path: one row of 27 features, one predict call, no
batching. 50 warm-up calls, then 1000 timed calls with time.perf_counter_ns.

Run it in the agreed testbed window (stable workload, nobody training) so
the p95 reflects real runtime conditions; it feeds horizon_v1.json (V3):

    ndt_busy "M1 latency" && { python scripts/experiments/benchmark_forecast_inference.py \
        --model-version forecast-v1; ndt_done; }
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.twin.forecasting.linear_model import records_to_matrix
from src.twin.forecasting.selection import measure_latency_ms, single_sample_predictors
from src.twin.forecasting.splitter import VALIDATION, split_records
from src.twin.forecasting.tracking import log_to_pointed_run
from src.twin.forecasting.versions import MODEL_VERSION, MODELS_ROOT, PROCESSED_ROOT, dataset_version

DEFAULT_OUTPUT = Path("data/benchmarks/forecast_inference_latency.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_average() -> list[float] | None:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):  # not available on Windows
        return None


def benchmark_row(dataset_path: Path, feature_names: list[str]) -> Any:
    """First validation row: a real feature vector the model was selected on."""
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        records = [dict(row) for row in csv.DictReader(handle)]
    validation = split_records(records)[VALIDATION]
    if not validation:
        raise RuntimeError("validation split is empty")
    return records_to_matrix(validation[:1], feature_names)


def main() -> None:
    args = parse_args()
    model_dir = MODELS_ROOT / args.model_version
    feature_names = list(_read_json(model_dir / "feature_schema.json")["feature_names"])
    manifest = _read_json(model_dir / "training_manifest.json")
    ma_window = int(manifest.get("selected_ma_window", 3))

    from xgboost import XGBRegressor

    xgboost = XGBRegressor()
    xgboost.load_model(str(model_dir / "xgboost.json"))
    predictors = single_sample_predictors(
        feature_names,
        benchmark_row(
            PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv", feature_names
        ),
        ma_window=ma_window,
        linear_model=joblib.load(model_dir / "linear_regression.joblib"),
        scaler=joblib.load(model_dir / "scaler.joblib"),
        random_forest=joblib.load(model_dir / "random_forest.joblib"),
        xgboost=xgboost,
    )

    load_before = _load_average()
    models = {
        name: measure_latency_ms(fn, warmup=args.warmup, repeats=args.repeats)
        for name, fn in predictors.items()
    }
    selection_path = model_dir / "model_selection.json"
    selected = _read_json(selection_path)["selected"] if selection_path.is_file() else None

    result = {
        "schema_version": "1.0",
        "model_version": args.model_version,
        "method": "single_sample_predict_perf_counter_ns",
        "feature_count": len(feature_names),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "selected_model": selected,
        "selected_p95_ms": models[selected]["p95_ms"] if selected in models else None,
        "models": {
            name: {key: round(value, 6) for key, value in stats.items()}
            for name, stats in models.items()
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "git_commit": _git_commit(),
            "measured_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for name, stats in models.items():
        print(
            f"{name:22} p50={stats['p50_ms']:.4f} ms  p95={stats['p95_ms']:.4f} ms  "
            f"p99={stats['p99_ms']:.4f} ms"
        )
    print(f"selected={selected}  p95={result['selected_p95_ms']} ms -> send to Vy (V3)")
    print("Wrote", args.output)

    if not args.no_mlflow:
        run_id = log_to_pointed_run(
            model_dir / "mlflow_run.json",
            metrics={
                f"latency_{name}_{q}": stats[f"{q}_ms"]
                for name, stats in models.items()
                for q in ("p50", "p95", "p99")
            },
            artifacts=[(args.output, "benchmarks")],
        )
        if run_id:
            print("MLflow run", run_id)


if __name__ == "__main__":
    main()
