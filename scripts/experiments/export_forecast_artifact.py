#!/usr/bin/env python3
"""Package a trained model directory as a forecast-v1 artifact (M3, contract C1).

Run after train_forecast_baselines.py and compute_forecast_margin.py on the
same --model-version. Writes model_card.json, metrics.json and adds
history_samples_required to feature_schema.json, then checks the whole
directory against the contract. Exit code 1 when the contract is violated.

    python scripts/experiments/train_forecast_baselines.py --model-version forecast-v1 \
        --run-name forecast-v1 --register --note "..."
    python scripts/experiments/compute_forecast_margin.py --model-version forecast-v1
    python scripts/experiments/export_forecast_artifact.py --model-version forecast-v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from src.twin.forecasting.artifact import (
    build_feature_schema,
    build_model_card,
    sha256_file,
    summarize_metrics,
    validate_artifact,
)
from src.twin.forecasting.tracking import git_tags
from src.twin.forecasting.versions import MODEL_VERSION, MODELS_ROOT, PROCESSED_ROOT, dataset_version

MODEL_CONFIG_PATH = Path("configs/forecasting/models.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument(
        "--model",
        default=None,
        help="model to deploy (default: 'selected' in model_selection.json)",
    )
    parser.add_argument("--config", type=Path, default=MODEL_CONFIG_PATH)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def read_optional(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    model_dir = MODELS_ROOT / args.model_version
    selection = read_json(model_dir / "model_selection.json")
    model_name = args.model or str(selection["selected"])
    manifest = read_json(model_dir / "training_manifest.json")
    pointer = read_optional(model_dir / "mlflow_run.json") or {}
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    registered_names = config.get("mlflow", {}).get("registered_names", {})

    margin = read_optional(model_dir / "error_margin.json")
    if margin is not None and margin.get("model_name") not in (None, model_name):
        print(
            f"error_margin.json was computed for {margin['model_name']!r}, not {model_name!r}; "
            "rerun compute_forecast_margin.py",
            file=sys.stderr,
        )
        return 1

    schema = read_json(model_dir / "feature_schema.json")
    write_json(model_dir / "feature_schema.json", build_feature_schema(schema["feature_names"]))

    scores = {row["model"]: row["selection_score"] for row in selection["models"]}
    dataset = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
    card = build_model_card(
        model_name=model_name,
        model_version=args.model_version,
        dataset_version=str(manifest["dataset_version"]),
        horizon_steps=int(manifest["horizon_steps"]),
        sampling_interval_seconds=1.0,
        mlflow_run_id=pointer.get("run_id"),
        registered_model=registered_names.get(model_name),
        registered_version=pointer.get("registered_versions", {}).get(model_name),
        git_commit=git_tags()["git_commit"],
        dataset_sha256=sha256_file(dataset) if dataset.is_file() else "unknown",
        selection_score=scores.get(model_name),
    )
    write_json(model_dir / "model_card.json", card)

    reports = {
        split_name: read_json(model_dir / f"{split_name}_metrics.json")
        for split_name in ("validation", "test", "unseen")
    }
    write_json(
        model_dir / "metrics.json",
        summarize_metrics(model_name, reports, read_optional(model_dir / "coverage.json")),
    )

    errors = validate_artifact(model_dir)
    for error in errors:
        print("CONTRACT:", error, file=sys.stderr)
    if errors:
        return 1
    print(f"OK {model_dir}: {model_name}, run {card['mlflow_run_id']}, "
          f"registered {card['registered_model']} v{card['registered_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
