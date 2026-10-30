#!/usr/bin/env python3
"""Train the Post-Migration Outcome models (Week 12).

Stages (one per day of the plan; only the ones written so far are accepted):
  direct   Day 2: Direct-RF and Direct-XGB, one model per base target, grid search
           fitted on TRAIN, chosen on VALIDATION. Test rows are dropped at load time.

    python3 scripts/experiments/train_outcome_models.py --stage direct

Outputs (default data/outcome/reports/direct/, gitignored):
  hyperparameter_search.csv   every candidate with its validation metrics
  hyperparameters.json        selected parameters per family and target (frozen for Day 4)
  direct_ml_metrics.csv       Analytical, Direct-RF, Direct-XGB on train and validation, 6 targets
  selection_scores.json       validation S (without the targets in selection_exclude, used to
                              choose models on Day 4) and S_plan (plan weights) per model
  models/                     direct_<rf|xgb>_<target>.(joblib|json)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome import trainer  # noqa: E402
from src.twin.outcome.dataset import load_dataset, load_dataset_config  # noqa: E402
from src.twin.outcome.metrics import selection_score  # noqa: E402
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS  # noqa: E402

METRIC_COLUMNS = ("model", "split", "target", "n", "mae", "rmse", "p95_abs_error", "bias", "r2")
SEARCH_COLUMNS = ("family", "target", "candidate", "params", "validation_n", "validation_mae",
                  "validation_rmse", "validation_p95_abs_error", "validation_bias",
                  "validation_r2")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _analytical(rows: Sequence[dict[str, Any]]) -> dict[str, list[float]]:
    return {t: [float(r[f"analytical_{t}"]) for r in rows] for t in ALL_TARGETS}


def stage_direct(args: argparse.Namespace) -> int:
    data_config = load_dataset_config(args.config)
    models_config = trainer.load_models_config(args.models_config)
    every = load_dataset(args.dataset / "outcome_dataset.csv")
    train = [r for r in every if r["split"] == "train"]
    validation = [r for r in every if r["split"] == "validation"]
    dropped = len(every) - len(train) - len(validation)
    del every  # test rows are not kept in memory past this point
    schema = json.loads((args.dataset / "feature_schema.json").read_text(encoding="utf-8"))
    features = list(schema["model_features"])
    if not train or not validation:
        raise ValueError("need train and validation rows")

    best, records = trainer.search(train, validation, features, models_config)
    out: Path = args.out
    model_dir = out / "models"
    files = trainer.save_models(best, model_dir, prefix="direct")

    predictions: dict[str, dict[str, dict[str, list[float]]]] = {
        "analytical": {"train": _analytical(train), "validation": _analytical(validation)}}
    reload_ok = True
    for family in trainer.FAMILIES:
        name = f"direct_{trainer.SHORT_NAMES[family]}"
        fitted = {t: best[(family, t)].model for t in models_config.targets}
        predictions[name] = {"train": trainer.predict_targets(fitted, train, features),
                             "validation": trainer.predict_targets(fitted, validation, features)}
        loaded = trainer.load_models(model_dir, family, models_config.targets, prefix="direct")
        again = trainer.predict_targets(loaded, validation, features)
        reload_ok &= all(abs(a - b) <= 1e-9 for t in BASE_TARGETS
                         for a, b in zip(again[t], predictions[name]["validation"][t]))

    table: list[dict[str, Any]] = []
    for name, by_split in predictions.items():
        table += trainer.metrics_table(name, "train", train, by_split["train"])
        table += trainer.metrics_table(name, "validation", validation, by_split["validation"])
    def score(by_split: dict[str, dict[str, list[float]]],
              weights: dict[str, float]) -> dict[str, float]:
        truth = {t: [float(r[t]) for r in validation] for t in weights}
        return selection_score(truth, {t: by_split["validation"][t] for t in weights},
                               weights, data_config.selection_epsilon)

    scores = {name: {"selection": score(by_split, data_config.effective_selection_weights),
                     "plan": score(by_split, data_config.selection_weights)}
              for name, by_split in predictions.items()}

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "hyperparameter_search.csv",
               [{**r, "params": json.dumps(trainer.json_params(r["params"]))} for r in records],
               SEARCH_COLUMNS)
    _write_csv(out / "direct_ml_metrics.csv", table, METRIC_COLUMNS)
    hyper = {
        "stage": "direct",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "feature_schema": schema["feature_schema"],
        "model_features": features,
        "selection": "lowest validation RMSE of the target; train only for fitting; test unused",
        "selection_score": {"weights": data_config.effective_selection_weights,
                            "excluded": list(data_config.selection_exclude),
                            "plan_weights": data_config.selection_weights},
        "rows": {"train": len(train), "validation": len(validation), "test_dropped": dropped},
        "candidates_per_target": {f: len(c.candidates())
                                  for f, c in models_config.families.items()},
        "selected": {f"{fam}/{t}": {"params": trainer.json_params(fit.params),
                                    "validation_rmse": fit.validation_rmse}
                     for (fam, t), fit in sorted(best.items())},
        "files": files,
        "save_load_identical": reload_ok,
    }
    (out / "hyperparameters.json").write_text(json.dumps(hyper, indent=2) + "\n",
                                              encoding="utf-8")
    (out / "selection_scores.json").write_text(json.dumps(scores, indent=2) + "\n",
                                               encoding="utf-8")

    print(f"train {len(train)}, validation {len(validation)}, test rows dropped {dropped}; "
          f"{len(features)} model features")
    print(f"candidates per target: {hyper['candidates_per_target']}, fitted {len(records)}")
    print(f"\nvalidation RMSE (n = {len(validation)})")
    print(f"{'model':<12}" + "".join(f"{t:>14}" for t in ALL_TARGETS) + f"{'S':>9}{'S_plan':>9}")
    for name in predictions:
        rmse = {r["target"]: r["rmse"] for r in table
                if r["model"] == name and r["split"] == "validation"}
        print(f"{name:<12}" + "".join(f"{rmse[t]:>14.4f}" for t in ALL_TARGETS)
              + f"{scores[name]['selection']['score']:>9.3f}"
              + f"{scores[name]['plan']['score']:>9.3f}")
    excluded = ", ".join(data_config.selection_exclude) or "none"
    print(f"S = selection score without {excluded} (chooses models); S_plan = plan weights")
    print(f"\nsave/load identical: {reload_ok}")
    print(f"wrote {out}")
    return 0 if reload_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stage", choices=("direct",), required=True,
                        help="hybrid (Day 3) and select-and-test (Day 4) come later")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/outcome/models.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/outcome/reports/direct"))
    args = parser.parse_args()
    try:
        return stage_direct(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
