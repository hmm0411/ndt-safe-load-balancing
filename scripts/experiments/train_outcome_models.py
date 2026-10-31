#!/usr/bin/env python3
"""Train the Post-Migration Outcome models (Week 12).

Stages (one per day of the plan; only the ones written so far are accepted):
  direct   Day 2: Direct-RF and Direct-XGB, one model per base target, grid search
           fitted on TRAIN, chosen on VALIDATION. Test rows are dropped at load time.
  hybrid   Day 3: residual RF and XGB (target r = y - y_A, features = model features plus
           the four Analytical predictions), same grids, same rule; Hybrid = y_A + r_hat.
           Needs the Day 2 output (--direct-dir) to compare against Direct.

    python3 scripts/experiments/train_outcome_models.py --stage direct
    python3 scripts/experiments/train_outcome_models.py --stage hybrid

Outputs of --stage direct (default data/outcome/reports/direct/, gitignored):
  hyperparameter_search.csv   every candidate with its validation metrics
  hyperparameters.json        selected parameters per family and target (frozen for Day 4)
  direct_ml_metrics.csv       Analytical, Direct-RF, Direct-XGB on train and validation, 6 targets
  selection_scores.json       validation S (without the targets in selection_exclude, used to
                              choose models on Day 4) and S_plan (plan weights) per model
  models/                     direct_<rf|xgb>_<target>.(joblib|json)

Outputs of --stage hybrid (default data/outcome/reports/hybrid/, gitignored):
  hyperparameter_search.csv   every residual candidate (metrics are on the residual)
  hyperparameters.json        selected residual parameters, Hybrid features, floor counts
  hybrid_metrics.csv          Analytical, Direct-RF/XGB, Hybrid-RF/XGB on train and validation
  residual_analysis.csv       Analytical residual by workload, switch share group and
                              destination level, with the RMSE left by each model
  selection_scores.json       S and S_plan of the five models on validation
  models/                     residual_<rf|xgb>_<target>.(joblib|json)
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

from src.twin.outcome import hybrid, residual, trainer  # noqa: E402
from src.twin.outcome.dataset import load_dataset, load_dataset_config  # noqa: E402
from src.twin.outcome.metrics import quantile, regression_metrics, selection_score  # noqa: E402
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


Rows = list[dict[str, Any]]
Predictions = dict[str, dict[str, dict[str, list[float]]]]  # model -> split -> target -> values


def _load_splits(dataset: Path) -> tuple[Rows, Rows, int, dict[str, Any]]:
    """Train and validation rows, number of test rows dropped, feature schema."""
    every = load_dataset(dataset / "outcome_dataset.csv")
    train = [r for r in every if r["split"] == "train"]
    validation = [r for r in every if r["split"] == "validation"]
    dropped = len(every) - len(train) - len(validation)
    del every  # test rows are not kept in memory past this point
    schema = json.loads((dataset / "feature_schema.json").read_text(encoding="utf-8"))
    if not train or not validation:
        raise ValueError("need train and validation rows")
    return train, validation, dropped, schema


def _scores(predictions: Predictions, validation: Rows,
            data_config: Any) -> dict[str, dict[str, dict[str, float]]]:
    def score(values: dict[str, list[float]], weights: dict[str, float]) -> dict[str, float]:
        truth = {t: [float(r[t]) for r in validation] for t in weights}
        return selection_score(truth, {t: values[t] for t in weights}, weights,
                               data_config.selection_epsilon)

    return {name: {"selection": score(by_split["validation"],
                                      data_config.effective_selection_weights),
                   "plan": score(by_split["validation"], data_config.selection_weights)}
            for name, by_split in predictions.items()}


def _table(predictions: Predictions, train: Rows, validation: Rows) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for name, by_split in predictions.items():
        table += trainer.metrics_table(name, "train", train, by_split["train"])
        table += trainer.metrics_table(name, "validation", validation, by_split["validation"])
    return table


def _print_validation(predictions: Predictions, table: list[dict[str, Any]],
                      scores: dict[str, dict[str, dict[str, float]]], n: int,
                      excluded: Sequence[str]) -> None:
    print(f"\nvalidation RMSE (n = {n})")
    print(f"{'model':<12}" + "".join(f"{t:>14}" for t in ALL_TARGETS) + f"{'S':>9}{'S_plan':>9}")
    for name in predictions:
        rmse = {r["target"]: r["rmse"] for r in table
                if r["model"] == name and r["split"] == "validation"}
        print(f"{name:<12}" + "".join(f"{rmse[t]:>14.4f}" for t in ALL_TARGETS)
              + f"{scores[name]['selection']['score']:>9.3f}"
              + f"{scores[name]['plan']['score']:>9.3f}")
    print(f"S = selection score without {', '.join(excluded) or 'none'} (chooses models); "
          "S_plan = plan weights")


def stage_direct(args: argparse.Namespace) -> int:
    data_config = load_dataset_config(args.config)
    models_config = trainer.load_models_config(args.models_config)
    train, validation, dropped, schema = _load_splits(args.dataset)
    features = list(schema["model_features"])

    best, records = trainer.search(train, validation, features, models_config)
    out: Path = args.out
    model_dir = out / "models"
    files = trainer.save_models(best, model_dir, prefix="direct")

    predictions: Predictions = {
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

    table = _table(predictions, train, validation)
    scores = _scores(predictions, validation, data_config)

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
    _print_validation(predictions, table, scores, len(validation), data_config.selection_exclude)
    print(f"\nsave/load identical: {reload_ok}")
    print(f"wrote {out}")
    return 0 if reload_ok else 1


RESIDUAL_GROUPS = ("workload", "switch_share_group", "destination_load_level")
RESIDUAL_COLUMNS = ("split", "group", "value", "target", "n", "residual_mean", "residual_std",
                    "residual_mean_abs", "residual_p95_abs")


def residual_analysis(rows_by_split: dict[str, Rows],
                      predictions: Predictions) -> list[dict[str, Any]]:
    """Analytical residual r = y - y_A per group, and the RMSE each model leaves there."""
    out: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        groups: list[tuple[str, str, list[int]]] = [("all", "all", list(range(len(rows))))]
        for key in RESIDUAL_GROUPS:
            for value in sorted({str(r[key]) for r in rows}):
                groups.append((key, value, [i for i, r in enumerate(rows) if str(r[key]) == value]))
        for key, value, index in groups:
            members = [rows[i] for i in index]
            for target in BASE_TARGETS:
                r = [float(v) for v in residual.residual_values(members, target)]
                mean = sum(r) / len(r)
                record: dict[str, Any] = {
                    "split": split, "group": key, "value": value, "target": target,
                    "n": len(r), "residual_mean": mean,
                    "residual_std": (sum((v - mean) ** 2 for v in r) / len(r)) ** 0.5,
                    "residual_mean_abs": sum(abs(v) for v in r) / len(r),
                    "residual_p95_abs": quantile([abs(v) for v in r], 0.95),
                }
                truth = [float(m[target]) for m in members]
                for name, by_split in predictions.items():
                    pred = [by_split[split][target][i] for i in index]
                    record[f"rmse_{name}"] = regression_metrics(truth, pred)["rmse"]
                out.append(record)
    return out


def stage_hybrid(args: argparse.Namespace) -> int:
    data_config = load_dataset_config(args.config)
    models_config = trainer.load_models_config(args.models_config)
    if models_config.targets != BASE_TARGETS:
        raise ValueError(f"hybrid needs residual models for all of {BASE_TARGETS}")
    train, validation, dropped, schema = _load_splits(args.dataset)
    model_features = list(schema["model_features"])
    features = residual.hybrid_features(model_features)

    direct_hyper_path = args.direct_dir / "hyperparameters.json"
    if not direct_hyper_path.is_file():
        raise FileNotFoundError(f"{direct_hyper_path} not found: run --stage direct first")
    direct_hyper = json.loads(direct_hyper_path.read_text(encoding="utf-8"))
    if direct_hyper.get("model_features") != model_features:
        raise ValueError(f"{args.direct_dir}: Direct models use other features than "
                         f"{args.dataset}; rerun --stage direct")

    best, records = trainer.search(train, validation, features, models_config,
                                   target_values=residual.residual_values)
    out: Path = args.out
    model_dir = out / "models"
    files = hybrid.save(best, model_dir)

    rows_by_split = {"train": train, "validation": validation}
    predictions: Predictions = {
        "analytical": {"train": _analytical(train), "validation": _analytical(validation)}}
    for family in trainer.FAMILIES:
        direct = trainer.load_models(args.direct_dir / "models", family, BASE_TARGETS,
                                     prefix="direct")
        predictions[f"direct_{trainer.SHORT_NAMES[family]}"] = {
            split: trainer.predict_targets(direct, rows, model_features)
            for split, rows in rows_by_split.items()}
    floors: dict[str, dict[str, dict[str, int]]] = {}
    reload_ok = True
    for family in trainer.FAMILIES:
        name = f"hybrid_{trainer.SHORT_NAMES[family]}"
        fitted = {t: best[(family, t)].model for t in BASE_TARGETS}
        predictions[name] = {split: hybrid.predict(fitted, rows, features)
                             for split, rows in rows_by_split.items()}
        floors[name] = {split: hybrid.floored(hybrid.analytical_base(rows),
                                              hybrid.predict_residuals(fitted, rows, features))
                        for split, rows in rows_by_split.items()}
        again = hybrid.predict(hybrid.load(model_dir, family), validation, features)
        reload_ok &= all(abs(a - b) <= 1e-9 for t in BASE_TARGETS
                         for a, b in zip(again[t], predictions[name]["validation"][t]))

    table = _table(predictions, train, validation)
    scores = _scores(predictions, validation, data_config)
    analysis = residual_analysis(rows_by_split, predictions)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "hyperparameter_search.csv",
               [{**r, "params": json.dumps(trainer.json_params(r["params"]))} for r in records],
               SEARCH_COLUMNS)
    _write_csv(out / "hybrid_metrics.csv", table, METRIC_COLUMNS)
    _write_csv(out / "residual_analysis.csv", analysis,
               RESIDUAL_COLUMNS + tuple(f"rmse_{name}" for name in predictions))
    hyper = {
        "stage": "hybrid",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "feature_schema": schema["feature_schema"],
        "model_features": model_features,
        "hybrid_features": features,
        "residual_target": "r = y - analytical_y for each base target; y_H = max(0, y_A + r_hat)",
        "selection": "lowest validation RMSE of the residual of the target; train only for "
                     "fitting; test unused",
        "selection_score": {"weights": data_config.effective_selection_weights,
                            "excluded": list(data_config.selection_exclude),
                            "plan_weights": data_config.selection_weights},
        "direct_reference": {"dir": str(args.direct_dir),
                             "created_at": direct_hyper.get("created_at")},
        "rows": {"train": len(train), "validation": len(validation), "test_dropped": dropped},
        "candidates_per_target": {f: len(c.candidates())
                                  for f, c in models_config.families.items()},
        "selected": {f"{fam}/{t}": {"params": trainer.json_params(fit.params),
                                    "validation_residual_rmse": fit.validation_rmse}
                     for (fam, t), fit in sorted(best.items())},
        "floored_at_zero": floors,
        "files": files,
        "save_load_identical": reload_ok,
    }
    (out / "hyperparameters.json").write_text(json.dumps(hyper, indent=2) + "\n",
                                              encoding="utf-8")
    (out / "selection_scores.json").write_text(json.dumps(scores, indent=2) + "\n",
                                               encoding="utf-8")

    print(f"train {len(train)}, validation {len(validation)}, test rows dropped {dropped}; "
          f"{len(model_features)} model features + {len(residual.ANALYTICAL_FEATURES)} "
          f"analytical = {len(features)} hybrid features")
    print(f"candidates per target: {hyper['candidates_per_target']}, fitted {len(records)}")
    _print_validation(predictions, table, scores, len(validation), data_config.selection_exclude)

    def lowest(prefix: str) -> str:
        return min((n for n in predictions if n.startswith(prefix)),
                   key=lambda n: scores[n]["selection"]["score"])

    best_direct, best_hybrid = lowest("direct_"), lowest("hybrid_")
    s_direct = scores[best_direct]["selection"]["score"]
    s_hybrid = scores[best_hybrid]["selection"]["score"]
    usrc = {r["model"]: r["rmse"] for r in table
            if r["split"] == "validation" and r["target"] == "u_src_post"}
    print(f"\nlowest validation S: Direct {best_direct} {s_direct:.3f}, "
          f"Hybrid {best_hybrid} {s_hybrid:.3f} -> Hybrid "
          + ("lower" if s_hybrid < s_direct else "NOT lower"))
    print(f"u_src_post RMSE: analytical {usrc['analytical']:.4f}, {best_direct} "
          f"{usrc[best_direct]:.4f}, {best_hybrid} {usrc[best_hybrid]:.4f}")
    print("(validation only; Best Direct / Best Hybrid are fixed and tested on Day 4)")
    n_floored = sum(v for by_split in floors.values() for counts in by_split.values()
                    for v in counts.values())
    print(f"values floored at 0: {n_floored}")
    print(f"\nsave/load identical: {reload_ok}")
    print(f"wrote {out}")
    return 0 if reload_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stage", choices=("direct", "hybrid"), required=True,
                        help="select-and-test (Day 4) comes later")
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/outcome/models.yaml"))
    parser.add_argument("--direct-dir", type=Path, default=Path("data/outcome/reports/direct"),
                        help="Day 2 output read by --stage hybrid")
    parser.add_argument("--out", type=Path,
                        help="default data/outcome/reports/<stage>")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("data/outcome/reports") / args.stage
    try:
        return stage_hybrid(args) if args.stage == "hybrid" else stage_direct(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
