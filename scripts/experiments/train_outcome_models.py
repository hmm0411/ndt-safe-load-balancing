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

--stage select-and-test (Day 4; default data/outcome/reports/final/, gitignored) chooses Best
Direct and Best Hybrid by validation S (configs/outcome/evaluation.yaml), writes
model_selection.json BEFORE reading any test row, then evaluates the test split ONCE
(test_lock.json refuses a second evaluation):
  model_selection.json        frozen choice, validation scores, sha256 of models and configs
  test_lock.json              opened / completed, selection hash, output hashes, test summary
  validation_metrics.csv      the five models on validation
  test_metrics.csv            Analytical, Best Direct, Best Hybrid on test
  per_workload_metrics.csv    the three models by workload, destination level, direction,
                              switch share and saturation flags (validation and test)
  model_comparison.csv        RMSE / P95 and improvements H/A, H/ML, ML/A (plan section 58)
  bootstrap_ci.csv            paired bootstrap CI over test attempts (RMSE, S, improvements)
  saturation_sensitivity.csv  all rows / without saturated attempts / saturated only
  overload_cases.csv          every overload_after row of validation and test
  ablation.csv                Best Hybrid without temporal / switch features / switch at all
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome import evaluation, hybrid, residual, selection, trainer  # noqa: E402
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


GROUP_COLUMNS = ("split", "group", "value", "model", "target", "n", "mae", "rmse",
                 "p95_abs_error", "bias")
COMPARISON_COLUMNS = ("split", "target", "direct_model", "hybrid_model", "rmse_analytical",
                      "rmse_direct", "rmse_hybrid", "p95_analytical", "p95_direct", "p95_hybrid",
                      "improvement_h_vs_a_pct", "improvement_h_vs_ml_pct",
                      "improvement_ml_vs_a_pct")
BOOTSTRAP_COLUMNS = ("kind", "model", "reference", "target", "estimate", "ci_low", "ci_high",
                     "share_better")
ABLATION_COLUMNS = ("variant", "split", "target", "n_features", "base", "n", "mae", "rmse",
                    "p95_abs_error", "bias")
FINAL_MODELS = ("analytical", "direct_rf", "direct_xgb", "hybrid_rf", "hybrid_xgb")


def _read_json(path: Path, hint: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: {hint}")
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _git_state() -> dict[str, Any]:
    def git(*cmd: str) -> str:
        result = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True,
                                check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {"commit": git("rev-parse", "HEAD"),
            "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _comparison(split: str, rows: Rows, preds: dict[str, dict[str, list[float]]],
                direct: str, hyb: str, scores: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for target in ALL_TARGETS:
        m = {name: regression_metrics([float(r[target]) for r in rows], preds[name][target])
             for name in ("analytical", direct, hyb)}
        a, d, h = (m[n]["rmse"] for n in ("analytical", direct, hyb))
        out.append({"split": split, "target": target, "direct_model": direct, "hybrid_model": hyb,
                    "rmse_analytical": a, "rmse_direct": d, "rmse_hybrid": h,
                    "p95_analytical": m["analytical"]["p95_abs_error"],
                    "p95_direct": m[direct]["p95_abs_error"], "p95_hybrid": m[hyb]["p95_abs_error"],
                    "improvement_h_vs_a_pct": evaluation.improvement(a, h),
                    "improvement_h_vs_ml_pct": evaluation.improvement(d, h),
                    "improvement_ml_vs_a_pct": evaluation.improvement(a, d)})
    a, d, h = scores["analytical"], scores[direct], scores[hyb]
    out.append({"split": split, "target": "S", "direct_model": direct, "hybrid_model": hyb,
                "rmse_analytical": a, "rmse_direct": d, "rmse_hybrid": h,
                "improvement_h_vs_a_pct": evaluation.improvement(a, h),
                "improvement_h_vs_ml_pct": evaluation.improvement(d, h),
                "improvement_ml_vs_a_pct": evaluation.improvement(a, d)})
    return out


def stage_select_and_test(args: argparse.Namespace) -> int:
    data_config = load_dataset_config(args.config)
    models_config = trainer.load_models_config(args.models_config)
    eval_config = evaluation.load_evaluation_config(args.eval_config)
    if models_config.targets != BASE_TARGETS:
        raise ValueError(f"needs models for all of {BASE_TARGETS}")
    train, validation, _, schema = _load_splits(args.dataset)
    model_features = list(schema["model_features"])
    hybrid_features = residual.hybrid_features(model_features)
    direct_hyper = _read_json(args.direct_dir / "hyperparameters.json", "run --stage direct first")
    hybrid_hyper = _read_json(args.hybrid_dir / "hyperparameters.json", "run --stage hybrid first")
    if direct_hyper.get("model_features") != model_features:
        raise ValueError(f"{args.direct_dir}: Direct models use other features; rerun them")
    if hybrid_hyper.get("hybrid_features") != hybrid_features:
        raise ValueError(f"{args.hybrid_dir}: residual models use other features; rerun them")

    models: dict[str, dict[str, Any]] = {}
    model_files: dict[str, Path] = {}
    for family, short in trainer.SHORT_NAMES.items():
        suffix = trainer.FAMILIES[family].SUFFIX
        models[f"direct_{short}"] = trainer.load_models(args.direct_dir / "models", family,
                                                         BASE_TARGETS, prefix="direct")
        models[f"hybrid_{short}"] = hybrid.load(args.hybrid_dir / "models", family)
        for t in BASE_TARGETS:
            for prefix, folder in (("direct", args.direct_dir), ("residual", args.hybrid_dir)):
                name = f"{prefix}_{short}_{t}{suffix}"
                model_files[f"models/{name}"] = folder / "models" / name

    def predict(name: str, rows: Rows) -> dict[str, list[float]]:
        if name == "analytical":
            return _analytical(rows)
        if name.startswith("direct_"):
            return trainer.predict_targets(models[name], rows, model_features)
        return hybrid.predict(models[name], rows, hybrid_features)

    # ---------------------------------------------------------- selection (validation only)
    val_pred = {name: predict(name, validation) for name in FINAL_MODELS}
    val_scores = _scores({n: {"validation": v} for n, v in val_pred.items()}, validation,
                         data_config)
    stored = {**_read_json(args.direct_dir / "selection_scores.json", "run --stage direct"),
              **_read_json(args.hybrid_dir / "selection_scores.json", "run --stage hybrid")}
    for name in FINAL_MODELS:
        before = float(stored[name]["selection"]["score"])
        if abs(before - val_scores[name]["selection"]["score"]) > 1e-9:
            raise ValueError(f"validation S of {name} is {val_scores[name]['selection']['score']}"
                             f", Day 2/3 wrote {before}: models or dataset changed")
    s_val = {n: val_scores[n]["selection"]["score"] for n in FINAL_MODELS}
    best_direct = evaluation.choose(s_val, eval_config.direct_candidates)
    best_hybrid = evaluation.choose(s_val, eval_config.hybrid_candidates)
    hybrid_family = "random_forest" if best_hybrid == "hybrid_rf" else "xgboost"
    direct_family = "random_forest" if best_direct == "direct_rf" else "xgboost"
    frozen = {
        "rule": "Best Direct / Best Hybrid = lowest validation S (selection weights without "
                "selection_exclude); ties keep the candidate order",
        "selection_weights": data_config.effective_selection_weights,
        "selection_exclude": list(data_config.selection_exclude),
        "candidates": {"direct": list(eval_config.direct_candidates),
                       "hybrid": list(eval_config.hybrid_candidates)},
        "validation_scores": {n: {"S": s_val[n], "S_plan": val_scores[n]["plan"]["score"]}
                              for n in FINAL_MODELS},
        "best_direct": best_direct,
        "best_hybrid": best_hybrid,
        "parameters": {
            best_direct: {t: direct_hyper["selected"][f"{direct_family}/{t}"]["params"]
                          for t in BASE_TARGETS},
            best_hybrid: {t: hybrid_hyper["selected"][f"{hybrid_family}/{t}"]["params"]
                          for t in BASE_TARGETS},
        },
        "bootstrap": {"resamples": eval_config.resamples, "seed": eval_config.seed,
                      "confidence": eval_config.confidence},
        "ablation": {name: {"drop": list(spec.drop), "persistence_base": spec.persistence_base}
                     for name, spec in eval_config.ablation.items()},
        "sha256": selection.tree_sha256({
            "outcome_dataset.csv": args.dataset / "outcome_dataset.csv",
            "feature_schema.json": args.dataset / "feature_schema.json",
            "configs/dataset.yaml": args.config, "configs/models.yaml": args.models_config,
            "configs/evaluation.yaml": args.eval_config,
            "direct/hyperparameters.json": args.direct_dir / "hyperparameters.json",
            "hybrid/hyperparameters.json": args.hybrid_dir / "hyperparameters.json",
            **model_files}),
    }
    selection_sha = selection.canonical_sha256(frozen)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    selection.check_can_open(out, selection_sha)  # refuses before anything is overwritten
    (out / selection.SELECTION_FILE).write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(), "git": _git_state(),
        "dataset": str(args.dataset),
        "rows": {"train": len(train), "validation": len(validation)},
        "frozen_sha256": selection_sha, "frozen": frozen}, indent=2) + "\n", encoding="utf-8")
    print("validation S: " + ", ".join(f"{n} {s_val[n]:.3f}" for n in FINAL_MODELS))
    print(f"Best Direct = {best_direct}, Best Hybrid = {best_hybrid} "
          f"(frozen {selection_sha[:12]}, written to {out / selection.SELECTION_FILE})")

    # ---------------------------------------------------------- the one test evaluation
    lock = selection.open_test(out, selection_sha)
    test = [r for r in load_dataset(args.dataset / "outcome_dataset.csv") if r["split"] == "test"]
    if not test:
        raise ValueError("no test rows")
    roles = ("analytical", best_direct, best_hybrid)
    test_pred = {name: predict(name, test) for name in roles}
    val_roles = {name: val_pred[name] for name in roles}
    test_scores = _scores({n: {"validation": v} for n, v in test_pred.items()}, test,
                          data_config)
    s_test = {n: test_scores[n]["selection"]["score"] for n in roles}

    # ablation: Best Hybrid's family and per-target parameters, refitted on train
    params = {t: hybrid_hyper["selected"][f"{hybrid_family}/{t}"]["params"] for t in BASE_TARGETS}
    module = trainer.FAMILIES[hybrid_family]

    def refit(features: Sequence[str], rows: Rows) -> dict[str, Any]:
        x = trainer.matrix(rows, features)
        fitted = {}
        for t in BASE_TARGETS:
            model = module.make(params[t])
            model.fit(x, residual.residual_values(rows, t))
            fitted[t] = model
        return fitted

    full = refit(hybrid_features, train)
    refit_diff = max(abs(a - b) for t in BASE_TARGETS for a, b in zip(
        hybrid.predict(full, validation, hybrid_features)[t], val_pred[best_hybrid][t]))
    variants: dict[str, dict[str, Any]] = {
        "analytical": {"n_features": 0, "base": "analytical",
                       "pred": {"validation": val_pred["analytical"],
                                "test": test_pred["analytical"]}},
        best_direct: {"n_features": len(model_features), "base": "none",
                      "pred": {"validation": val_pred[best_direct],
                               "test": test_pred[best_direct]}},
        "hybrid_full": {"n_features": len(hybrid_features), "base": "analytical",
                                  "pred": {"validation": val_pred[best_hybrid],
                                           "test": test_pred[best_hybrid]}},
    }
    for name, spec in eval_config.ablation.items():
        feats = residual.hybrid_features([f for f in model_features if f not in spec.drop])

        def rebase(rows: Rows, spec: evaluation.AblationSpec = spec) -> Rows:
            if not spec.persistence_base:
                return rows
            return evaluation.with_base(rows, evaluation.persistence_base(rows))

        fitted = refit(feats, rebase(train))
        variants[name] = {"n_features": len(feats),
                          "base": "persistence" if spec.persistence_base else "analytical",
                          "pred": {"validation": hybrid.predict(fitted, rebase(validation), feats),
                                   "test": hybrid.predict(fitted, rebase(test), feats)}}
    ablation_rows: list[dict[str, Any]] = []
    ablation_scores: dict[str, dict[str, float]] = {}
    for name, v in variants.items():
        ablation_scores[name] = {}
        for split, rows in (("validation", validation), ("test", test)):
            for row in trainer.metrics_table(name, split, rows, v["pred"][split]):
                ablation_rows.append({"variant": name, "n_features": v["n_features"],
                                      "base": v["base"], **row})
            score = _scores({name: {"validation": v["pred"][split]}}, rows,
                            data_config)[name]["selection"]["score"]
            ablation_scores[name][split] = score
            ablation_rows.append({"variant": name, "split": split, "target": "S",
                                  "n_features": v["n_features"], "base": v["base"],
                                  "rmse": score})

    weights = data_config.effective_selection_weights
    comparisons = [("analytical", best_hybrid), (best_direct, best_hybrid),
                   ("analytical", best_direct)]
    boot = evaluation.paired_bootstrap(test, test_pred, comparisons, weights,
                                       data_config.selection_epsilon, eval_config.resamples,
                                       eval_config.seed, eval_config.confidence)
    files = {
        "validation_metrics.csv": (
            [r for n in FINAL_MODELS
             for r in trainer.metrics_table(n, "validation", validation, val_pred[n])],
            METRIC_COLUMNS),
        "test_metrics.csv": (
            [r for n in roles for r in trainer.metrics_table(n, "test", test, test_pred[n])],
            METRIC_COLUMNS),
        "per_workload_metrics.csv": (
            evaluation.group_metrics(validation, val_roles, "validation", eval_config.group_keys)
            + evaluation.group_metrics(test, test_pred, "test", eval_config.group_keys),
            GROUP_COLUMNS),
        "model_comparison.csv": (
            _comparison("validation", validation, val_pred, best_direct, best_hybrid, s_val)
            + _comparison("test", test, test_pred, best_direct, best_hybrid, s_test),
            COMPARISON_COLUMNS),
        "bootstrap_ci.csv": (boot, BOOTSTRAP_COLUMNS),
        "saturation_sensitivity.csv": (
            evaluation.saturation_sensitivity(validation, val_roles, "validation")
            + evaluation.saturation_sensitivity(test, test_pred, "test"), GROUP_COLUMNS),
        "ablation.csv": (ablation_rows, ABLATION_COLUMNS),
    }
    cases = (evaluation.overload_cases(validation, val_roles, "validation",
                                       data_config.overload_threshold)
             + evaluation.overload_cases(test, test_pred, "test",
                                         data_config.overload_threshold))
    case_columns = ["split", "attempt_id", "workload", "phase", "direction",
                    "destination_load_level", "switch_share_group", "saturated_before",
                    "saturated_after", "dst_u_mean_pre", "u_dst_post"]
    for name in roles:
        case_columns += [f"pred_{name}", f"error_{name}", f"flags_overload_{name}"]
    for name, (table_rows, columns) in files.items():
        _write_csv(out / name, table_rows, columns)
    _write_csv(out / "overload_cases.csv", cases, case_columns)
    outputs = {name: out / name for name in [*files, "overload_cases.csv",
                                             selection.SELECTION_FILE]}
    test_rmse = {n: {t: regression_metrics([float(r[t]) for r in test], test_pred[n][t])["rmse"]
                     for t in ALL_TARGETS} for n in roles}
    lock = selection.complete_test(out, outputs, {
        "n_test": len(test), "best_direct": best_direct, "best_hybrid": best_hybrid,
        "test_S": s_test, "test_rmse": test_rmse,
        "ablation_S": ablation_scores, "refit_max_abs_diff": refit_diff})

    # ---------------------------------------------------------- report
    print(f"\ntest opened (attempt {lock['attempt']}), n = {len(test)}")
    print(f"{'model':<14}" + "".join(f"{t:>14}" for t in ALL_TARGETS) + f"{'S':>9}{'S_plan':>9}")
    for n in roles:
        print(f"{n:<14}" + "".join(f"{test_rmse[n][t]:>14.4f}" for t in ALL_TARGETS)
              + f"{s_test[n]:>9.3f}{test_scores[n]['plan']['score']:>9.3f}")
    level = int(round(eval_config.confidence * 100))
    print(f"\nimprovement on test, % of RMSE (paired bootstrap {level} % CI, "
          f"{eval_config.resamples} resamples; share = resamples where the model is better)")
    for reference, model in comparisons:
        for row in boot:
            if (row["kind"] == "improvement" and row["reference"] == reference
                    and row["model"] == model and row["target"] in
                    ("u_dst_post", "u_src_post", "i_post", "S")):
                print(f"  {model} vs {reference:<11} {row['target']:<11}"
                      f"{row['estimate']:>8.1f}  [{row['ci_low']:>7.1f}, {row['ci_high']:>7.1f}]"
                      f"  share {row['share_better']:.2f}")
    sat = {r["value"]: r for r in files["saturation_sensitivity.csv"][0]
           if r["split"] == "test" and r["model"] == best_hybrid and r["target"] == "u_src_post"}
    print("\nsaturation (test, " + best_hybrid + ", u_src_post RMSE): "
          + ", ".join(f"{k} n={int(v['n'])} {v['rmse']:.4f}" for k, v in sat.items()))
    print("\noverload cases (u_dst_post >= "
          f"{data_config.overload_threshold:g}): {len(cases)}")
    for case in cases:
        print(f"  {case['split']:<10} {case['attempt_id']:<40} actual {case['u_dst_post']:.3f}  "
              + "  ".join(f"{n} {case[f'pred_{n}']:.3f}" for n in roles))
    print("\nablation S (validation / test)")
    for name in variants:
        print(f"  {name:<28}{variants[name]['n_features']:>4} features  base "
              f"{variants[name]['base']:<12}{ablation_scores[name]['validation']:>7.3f}"
              f"{ablation_scores[name]['test']:>8.3f}")
    print(f"  refit of the full Hybrid vs saved models: max |diff| {refit_diff:.2e}")
    print(f"\ntest_lock: {lock['status']} at {lock['completed_at']}")
    print(f"wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--stage", choices=("direct", "hybrid", "select-and-test"),
                        required=True)
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/outcome/models.yaml"))
    parser.add_argument("--eval-config", type=Path,
                        default=Path("configs/outcome/evaluation.yaml"))
    parser.add_argument("--direct-dir", type=Path, default=Path("data/outcome/reports/direct"),
                        help="Day 2 output (read by --stage hybrid and select-and-test)")
    parser.add_argument("--hybrid-dir", type=Path, default=Path("data/outcome/reports/hybrid"),
                        help="Day 3 output (read by --stage select-and-test)")
    parser.add_argument("--out", type=Path,
                        help="default data/outcome/reports/<stage>")
    args = parser.parse_args()
    if args.out is None:
        args.out = Path("data/outcome/reports") / (
            "final" if args.stage == "select-and-test" else args.stage)
    stages = {"direct": stage_direct, "hybrid": stage_hybrid,
              "select-and-test": stage_select_and_test}
    try:
        return stages[args.stage](args)
    except (FileNotFoundError, ValueError, selection.LockedTestError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
