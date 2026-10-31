#!/usr/bin/env python3
"""Margins, test coverage and the post-migration-v1 artifact (Week 12, Day 5).

Reads the frozen Day 4 choice (data/outcome/reports/final/model_selection.json, with a
completed test_lock.json) and the saved Day 2 / Day 3 models; chooses nothing new.

  1. m_U, m_R on VALIDATION for Analytical, Best Direct, Best Hybrid (one-sided, split
     conformal, configs/outcome/artifact.yaml) -> uncertainty_margin.json
  2. coverage of those margins on TEST (overall, by group, saturated, overload) and what a
     gate "prediction + m_U >= 1" would have rejected -> coverage_metrics.json
  3. the artifact data/models/outcome/post-migration-v1/ for the deployed model (lowest
     validation S), checked by reloading it; inference_latency.json is added by the runtime
     side (Vy) and is reported as pending

    python3 scripts/experiments/export_outcome_artifact.py
    python3 scripts/experiments/export_outcome_artifact.py --check     # artifact usable?
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome import artifact as art  # noqa: E402
from src.twin.outcome import hybrid, margins, predictor, residual, trainer  # noqa: E402
from src.twin.outcome.dataset import load_dataset, load_dataset_config  # noqa: E402
from src.twin.outcome.frozen import (  # noqa: E402
    ROLES,
    FinalPaths,
    FrozenModels,
    deployed_model,
    family_of,
    load_frozen,
)
from src.twin.outcome.targets import BASE_TARGETS, target_schema  # noqa: E402


def _git() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        result = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True,
                                check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {"commit": run("rev-parse", "HEAD"),
            "tracked_changes": bool(run("status", "--porcelain", "--untracked-files=no"))}


def check(artifact_dir: Path, capacity_file: Path) -> int:
    running = art.capacity_version(capacity_file)
    problem = predictor.artifact_problem(artifact_dir, running)
    missing = [e for e in predictor.ARTIFACT_ENTRIES if not (artifact_dir / e).exists()]
    if problem is None:
        try:
            runtime = predictor._load_runtime(artifact_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"artifact {artifact_dir}: files present but not loadable: {exc}")
            return 1
        print(f"artifact {artifact_dir}: READY for the runtime predictor "
              f"(capacity {running}, m_U {runtime.m_u:.4f}, m_R {runtime.m_r:.4f} ms, "
              f"{len(runtime.hybrid_features)} hybrid features)")
        return 0
    print(f"artifact {artifact_dir}: {problem}" + (f"; missing {missing}" if missing else ""))
    if missing == [art.LATENCY_FILE]:
        print("only inference_latency.json is missing: it is written by the runtime side "
              "(Vy, Day 5 inference benchmark)")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/outcome/models.yaml"))
    parser.add_argument("--eval-config", type=Path,
                        default=Path("configs/outcome/evaluation.yaml"))
    parser.add_argument("--artifact-config", type=Path,
                        default=Path("configs/outcome/artifact.yaml"))
    parser.add_argument("--direct-dir", type=Path, default=Path("data/outcome/reports/direct"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("data/outcome/reports/hybrid"))
    parser.add_argument("--final-dir", type=Path, default=Path("data/outcome/reports/final"))
    parser.add_argument("--capacity-file", type=Path,
                        default=Path("data/benchmarks/controller_capacity.json"))
    parser.add_argument("--artifact-dir", type=Path,
                        help="default: artifact_dir of --artifact-config")
    parser.add_argument("--check", action="store_true",
                        help="only report whether the artifact is usable by the predictor")
    args = parser.parse_args()
    try:
        config = art.load_artifact_config(args.artifact_config)
        artifact_dir: Path = args.artifact_dir or config.artifact_dir
        if args.check:
            return check(artifact_dir, args.capacity_file)
        return export(args, config, artifact_dir)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 2


def export(args: argparse.Namespace, config: art.ArtifactConfig, artifact_dir: Path) -> int:
    if config.model_version != predictor.MODEL_VERSION:
        raise ValueError(f"artifact.yaml model_version {config.model_version} != predictor "
                         f"{predictor.MODEL_VERSION}")
    if artifact_dir.exists():
        raise FileExistsError(f"{artifact_dir} exists: an artifact is frozen once written; "
                              "bump model_version instead of overwriting it")
    data_config = load_dataset_config(args.config)
    paths = FinalPaths(args.dataset, args.config, args.models_config, args.eval_config,
                       args.direct_dir, args.hybrid_dir, args.final_dir)
    frozen = load_frozen(paths)
    deployed = deployed_model(frozen)
    print(f"Day 4 selection {frozen.selection['frozen_sha256'][:12]} (test completed "
          f"{frozen.lock.get('completed_at')}): Best Direct {frozen.best_direct}, Best Hybrid "
          f"{frozen.best_hybrid}; deployed (lowest validation S) {deployed}")

    rows = load_dataset(args.dataset / "outcome_dataset.csv")
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "validation", "test")}
    models = FrozenModels(paths, frozen)
    pred = {s: models.predict_roles(split[s]) for s in ("validation", "test")}
    names = [frozen.name_of(role) for role in ROLES]

    # 1. margins on validation -------------------------------------------------------
    level = config.margin_level
    margin_table: dict[str, dict[str, Any]] = {}
    for name in names:
        margin_table[name] = {}
        for key, target in config.margins.items():
            margin_table[name][key] = margins.one_sided_margin(
                [float(r[target]) for r in split["validation"]], pred["validation"][name][target],
                level)
            margin_table[name][key]["target"] = target
    uncertainty = {
        "model_version": config.model_version, "deployed_model": deployed,
        "m_U": margin_table[deployed]["m_U"]["value"],
        "m_R": margin_table[deployed]["m_R"]["value"],
        "units": {"m_U": "utilization (pps / c_safe_pps)", "m_R": "ms"},
        "split": "validation", "level": level, "by_model": margin_table,
        "rule": "fixed on validation before any coverage is computed; not re-tuned on test",
    }

    # 2. coverage on test (validation shown for reference) ------------------------------
    groups = {g: (lambda r, g=g: r[g]) for g in config.coverage_groups}
    groups["saturated"] = lambda r: bool(r["saturated_before"] or r["saturated_after"])
    groups["overload_after"] = lambda r: bool(r["overload_after"])
    coverage_out: dict[str, Any] = {"level": level, "by_model": {}}
    for name in names:
        entry: dict[str, Any] = {}
        for key, target in config.margins.items():
            m = margin_table[name][key]["value"]
            entry[key] = {
                "margin": m, "target": target,
                "validation": margins.coverage(
                    [float(r[target]) for r in split["validation"]],
                    pred["validation"][name][target], m),
                "test": margins.coverage_by_group(split["test"], target,
                                                  pred["test"][name][target], m, groups),
            }
        m_u = margin_table[name]["m_U"]["value"]
        entry["gate_u_dst"] = {
            s: margins.gate_confusion([float(r["u_dst_post"]) for r in split[s]],
                                      pred[s][name]["u_dst_post"], m_u,
                                      data_config.overload_threshold)
            for s in ("validation", "test")}
        coverage_out["by_model"][name] = entry
    args.final_dir.mkdir(parents=True, exist_ok=True)
    art.write_json(args.final_dir / "uncertainty_margin.json", uncertainty)
    art.write_json(args.final_dir / "coverage_metrics.json", coverage_out)

    # 3. artifact ---------------------------------------------------------------------
    if not deployed.startswith("hybrid_"):
        print(f"\ndeployed model is {deployed}, but the runtime predictor (predictor.py) "
              "only serves a Hybrid artifact: margins and coverage are written, the artifact "
              "is NOT; agree with Vy on the predictor before exporting")
        _report(names, margin_table, coverage_out, deployed)
        return 1
    family = family_of(deployed)
    short = trainer.SHORT_NAMES[family]
    direct_family = family_of(frozen.best_direct)
    dshort = trainer.SHORT_NAMES[direct_family]
    tmp = artifact_dir.with_name(artifact_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "residual_models").mkdir(parents=True)
    (tmp / "direct_model").mkdir()
    (tmp / "hybrid_model").mkdir()
    for t in BASE_TARGETS:
        suffix = trainer.FAMILIES[family].SUFFIX
        shutil.copy2(args.hybrid_dir / "models" / f"residual_{short}_{t}{suffix}",
                     tmp / "residual_models" / f"residual_{short}_{t}{suffix}")
        dsuffix = trainer.FAMILIES[direct_family].SUFFIX
        shutil.copy2(args.direct_dir / "models" / f"direct_{dshort}_{t}{dsuffix}",
                     tmp / "direct_model" / f"direct_{dshort}_{t}{dsuffix}")

    dmanifest = json.loads((args.dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((args.dataset / "feature_schema.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((args.dataset / "split_manifest.json").read_text(
        encoding="utf-8"))
    params = frozen.frozen["parameters"]
    lock_summary = dict(frozen.lock.get("test_summary") or {})
    val_rows = art.read_csv(args.final_dir / "validation_metrics.csv")
    test_rows = art.read_csv(args.final_dir / "test_metrics.csv")
    test_metrics = art.metrics_by_model(test_rows)

    art.write_json(tmp / "analytical_config.json", {
        "model": "analytical-load-conservation",
        "u_src_post": "max(0, lambda_src - lambda_s) / C_src",
        "u_dst_post": "(lambda_dst + lambda_s) / C_dst (not clamped)",
        "response": "persistence: mean response_p95_ms of the pre window",
        "inputs": "processed Packet-In rates averaged over the 5-snapshot pre window; "
                  "lambda_s = moved switch's rate reported by the source",
        "capacity_version": dmanifest["capacity_version"], "c_safe_pps": dmanifest["c_safe_pps"],
        "code": "src/twin/outcome/analytical.py"})
    art.write_json(tmp / "direct_model" / "direct_model.json", {
        "model": frozen.best_direct, "family": direct_family,
        "features": models.model_features, "parameters": params[frozen.best_direct],
        "role": "comparison only; the runtime serves the Hybrid model"})
    art.write_json(tmp / "hybrid_model" / "hybrid_model.json", {
        "model": deployed, "family": family,
        "prediction": "max(0, analytical + residual) per base target; r95_post = max of the "
                      "two responses; i_post = |u_src_post - u_dst_post| / 2",
        "residual_models": "../residual_models", "features": models.hybrid_features,
        "code": "src/twin/outcome/hybrid.py"})
    art.write_json(tmp / "feature_schema.json", {
        **schema, "hybrid_features": models.hybrid_features,
        "analytical_features": list(residual.ANALYTICAL_FEATURES)})
    art.write_json(tmp / "target_schema.json", target_schema())
    art.write_json(tmp / "hyperparameters.json", {
        "deployed": {"model": deployed, "parameters": params[deployed]},
        "best_direct": {"model": frozen.best_direct, "parameters": params[frozen.best_direct]},
        "grids": yaml.safe_load(args.models_config.read_text(encoding="utf-8")),
        "selection": frozen.frozen["rule"]})
    art.write_json(tmp / "validation_metrics.json", {
        "split": "validation", "n": len(split["validation"]),
        "selection_score_S": {n: frozen.validation_s(n)
                              for n in frozen.frozen["validation_scores"]},
        "metrics": art.metrics_by_model(val_rows)})
    art.write_json(tmp / "test_metrics.json", {
        "split": "test", "n": len(split["test"]), "evaluated_once_at": frozen.lock.get(
            "completed_at"), "selection_score_S": lock_summary.get("test_S"),
        "metrics": test_metrics})
    art.write_json(tmp / "per_workload_metrics.json", {
        "rows": art.read_csv(args.final_dir / "per_workload_metrics.csv")})
    art.write_json(tmp / "uncertainty_margin.json", uncertainty)
    art.write_json(tmp / "model_card.json", {
        "model_name": predictor.MODEL_NAME, "model_version": config.model_version,
        "purpose": "predict the controller state after one switch migration (what-if)",
        "supported_action": "single switch ownership migration between two controllers",
        "hybrid_family": family, "deployed_model": deployed,
        "outputs": ["u_src_post", "u_dst_post", "r_src95_post", "r_dst95_post", "r95_post",
                    "i_post", "m_U", "m_R"],
        "selection": {"rule": "Best Direct / Best Hybrid by lowest validation S; deployed = "
                              "lowest validation S among Analytical, Best Direct, Best Hybrid",
                      "validation_S": {n: frozen.validation_s(n) for n in names},
                      "test_S": lock_summary.get("test_S")},
        "test_rmse": lock_summary.get("test_rmse"),
        "margins": {"m_U": uncertainty["m_U"], "m_R": uncertainty["m_R"], "level": level},
        "limitations": [
            "trained on the 2-controller topology (2C20S) only",
            "requires a valid TwinState and 5 consecutive valid snapshots",
            "does not authorize migration: no safe field, the Safety Gate decides (Week 13)",
            "trained on COMMITTED migrations only; rollback and NO_OP are not regression "
            "targets",
            "response_p95_ms is controller processing time and does not rise with overload "
            "on this testbed",
            "U is a lower bound of the offered load when a controller saturates "
            "(14 saturated attempts kept in the data)",
            "few overload examples (10 in total, 1 in test): coverage near U = 1 is weakly "
            "supported",
            f"valid only for capacity {dmanifest['capacity_version']}",
        ]})
    art.write_manifest_files(tmp, {
        "model_name": predictor.MODEL_NAME, "model_version": config.model_version,
        "source_dataset": dmanifest["source_dataset"],
        "processed_dataset": dmanifest["dataset_version"],
        "processed_dataset_files": dmanifest["files"],
        "feature_schema": predictor.FEATURE_SCHEMA, "target_schema": predictor.TARGET_SCHEMA,
        "capacity_version": dmanifest["capacity_version"],
        "c_safe_pps": dmanifest["c_safe_pps"],
        "train_runs": split_manifest["splits"]["train"],
        "validation_runs": split_manifest["splits"]["validation"],
        "test_runs": split_manifest["splits"]["test"],
        "deployed_model": deployed, "hybrid_family": family, "best_direct": frozen.best_direct,
        "model_selection_sha256": frozen.selection["frozen_sha256"],
        "test_evaluated_at": frozen.lock.get("completed_at"),
        "git": _git(), "created_at": datetime.now(timezone.utc).isoformat(),
        "pending": [art.LATENCY_FILE]})

    # reload check: the copied residual models give the Day 4 predictions
    loaded = hybrid.load(tmp / "residual_models", family)
    again = hybrid.predict(loaded, split["validation"], models.hybrid_features)
    diff = max(abs(a - b) for t in BASE_TARGETS
               for a, b in zip(again[t], pred["validation"][deployed][t]))
    if diff > 1e-9:
        raise ValueError(f"artifact models do not reproduce the saved ones (max diff {diff})")
    tmp.rename(artifact_dir)

    _report(names, margin_table, coverage_out, deployed)
    print(f"\nartifact written to {artifact_dir} (reload check max |diff| {diff:.1e})")
    check(artifact_dir, args.capacity_file)
    print(f"wrote {args.final_dir / 'uncertainty_margin.json'}, "
          f"{args.final_dir / 'coverage_metrics.json'}")
    return 0


def _report(names: list[str], table: dict[str, Any], cov: dict[str, Any],
            deployed: str) -> None:
    print("\nmargins on validation (one-sided, level "
          f"{cov['level']:g}) and coverage on test")
    print(f"{'model':<12}{'m_U':>8}{'cov U test':>12}{'m_R ms':>9}{'cov R test':>12}"
          f"{'missed ovl':>12}{'false alarm':>13}")
    for name in names:
        e = cov["by_model"][name]
        u = e["m_U"]["test"][0]
        r = e["m_R"]["test"][0]
        g = e["gate_u_dst"]["test"]
        mark = " *" if name == deployed else ""
        print(f"{name + mark:<12}{table[name]['m_U']['value']:>8.4f}"
              f"{u['covered']:>5}/{u['n']:<3} {u['coverage']:.2f}"
              f"{table[name]['m_R']['value']:>9.4f}{r['covered']:>5}/{r['n']:<3} "
              f"{r['coverage']:.2f}{g['missed_overload']:>6}/{g['overload']:<5}"
              f"{g['false_alarm']:>7}/{g['false_alarm'] + g['true_negative']:<5}")
    e = cov["by_model"][deployed]["m_U"]["test"]
    print(f"\n{deployed}: coverage of m_U on test by group")
    for row in e:
        if row["group"] != "all":
            print(f"  {row['group']:<24}{row['value']:<14}{row['covered']:>3}/{row['n']:<3}"
                  f" {row['coverage']:.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
