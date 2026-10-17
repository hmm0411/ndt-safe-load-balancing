#!/usr/bin/env python3
"""Week 10 evidence package (tuần 10, mục 29-34 của kế hoạch).

Builds evidence/week10/ from what the earlier steps already produced. Nothing
is retrained and nothing is invented: a missing input is listed as MISSING in
forecasting_dod.txt.

Run on the VM from the repo root (after the testbed runs):

    python scripts/analysis/week10_evidence.py
    python scripts/analysis/week10_evidence.py --ci      # also runs make ci -> ci.txt

Outputs (evidence/week10/):
  dataset_statistics.csv            10 variables: count, mean, std, min, p25, median, p75, p95, max, missing %
  dataset_statistics_by_workload.csv
  dataset_values.csv                one row per valid raw sample (for your own plots)
  hist_<variable>.csv               30-bin histograms of the 10 variables (Figure 1)
  dataset_correlation.csv           Pearson matrix of the 10 variables
  dataset_funnel.csv                9000 nominal -> raw -> valid -> processed
  dataset_split_distribution.csv    workload x split counts
  workload_profiles.csv             U(t) of one run per workload (Figure 2)
  quality_score_config.json         configs/quality.yaml + the 4 weight variants
  quality_sensitivity.csv           per variant: score spread, rank agreement, Hard Gate counts
  quality_sensitivity_by_run.csv
  quality_sensitivity_snapshots.csv one row per replayed snapshot, one column per variant
  rf_validation_search.csv, xgb_validation_search.csv
  model_comparison.csv              model x split: MAE, RMSE, MAPE, R2, P95|e|, latency, S_m
  per_workload_metrics.csv, skill_vs_persistence.csv, feature_importance.csv
  validation_metrics.json, test_metrics.json, unseen_metrics.json
  coverage.csv, overload_detection.csv, ood_degradation.csv, error_margin.json
  inference_latency.json, inference_latency.csv, horizon_v1.json, horizon_v1.csv
  <scenario>_runtime.json           stable, gradual, burst, invalid_snapshot, hot_switch, migration
  runtime_<scenario>_<controller>.csv  per-snapshot series (Figure 9 and the other runs)
  runtime_summary.csv, runtime_reasons.csv, runtime_ood_features.csv
  ci.txt (with --ci), forecasting_dod.txt
  latex/                            booktabs tables + pgfplots figures 1-9 reading the CSVs,
                                    main.tex compiles them all (xelatex main.tex)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from scripts.analysis.forecast_runtime_report import active_window, load_rows, realized, summarize
from src.twin.forecasting.splitter import assign_split
from src.twin.forecasting.versions import MODEL_VERSION, MODELS_ROOT, PROCESSED_ROOT, dataset_version
from src.twin.state.quality_score import (
    COMPONENTS,
    load_quality_config,
    quality_components,
    quality_score,
)

OUT = Path("evidence/week10")
RESULTS = Path("results/week10")
ARTIFACT = MODELS_ROOT / "forecast-v1"
FIVE_MODELS = ("persistence", "moving_average_w3", "linear_regression", "random_forest", "xgboost")
SPLITS = ("validation", "test", "unseen")
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot-switch")

# Kế hoạch Tuần 10, mục 30.
DATASET_VARIABLES = (
    "utilization",
    "processed_packet_in_rate",
    "flow_mod_rate",
    "process_cpu_percent",
    "process_memory_rss_mb",
    "response_p95_ms",
    "max_switch_control_load_share",
    "age_of_twin_ms",
    "completeness_ratio",
    "synchronization_jitter_ms",
)

# Kế hoạch Tuần 10, mục 31. Order = COMPONENTS.
QUALITY_VARIANTS: dict[str, dict[str, float]] = {
    "base": dict(zip(COMPONENTS, (0.30, 0.25, 0.20, 0.10, 0.10, 0.05))),
    "consistency_heavy": dict(zip(COMPONENTS, (0.35, 0.20, 0.20, 0.10, 0.10, 0.05))),
    "completeness_heavy": dict(zip(COMPONENTS, (0.25, 0.30, 0.20, 0.10, 0.10, 0.05))),
    "freshness_heavy": dict(zip(COMPONENTS, (0.25, 0.25, 0.25, 0.10, 0.10, 0.05))),
}

# scenario -> RUN_ID of the testbed run (w10-testbed.sh). Override with --scenario.
DEFAULT_SCENARIOS = {
    "stable": "w10-r1-stable",
    "gradual": "w10-r3-gradual",
    "burst": "w10-r2b-burst",
    "invalid_snapshot": "w10-r4-telemetry-loss",
    "hot_switch": "w10-r6-hot-switch",
    "migration": "w10-r7-migration",
}


# --------------------------------------------------------------------------- io


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    cols = list(columns or _columns(rows))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _cell(row.get(c)) for c in cols})


def _columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    cols: list[str] = []
    for row in rows:
        cols.extend(k for k in row if k not in cols)
    return cols


def _cell(value: Any) -> Any:
    if value is None:
        return "nan"
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


class Evidence:
    """Collects what was written and what was missing (forecasting_dod.txt)."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.written: list[str] = []
        self.missing: list[str] = []

    def path(self, name: str) -> Path:
        self.written.append(name)
        return self.out / name

    def copy(self, src: Path, name: str) -> bool:
        if not src.is_file():
            self.missing.append(f"{name} (needs {src})")
            return False
        shutil.copyfile(src, self.path(name))
        return True

    def need(self, src: Path, what: str) -> bool:
        if src.is_file() or src.is_dir():
            return True
        self.missing.append(f"{what} (needs {src})")
        return False


# --------------------------------------------------------------------- dataset


def describe(series: pd.Series) -> dict[str, Any]:
    total = int(series.size)
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    q = valid.quantile([0.25, 0.5, 0.75, 0.95]) if len(valid) else pd.Series(dtype=float)
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()) if len(valid) else None,
        "std": float(valid.std(ddof=0)) if len(valid) else None,
        "min": float(valid.min()) if len(valid) else None,
        "p25": float(q.get(0.25, math.nan)) if len(valid) else None,
        "median": float(q.get(0.5, math.nan)) if len(valid) else None,
        "p75": float(q.get(0.75, math.nan)) if len(valid) else None,
        "p95": float(q.get(0.95, math.nan)) if len(valid) else None,
        "max": float(valid.max()) if len(valid) else None,
        "missing_pct": 100.0 * (total - int(valid.size)) / total if total else None,
    }


def _histogram(values: list[float], bins: int) -> tuple[list[int], list[float]]:
    lo, hi = min(values), max(values)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0] * bins
    for x in values:
        counts[min(int((x - lo) / width), bins - 1)] += 1
    return counts, edges


def load_raw_samples(raw_root: Path, run_ids: Iterable[str]) -> pd.DataFrame:
    frames = []
    for run_id in run_ids:
        path = raw_root / run_id / "forecast_samples.jsonl"
        if path.is_file():
            frames.append(pd.read_json(path, lines=True))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def dataset_evidence(ev: Evidence, raw_root: Path, nominal: int) -> None:
    data_dir = PROCESSED_ROOT / dataset_version()
    dataset_csv = data_dir / "forecast_dataset.csv"
    if not ev.need(dataset_csv, "dataset statistics"):
        return
    df = pd.read_csv(dataset_csv)
    df["split"] = df["run_id"].map(assign_split)
    raw = load_raw_samples(raw_root, sorted(df["run_id"].unique()))
    if raw.empty:
        ev.missing.append(f"raw samples for dataset statistics (needs {raw_root}/<run_id>/forecast_samples.jsonl)")
        return
    raw["split"] = raw["run_id"].map(assign_split)
    valid = raw[raw["snapshot_valid"].astype(bool)].copy()

    rows = [{"variable": v, "source": "raw valid samples", **describe(valid[v])}
            for v in DATASET_VARIABLES if v in valid]
    write_csv(ev.path("dataset_statistics.csv"), rows)
    by_wl = []
    for workload in [w for w in WORKLOADS if w in set(valid["workload_type"])]:
        part = valid[valid["workload_type"] == workload]
        by_wl += [{"workload": workload, "variable": v, **describe(part[v])}
                  for v in DATASET_VARIABLES if v in part]
    write_csv(ev.path("dataset_statistics_by_workload.csv"), by_wl)

    cols = ["run_id", "controller_id", "workload_type", "split", *[v for v in DATASET_VARIABLES if v in valid]]
    valid[cols].to_csv(ev.path("dataset_values.csv"), index=False, na_rep="nan")
    # Pre-binned histograms (30 bins): pgfplots plots them instantly.
    for v in DATASET_VARIABLES:
        if v not in valid:
            continue
        values = pd.to_numeric(valid[v], errors="coerce").dropna()
        if values.empty:
            continue
        counts, edges = _histogram(values.to_list(), 30)
        write_csv(ev.path(f"hist_{v}.csv"),
                  [{"left": edges[i], "right": edges[i + 1], "count": counts[i]} for i in range(len(counts))]
                  + [{"left": edges[-1], "right": edges[-1], "count": 0}])
    corr = valid[[v for v in DATASET_VARIABLES if v in valid]].corr(method="pearson")
    corr.index.name = "variable"
    corr.to_csv(ev.path("dataset_correlation.csv"), na_rep="nan")

    report_path = data_dir / "dataset_quality_report.json"
    report = read_json(report_path) if report_path.is_file() else {}
    funnel = [
        {"stage": "nominal (runs x duration x controllers)", "samples": nominal},
        {"stage": "raw samples written", "samples": int(len(raw))},
        {"stage": "valid snapshots", "samples": int(len(valid))},
    ]
    for key, label in (("dropped_for_history", "dropped: not enough history"),
                       ("dropped_for_future_label", "dropped: no label at t+h")):
        if report.get(key) is not None:
            funnel.append({"stage": label, "samples": report[key]})
    funnel.append({"stage": "processed ML samples", "samples": int(len(df))})
    write_csv(ev.path("dataset_funnel.csv"), funnel)

    counts = pd.crosstab(df["workload_type"], df["split"]).reindex(
        columns=["train", "validation", "test", "unseen"], fill_value=0)
    counts.index.name = "workload"
    counts.to_csv(ev.path("dataset_split_distribution.csv"))

    profiles = []
    for workload in WORKLOADS:
        runs = sorted(raw.loc[raw["workload_type"] == workload, "run_id"].unique())
        if not runs:
            continue
        part = raw[(raw["run_id"] == runs[0])].copy()
        part["t"] = pd.to_datetime(part["observed_at"], utc=True)
        t0 = part["t"].min()
        for _, r in part.sort_values("t").iterrows():
            profiles.append({"workload": workload, "run_id": runs[0], "controller_id": r["controller_id"],
                             "t_s": (r["t"] - t0).total_seconds(), "utilization": r["utilization"]})
    write_csv(ev.path("workload_profiles.csv"), profiles)
    # one wide file per workload for pgfplots
    for workload in WORKLOADS:
        part = [p for p in profiles if p["workload"] == workload and p["controller_id"] == "c1"]
        if part:
            write_csv(ev.path(f"workload_profile_{workload.replace('-', '_')}.csv"), part,
                      ["t_s", "utilization"])


# --------------------------------------------------------------- quality score


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def replay_quality(snapshots: list[dict[str, Any]], weights: Mapping[str, Mapping[str, float]],
                   thresholds: Mapping[str, Any], telemetry: Mapping[str, Any],
                   expected_controllers: set[str], expected_switches: set[str]) -> list[dict[str, Any]]:
    """Recompute TwinQualityAssessor for recorded snapshots (src/twin/state/quality.py).

    The twinning rate is replayed with the same window as the orchestrator
    (configs/telemetry.yaml). The Hard Gate does not read the weights, so
    ``hard_valid`` is the same for every variant by construction; the output
    shows it.
    """
    tq = telemetry.get("twin_quality", {})
    window = float(tq.get("twinning_window_seconds", 60))
    min_rate = float(tq.get("min_twinning_rate", 0.95))
    min_complete = float(tq.get("min_completeness_ratio", 1.0))
    max_age = float(telemetry.get("freshness_max_age_ms", 2500))
    attempts: deque[tuple[datetime, bool]] = deque()
    rows = []
    for snap in sorted(snapshots, key=lambda s: s["created_at"]):
        created = _ts(snap["created_at"])
        quality = snap.get("quality") or {}
        attempts.append((created, bool(quality.get("valid"))))
        while attempts and attempts[0][0] < created - timedelta(seconds=window):
            attempts.popleft()
        rate = sum(ok for _, ok in attempts) / len(attempts)
        ctrls = snap.get("controllers") or []
        observed = [_ts(c["observed_at"]) for c in ctrls]
        age = max((max(0.0, (created - t).total_seconds() * 1000) for t in observed), default=math.inf)
        jitter = (max(observed) - min(observed)).total_seconds() * 1000 if len(observed) > 1 else 0.0
        collection = max((max(0.0, (_ts(c["ingested_at"]) - _ts(c["observed_at"])).total_seconds() * 1000)
                          for c in ctrls), default=math.inf)
        present = (len({c["controller_id"] for c in ctrls} & expected_controllers)
                   + len({s["switch_id"] for s in snap.get("switches") or []} & expected_switches)
                   + len({o["switch_id"] for o in snap.get("ownership") or []} & expected_switches))
        expected_total = len(expected_controllers) + 2 * len(expected_switches)
        completeness = present / expected_total if expected_total else 0.0
        hard_valid = bool(quality.get("valid")) and age <= max_age and completeness >= min_complete \
            and rate >= min_rate
        comps = quality_components(
            consistency_score=float(quality.get("consistency_score", 1.0)),
            completeness_ratio=completeness, age_ms=age, jitter_ms=jitter,
            collection_ms=collection, twinning_rate=rate, thresholds=thresholds)
        row: dict[str, Any] = {"snapshot_id": snap.get("snapshot_id"), "created_at": snap["created_at"],
                               "hard_valid": hard_valid, **{f"c_{k}": v for k, v in comps.items()}}
        for name, w in weights.items():
            row[name] = quality_score(comps, w)
        rows.append(row)
    return rows


def quality_evidence(ev: Evidence, snapshots_dir: Path, runtime_dir: Path,
                     scenarios: Mapping[str, str]) -> None:
    config = load_quality_config(Path("configs/quality.yaml"))
    write_json(ev.path("quality_score_config.json"),
               {"config": config, "variants": QUALITY_VARIANTS,
                "note": "Hard Gate (valid) does not depend on the weights."})
    telemetry = yaml.safe_load(Path("configs/telemetry.yaml").read_text(encoding="utf-8"))
    topo = yaml.safe_load(Path("configs/controllers_2c20s.yaml").read_text(encoding="utf-8"))
    exp_c, exp_s = set(topo["controllers"]), set(topo["switches"])
    all_rows: list[dict[str, Any]] = []
    for scenario, run_id in scenarios.items():
        path = snapshots_dir / f"{run_id}.jsonl"
        if not path.is_file():
            continue
        snaps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = replay_quality(snaps, QUALITY_VARIANTS, config["thresholds"], telemetry, exp_c, exp_s)
        # The snapshot file keeps every start of that RUN_ID (reruns, idle tails):
        # keep the workload window of the forecast log, replayed with its history.
        forecast = runtime_dir / f"{run_id}.jsonl"
        if forecast.is_file():
            window = active_window(load_rows(forecast))
            if window:
                lo = min(r["created_at"] for r in window)
                hi = max(r["created_at"] for r in window)
                rows = [r for r in rows if lo <= _ts(r["created_at"]) <= hi]
        t0 = _ts(rows[0]["created_at"]) if rows else None
        for r in rows:
            r["scenario"], r["run_id"] = scenario, run_id
            r["t_s"] = (_ts(r["created_at"]) - t0).total_seconds() if t0 else None
        all_rows += rows
        for r in rows:  # pgfplots marker for Hard Gate = invalid
            r["invalid_mark"] = None if r["hard_valid"] else 0.02
        write_csv(ev.path(f"quality_sensitivity_{scenario}.csv"), rows,
                  ["t_s", "hard_valid", "invalid_mark", *QUALITY_VARIANTS])
    if not all_rows:
        ev.missing.append(f"quality sensitivity (needs {snapshots_dir}/<RUN_ID>.jsonl of the W10 runs)")
        return
    write_csv(ev.path("quality_sensitivity_snapshots.csv"), all_rows,
              ["scenario", "run_id", "snapshot_id", "t_s", "hard_valid",
               *[f"c_{c}" for c in COMPONENTS], *QUALITY_VARIANTS])
    df = pd.DataFrame(all_rows)
    base = df["base"]
    summary = []
    for name in QUALITY_VARIANTS:
        s = df[name]
        top = set(df.nlargest(max(1, len(df) // 10), "base").index)
        top_v = set(df.nlargest(max(1, len(df) // 10), name).index)
        summary.append({
            "variant": name,
            **{f"w_{k}": v for k, v in QUALITY_VARIANTS[name].items()},
            "snapshots": int(len(s)),
            "score_mean": float(s.mean()), "score_std": float(s.std(ddof=0)),
            "score_min": float(s.min()), "score_p5": float(s.quantile(0.05)), "score_max": float(s.max()),
            "score_mean_valid": float(s[df["hard_valid"]].mean()) if df["hard_valid"].any() else None,
            "score_mean_invalid": float(s[~df["hard_valid"]].mean()) if (~df["hard_valid"]).any() else None,
            "mean_abs_diff_vs_base": float((s - base).abs().mean()),
            "max_abs_diff_vs_base": float((s - base).abs().max()),
            "spearman_vs_base": float(s.corr(base, method="spearman")),
            "kendall_vs_base": float(s.corr(base, method="kendall")),
            "top10pct_overlap_vs_base": len(top & top_v) / len(top),
            "hard_valid": int(df["hard_valid"].sum()),
            "hard_invalid": int((~df["hard_valid"]).sum()),
            # A forecast is produced only for a valid TwinState: eligibility follows the Hard Gate.
            "runtime_eligible": int(df["hard_valid"].sum()),
        })
    write_csv(ev.path("quality_sensitivity.csv"), summary)
    by_run = []
    for (scenario, run_id), part in df.groupby(["scenario", "run_id"]):
        for name in QUALITY_VARIANTS:
            by_run.append({"scenario": scenario, "run_id": run_id, "variant": name,
                           "snapshots": int(len(part)), "hard_valid": int(part["hard_valid"].sum()),
                           "score_mean": float(part[name].mean()), "score_min": float(part[name].min())})
    write_csv(ev.path("quality_sensitivity_by_run.csv"), by_run)


# ----------------------------------------------------------------------- models


def model_evidence(ev: Evidence, training_dir: Path) -> None:
    for split in SPLITS:
        ev.copy(training_dir / f"{split}_metrics.json", f"{split}_metrics.json")
    selection = read_json(RESULTS / "model_selection.json") if (RESULTS / "model_selection.json").is_file() else {}
    sel = {m["model"]: m for m in selection.get("models", [])}
    rows = []
    for split in SPLITS:
        path = training_dir / f"{split}_metrics.json"
        if not path.is_file():
            continue
        for model, metrics in read_json(path).items():
            rows.append({"model": model, "split": split, "in_five_models": model in FIVE_MODELS,
                         **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
                         "selection_score": sel.get(model, {}).get("selection_score") if split == "validation" else None,
                         "selected": model == selection.get("selected")})
    if rows:
        write_csv(ev.path("model_comparison.csv"), rows)
        for split in SPLITS:
            part = [r for r in rows if r["split"] == split and r["in_five_models"]]
            part.sort(key=lambda r: FIVE_MODELS.index(r["model"]))
            if part:
                write_csv(ev.path(f"model_comparison_{split}.csv"), part)
    if selection:
        write_csv(ev.path("model_selection.csv"),
                  [{**m, "rank": i + 1} for i, m in enumerate(
                      sorted(selection.get("models", []), key=lambda m: m.get("selection_score", math.inf)))])

    search_path = training_dir / "hyperparameter_search.json"
    if ev.need(search_path, "RF/XGB validation search"):
        search = read_json(search_path)
        for model, name in (("random_forest", "rf_validation_search.csv"), ("xgboost", "xgb_validation_search.csv")):
            block = search.get(model, {})
            selected = block.get("selected", {})
            cands = []
            for i, c in enumerate(block.get("candidates", [])):
                params = c.get("params", {})
                cands.append({"candidate": i, **{f"param_{k}": v for k, v in params.items()},
                              **{k: v for k, v in c.items() if k != "params"},
                              "selected": params == selected,
                              "selection_split": search.get("selection_split")})
            write_csv(ev.path(name), cands)

    pw = training_dir / "per_workload_metrics.json"
    if ev.need(pw, "per-workload metrics"):
        rows = []
        for model, splits in read_json(pw).items():
            for split, workloads in splits.items():
                for workload, metrics in workloads.items():
                    rows.append({"model": model, "split": split, "workload": workload, **metrics})
        write_csv(ev.path("per_workload_metrics.csv"), rows)
        wide = []
        for workload in WORKLOADS:
            row: dict[str, Any] = {"workload": workload}
            for r in rows:
                if r["workload"] == workload and r["model"] in FIVE_MODELS:
                    row[r["model"]] = r["rmse"]
            if len(row) > 1:
                wide.append(row)
        write_csv(ev.path("per_workload_rmse_wide.csv"), wide, ["workload", *FIVE_MODELS])

    sk = training_dir / "skill_vs_persistence.json"
    if sk.is_file():
        write_csv(ev.path("skill_vs_persistence.csv"),
                  [{"split": s, "workload": w, "model": m, "rmse_ratio_vs_persistence": v}
                   for s, wl in read_json(sk).items() for w, ms in wl.items() for m, v in ms.items()])
    fi = training_dir / "feature_importance.json"
    if fi.is_file():
        write_csv(ev.path("feature_importance.csv"),
                  [{"model": m, "feature": f, "importance": v}
                   for m, feats in read_json(fi).items() for f, v in feats.items()])


def calibration_evidence(ev: Evidence) -> None:
    ev.copy(RESULTS / "error_margin.json", "error_margin.json")
    cov_path = RESULTS / "coverage.json"
    if ev.need(cov_path, "Coverage95"):
        cov = read_json(cov_path)
        rows, over = [], []
        for split, block in cov.get("splits", {}).items():
            for gate in ("with_gate", "without_gate"):
                g = block.get(gate, {})
                rows.append({"split": split, "gate": gate, "group": "all", "name": "all",
                             "coverage": g.get("coverage"), "samples": block.get("samples")})
                for group in ("by_controller", "by_workload"):
                    for name, v in g.get(group, {}).items():
                        rows.append({"split": split, "gate": gate, "group": group, "name": name, **v})
                if g.get("overload"):
                    over.append({"split": split, "gate": gate, **g["overload"]})
        write_csv(ev.path("coverage.csv"), rows)
        write_csv(ev.path("overload_detection.csv"), over)
    ood_path = RESULTS / "ood_degradation.json"
    if ev.need(ood_path, "OOD degradation"):
        ood = read_json(ood_path)
        rows = [{"split": s, **{k: v for k, v in b.items() if not isinstance(v, dict)},
                 "ood_to_id_rmse_ratio": ood.get("ood_to_id_rmse_ratio")}
                for s, b in ood.get("splits", {}).items()]
        write_csv(ev.path("ood_degradation.csv"), rows)

    lat = Path("data/benchmarks/forecast_inference_latency.json")
    if ev.copy(lat, "inference_latency.json"):
        data = read_json(lat)
        env = data.get("environment", {})
        write_csv(ev.path("inference_latency.csv"),
                  [{"model": m, **v, "measured_at": env.get("measured_at"),
                    "load_average_before": env.get("load_average_before"),
                    "load_average_after": env.get("load_average_after")}
                   for m, v in data.get("models", {}).items()])
    horizon = Path("data/benchmarks/horizon_v1.json")
    if ev.copy(horizon, "horizon_v1.json"):
        write_csv(ev.path("horizon_v1.csv"), [read_json(horizon)])


# ---------------------------------------------------------------------- runtime


def runtime_evidence(ev: Evidence, runtime_dir: Path, scenarios: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for scenario, run_id in scenarios.items():
        path = runtime_dir / f"{run_id}.jsonl"
        stored = RESULTS / "runtime" / f"{run_id}_summary.json"
        if not path.is_file():
            if stored.is_file():
                summary = read_json(stored)
                summary.update({"scenario": scenario, "source": str(stored),
                                "note": "raw forecast JSONL not found: summary only, no series"})
                write_json(ev.path(f"{scenario}_runtime.json"), summary)
                summaries[scenario] = summary
            else:
                ev.missing.append(f"{scenario} runtime (needs {path}; run it with w10-testbed.sh)")
            continue
        rows = active_window(load_rows(path))
        summary = summarize(run_id, load_rows(path))
        summary.update({"scenario": scenario, "source": str(path)})
        write_json(ev.path(f"{scenario}_runtime.json"), summary)
        summaries[scenario] = summary
        actual = {(id(r)): u for r, u in realized(rows)}
        t0 = min(r["created_at"] for r in rows)
        for cid in sorted({r["controller_id"] for r in rows}):
            series = []
            for r in sorted((r for r in rows if r["controller_id"] == cid), key=lambda r: r["created_at"]):
                t = (r["created_at"] - t0).total_seconds()
                h = float(r.get("horizon_seconds") or 0.0)
                ready = r.get("status") == "READY"
                u, uhat, ub = r.get("current_utilization"), r.get("predicted_utilization"), r.get("upper_utilization_bound")
                series.append({
                    "t_s": t, "t_target_s": t + h if ready else None, "status": r.get("status"),
                    "reason": r.get("reason") or "", "u": u, "u_hat": uhat, "upper": ub,
                    "margin": r.get("error_margin_utilization"),
                    "u_at_target": actual.get(id(r)),
                    "risk": int(bool(r.get("overload_risk"))) if ready else None,
                    "alert_upper": ub if ready and r.get("overload_risk") else None,
                    "ood": int(bool(r.get("ood_fallback"))) if ready else None,
                    "ood_u_hat": uhat if ready and r.get("ood_fallback") else None,
                    "measured_overload_u": u if u is not None and float(u) >= 1.0 else None,
                    "ood_features": r.get("ood_features") or [],
                    "inference_ms": r.get("inference_time_ms"), "service_ms": r.get("service_time_ms"),
                    "quality_score": r.get("quality_score"),
                })
            write_csv(ev.path(f"runtime_{scenario}_{cid}.csv"), series)
    if summaries:
        flat = [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries.values()]
        write_csv(ev.path("runtime_summary.csv"), flat)
        write_csv(ev.path("runtime_reasons.csv"),
                  [{"scenario": sc, "reason": r, "count": n}
                   for sc, s in summaries.items() for r, n in (s.get("not_ready_reasons") or {}).items()])
        write_csv(ev.path("runtime_ood_features.csv"),
                  [{"scenario": sc, "feature": f, "count": n}
                   for sc, s in summaries.items() for f, n in (s.get("ood_features") or {}).items()])
    nm = RESULTS / "runtime" / "w10-r5-no-model.json"
    if nm.is_file():
        write_json(ev.path("missing_model_runtime.json"), {"scenario": "missing_model", **read_json(nm)})
    return summaries


# --------------------------------------------------------------------- CI + DoD


def run_ci(ev: Evidence) -> None:
    lines = [f"# make ci + git diff --check, {datetime.now(timezone.utc).isoformat()}"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    lines.append(f"git_commit: {commit.stdout.strip()}")
    for cmd in (["make", "ci"], ["git", "diff", "--check"]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines += [f"\n$ {' '.join(cmd)}", proc.stdout[-20000:], proc.stderr[-20000:], f"exit_code: {proc.returncode}"]
    ev.path("ci.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def dod(ev: Evidence, training_dir: Path, runtime: Mapping[str, dict[str, Any]]) -> None:
    out = ev.out

    def ok(name: str) -> bool:
        return (out / name).is_file()

    search = read_json(training_dir / "hyperparameter_search.json") if (training_dir / "hyperparameter_search.json").is_file() else {}
    test_metrics = read_json(training_dir / "test_metrics.json") if (training_dir / "test_metrics.json").is_file() else {}
    xgb_test = test_metrics.get("xgboost", {})
    sens = list(csv.DictReader((out / "quality_sensitivity.csv").open())) if ok("quality_sensitivity.csv") else []
    ci_text = (out / "ci.txt").read_text(encoding="utf-8") if ok("ci.txt") else ""
    gradual = runtime.get("gradual", {})
    invalid = runtime.get("invalid_snapshot", {})
    horizon = read_json(Path("data/benchmarks/horizon_v1.json")) if Path("data/benchmarks/horizon_v1.json").is_file() else {}
    coverage = read_json(RESULTS / "coverage.json") if (RESULTS / "coverage.json").is_file() else {}
    test_cov = coverage.get("splits", {}).get("test", {}).get("with_gate", {}).get("coverage")
    ood = read_json(RESULTS / "ood_degradation.json") if (RESULTS / "ood_degradation.json").is_file() else {}
    app = Path("src/orchestrator/app.py").read_text(encoding="utf-8") if Path("src/orchestrator/app.py").is_file() else ""
    items: list[tuple[str, bool | None, str]] = [
        ("Week9 dataset frozen", _exists(PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"), str(PROCESSED_ROOT / dataset_version())),
        ("RF", "random_forest" in test_metrics, "test_metrics.json"),
        ("XGBoost", "xgboost" in test_metrics, "test_metrics.json"),
        ("RF tuning Validation-only", bool(search.get("random_forest")) and search.get("selection_split") == "validation", "rf_validation_search.csv"),
        ("XGB tuning Validation-only", bool(search.get("xgboost")) and search.get("selection_split") == "validation", "xgb_validation_search.csv"),
        ("Five-model comparison", all(m in test_metrics for m in FIVE_MODELS), "model_comparison.csv"),
        ("MAE", "mae" in xgb_test, "model_comparison.csv"),
        ("RMSE", "rmse" in xgb_test, "model_comparison.csv"),
        ("MAPEe", "mape" in xgb_test, "model_comparison.csv"),
        ("R2", "r2" in xgb_test, "model_comparison.csv"),
        ("P95 absolute error", "p95_absolute_error" in xgb_test, "model_comparison.csv"),
        ("inference mean/P95", ok("inference_latency.csv"), "inference_latency.csv"),
        ("ID evaluation", ok("test_metrics.json"), "test_metrics.json"),
        ("Unseen Hot-switch", ok("unseen_metrics.json"), "unseen_metrics.json"),
        ("OOD degradation ratio", ood.get("ood_to_id_rmse_ratio") is not None, "ood_degradation.csv"),
        ("Weighted Validation Score", ok("model_selection.csv"), "model_selection.csv"),
        ("Final model locked", _exists(ARTIFACT / "model_card.json"), str(ARTIFACT / "model_card.json")),
        ("Validation P95 margin", ok("error_margin.json"), "error_margin.json"),
        ("Test Coverage95", test_cov is not None and test_cov >= 0.95, f"coverage.csv (test, with gate: {test_cov})"),
        ("Model artifact", _exists(ARTIFACT / "feature_schema.json", ARTIFACT / "feature_ranges.json"), str(ARTIFACT)),
        ("Model card", _exists(ARTIFACT / "model_card.json"), str(ARTIFACT / "model_card.json")),
        ("History buffer", _exists(Path("src/twin/forecasting/history.py"), Path("tests/unit/test_forecast_history.py")), "src/twin/forecasting/history.py"),
        ("Runtime Predictor", _exists(Path("src/twin/forecasting/predictor.py"), Path("tests/unit/test_forecast_predictor.py")), "src/twin/forecasting/predictor.py"),
        ("Overload Risk Detector", _exists(Path("tests/unit/test_overload_detector.py")), "tests/unit/test_overload_detector.py"),
        ("consistency_score", _exists(Path("tests/unit/test_snapshot_consistency_score.py")), "tests/unit/test_snapshot_consistency_score.py"),
        ("quality_score", _exists(Path("src/twin/state/quality_score.py"), Path("tests/unit/test_twin_quality_score.py")), "src/twin/state/quality_score.py"),
        ("Hard Gate preserved", (len({r["hard_valid"] for r in sens}) == 1) if sens else None, "quality_sensitivity.csv (hard_valid equal for all variants)"),
        ("Invalid state -> NOT_READY", ((invalid.get("not_ready_reasons") or {}).get("INVALID_SNAPSHOT", 0) > 0) if invalid else None, "invalid_snapshot_runtime.json"),
        ("/api/v1/twin/forecast", '"/api/v1/twin/forecast"' in app, "src/orchestrator/app.py"),
        ("Stable runtime", (runtime["stable"].get("overload_alerts", 1) == 0) if "stable" in runtime else None, "stable_runtime.json"),
        ("Gradual proactive runtime", (gradual.get("overload_hits", 0) > 0 and gradual.get("overload_misses", 1) == 0) if gradual else None, "gradual_runtime.json"),
        ("Burst runtime recorded", True if "burst" in runtime else None, "burst_runtime.json"),
        ("Horizon v1 measured", horizon.get("feasible") is True, "horizon_v1.json"),
        ("Dataset statistics", True if ok("dataset_statistics.csv") else None, "dataset_statistics.csv"),
        ("Quality sensitivity", True if ok("quality_sensitivity.csv") else None, "quality_sensitivity.csv"),
        ("make ci", ("exit_code: 0" in ci_text) if ci_text else None, "ci.txt (run with --ci)"),
    ]
    width = max(len(n) for n, _, _ in items)
    lines = [f"# Week 10 Definition of Done, generated {datetime.now(timezone.utc).isoformat()}",
             "# PASS = evidence found and criterion met; MISSING = evidence not produced yet; FAIL = criterion not met", ""]
    for name, status, where in items:
        label = "MISSING" if status is None else ("PASS" if status else "FAIL")
        lines.append(f"{name:<{width}}  {label:<7}  {where}")
    passed = sum(1 for _, s, _ in items if s)
    lines += ["", f"{passed}/{len(items)} PASS", "", "Not produced (inputs missing):"]
    lines += [f"  - {m}" for m in ev.missing] or ["  (none)"]
    (out / "forecasting_dod.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ------------------------------------------------------------------------ LaTeX

LATEX_MAIN = r"""% Week 10 evidence: tables and figures read the CSV files one level up.
% Compile: cd evidence/week10/latex && xelatex main.tex   (or lualatex; needs fontspec)
% In the thesis: \usepackage{pgfplots,booktabs}, \pgfplotsset{compat=1.17},
% \usepgfplotslibrary{groupplots,statistics,fillbetween}, \newcommand{\evdir}{<path to evidence/week10>}
% then \input{<path>/fig9_gradual_timeline.tex} etc.
\documentclass[11pt]{article}
\usepackage{fontspec}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{booktabs,float,caption}
\usepackage{pgfplots,pgfplotstable}
\pgfplotsset{compat=1.17}
\usepgfplotslibrary{groupplots,statistics,fillbetween}
\newcommand{\evdir}{..}
\pgfplotsset{every axis/.append style={font=\small},
  w10/.style={width=0.95\linewidth,height=5cm,grid=major,grid style={gray!25}}}
\begin{document}
\section*{Tuần 10 -- Bảng và hình từ evidence/week10}
%BODY%
\end{document}
"""


class Tex(str):
    """A table cell that is already LaTeX (not escaped)."""


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v).replace("_", r"\_")


def latex_table(path: Path, caption: str, label: str, header: Sequence[str],
                rows: Sequence[Sequence[Any]], digits: int = 4, align: str | None = None) -> None:
    align = align or "l" + "r" * (len(header) - 1)
    def cell(c: Any) -> str:
        if isinstance(c, Tex):
            return str(c)
        if isinstance(c, str):
            return c.replace("_", r"\_").replace("%", r"\%")
        return _fmt(c, digits)

    body = "\n".join(" & ".join(cell(c) for c in row) + r" \\" for row in rows)
    path.write_text(
        "\\begin{table}[H]\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{align}}}\\toprule\n"
        + " & ".join(header) + r" \\\midrule" + "\n" + body + "\n\\bottomrule\\end{tabular}\n\\end{table}\n",
        encoding="utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else []


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


FIGURES: dict[str, tuple[str, str]] = {
    # file: (needs csv, tikz body). \evdir is the evidence/week10 folder.
    "fig1_dataset_distributions.tex": ("hist_utilization.csv", r"""
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.6cm,vertical sep=1.6cm},
  width=0.48\linewidth,height=4.2cm,ylabel={samples},grid=major,grid style={gray!25},ymin=0,
  xticklabel style={/pgf/number format/fixed},scaled x ticks=false]
\nextgroupplot[xlabel={utilization $U$}]
\addplot[fill=blue!30,draw=blue!60,const plot mark left] table[col sep=comma,x=left,y=count]{\evdir/hist_utilization.csv} \closedcycle;
\nextgroupplot[xlabel={processed Packet-In rate (pps)}]
\addplot[fill=blue!30,draw=blue!60,const plot mark left] table[col sep=comma,x=left,y=count]{\evdir/hist_processed_packet_in_rate.csv} \closedcycle;
\nextgroupplot[xlabel={CPU (\%)}]
\addplot[fill=blue!30,draw=blue!60,const plot mark left] table[col sep=comma,x=left,y=count]{\evdir/hist_process_cpu_percent.csv} \closedcycle;
\nextgroupplot[xlabel={response P95 (ms)}]
\addplot[fill=blue!30,draw=blue!60,const plot mark left] table[col sep=comma,x=left,y=count]{\evdir/hist_response_p95_ms.csv} \closedcycle;
\end{groupplot}
\end{tikzpicture}"""),
    "fig2_workload_profiles.tex": ("workload_profile_stable.csv", r"""
\begin{tikzpicture}
\begin{axis}[w10,xmin=0,xlabel={time in run (s)},ylabel={$U$ of c1},legend pos=outer north east,
  legend style={font=\scriptsize},cycle list name=color list]
\foreach \w/\n in {stable/stable,gradual/gradual,burst/burst,oscillating/oscillating,hot_switch/hot-switch}{
  \IfFileExists{\evdir/workload_profile_\w.csv}{
    \addplot+[no marks,thick] table[col sep=comma,x=t_s,y=utilization]{\evdir/workload_profile_\w.csv};
    \addlegendentryexpanded{\n}}{}}
\addplot[red,dashed,domain=0:200] {1.0};
\end{axis}
\end{tikzpicture}"""),
    "fig3_validation_models.tex": ("model_comparison_validation.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/model_comparison_validation.csv}\val
\begin{tikzpicture}
\begin{axis}[w10,ybar,bar width=12pt,symbolic x coords={persistence,moving_average_w3,linear_regression,random_forest,xgboost},
  xtick={persistence,moving_average_w3,linear_regression,random_forest,xgboost},xticklabels={Persistence,MA(3),LR,RF,XGBoost},ylabel={validation RMSE},
  nodes near coords,nodes near coords style={font=\tiny,/pgf/number format/.cd,fixed,precision=3},ymin=0]
\addplot table[x=model,y=rmse]{\val};
\end{axis}
\end{tikzpicture}"""),
    "fig4_test_rmse.tex": ("model_comparison_test.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/model_comparison_test.csv}\tst
\begin{tikzpicture}
\begin{axis}[w10,ybar,bar width=10pt,symbolic x coords={persistence,moving_average_w3,linear_regression,random_forest,xgboost},
  xtick={persistence,moving_average_w3,linear_regression,random_forest,xgboost},xticklabels={Persistence,MA(3),LR,RF,XGBoost},ylabel={test error},legend pos=north east,ymin=0]
\addplot table[x=model,y=rmse]{\tst}; \addlegendentry{RMSE}
\addplot table[x=model,y=p95_absolute_error]{\tst}; \addlegendentry{P95 $|e|$}
\end{axis}
\end{tikzpicture}"""),
    "fig5_per_workload_rmse.tex": ("per_workload_rmse_wide.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/per_workload_rmse_wide.csv}\pw
\begin{tikzpicture}
\begin{axis}[w10,ybar,bar width=5pt,symbolic x coords={stable,gradual,burst,oscillating,hot-switch},xtick={stable,gradual,burst,oscillating,hot-switch},
  ylabel={RMSE},legend style={font=\scriptsize,at={(0.5,-0.2)},anchor=north,legend columns=5},ymin=0]
\addplot table[x=workload,y=persistence]{\pw}; \addlegendentry{Persistence}
\addplot table[x=workload,y=moving_average_w3]{\pw}; \addlegendentry{MA(3)}
\addplot table[x=workload,y=linear_regression]{\pw}; \addlegendentry{LR}
\addplot table[x=workload,y=random_forest]{\pw}; \addlegendentry{RF}
\addplot table[x=workload,y=xgboost]{\pw}; \addlegendentry{XGBoost}
\end{axis}
\end{tikzpicture}"""),
    "fig6_test_vs_hotswitch.tex": ("ood_degradation.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/ood_degradation.csv}\ood
\begin{tikzpicture}
\begin{axis}[w10,ybar,bar width=10pt,symbolic x coords={validation,test,unseen},xtick={validation,test,unseen},
  xticklabels={Validation (ID),Test (ID),Hot-switch (OOD)},ylabel={RMSE},ymin=0,
  legend style={font=\scriptsize,at={(0.5,-0.22)},anchor=north,legend columns=3}]
\addplot table[x=split,y=rmse_without_gate]{\ood}; \addlegendentry{XGBoost, no gate}
\addplot table[x=split,y=rmse_with_gate]{\ood}; \addlegendentry{XGBoost + OOD gate}
\addplot table[x=split,y=rmse_persistence]{\ood}; \addlegendentry{Persistence}
\end{axis}
\end{tikzpicture}"""),
    "fig7_inference_p95.tex": ("inference_latency.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/inference_latency.csv}\lat
\begin{tikzpicture}
\begin{axis}[w10,ybar,bar width=12pt,ymode=log,log origin=infty,
  symbolic x coords={persistence,moving_average_w3,linear_regression,random_forest,xgboost},xtick={persistence,moving_average_w3,linear_regression,random_forest,xgboost},
  xticklabels={Persistence,MA(3),LR,RF,XGBoost},ylabel={single-sample P95 (ms, log)},legend pos=north west]
\addplot table[x=model,y=p95_ms]{\lat}; \addlegendentry{P95}
\addplot table[x=model,y=p50_ms]{\lat}; \addlegendentry{P50}
\end{axis}
\end{tikzpicture}"""),
    "fig8_quality_sensitivity.tex": ("quality_sensitivity_invalid_snapshot.csv", r"""
\pgfplotstableread[col sep=comma]{\evdir/quality_sensitivity_invalid_snapshot.csv}\qs
\begin{tikzpicture}
\begin{axis}[w10,xlabel={time in run (s), telemetry-loss run},ylabel={Quality Score},ymin=0,ymax=1.05,
  legend style={font=\scriptsize,at={(0.5,-0.32)},anchor=north,legend columns=3}]
\addplot+[no marks,thick] table[x=t_s,y=base]{\qs}; \addlegendentry{base}
\addplot+[no marks] table[x=t_s,y=consistency_heavy]{\qs}; \addlegendentry{consistency-heavy}
\addplot+[no marks] table[x=t_s,y=completeness_heavy]{\qs}; \addlegendentry{completeness-heavy}
\addplot+[no marks] table[x=t_s,y=freshness_heavy]{\qs}; \addlegendentry{freshness-heavy}
\addplot[only marks,mark=|,red,mark size=2pt,unbounded coords=discard] table[x=t_s,y=invalid_mark]{\qs};
\addlegendentry{Hard Gate: invalid}
\end{axis}
\end{tikzpicture}"""),
    "fig9_gradual_timeline.tex": ("runtime_gradual_c1.csv", r"""
\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 2,vertical sep=1.2cm},width=0.95\linewidth,height=5cm,
  grid=major,grid style={gray!25},unbounded coords=jump,ylabel={$U$},
  legend style={font=\scriptsize,at={(0.5,-0.32)},anchor=north,legend columns=4}]
\nextgroupplot[title={c1}]
\addplot[name path=hat,blue,thin] table[col sep=comma,x=t_target_s,y=u_hat]{\evdir/runtime_gradual_c1.csv};
\addplot[name path=up,draw=none,forget plot] table[col sep=comma,x=t_target_s,y=upper]{\evdir/runtime_gradual_c1.csv};
\addplot[blue!25,forget plot] fill between[of=hat and up];
\addplot[black,thick] table[col sep=comma,x=t_s,y=u]{\evdir/runtime_gradual_c1.csv};
\addplot[red,dashed,domain=0:200,forget plot] {1.0};
\addplot[only marks,red,mark=*,mark size=1.2pt] table[col sep=comma,x=t_target_s,y=alert_upper]{\evdir/runtime_gradual_c1.csv};
\addplot[only marks,orange,mark=x] table[col sep=comma,x=t_target_s,y=ood_u_hat]{\evdir/runtime_gradual_c1.csv};
\addplot[only marks,black,mark=triangle*] table[col sep=comma,x=t_s,y=measured_overload_u]{\evdir/runtime_gradual_c1.csv};
\nextgroupplot[title={c2},xlabel={seconds since workload start}]
\addplot[name path=hat2,blue,thin] table[col sep=comma,x=t_target_s,y=u_hat]{\evdir/runtime_gradual_c2.csv};
\addlegendentry{$\hat U$ at $t+h$}
\addplot[name path=up2,draw=none,forget plot] table[col sep=comma,x=t_target_s,y=upper]{\evdir/runtime_gradual_c2.csv};
\addplot[blue!25] fill between[of=hat2 and up2]; \addlegendentry{$+m_f$}
\addplot[black,thick] table[col sep=comma,x=t_s,y=u]{\evdir/runtime_gradual_c2.csv}; \addlegendentry{$U(t)$}
\addplot[red,dashed,domain=0:200] {1.0}; \addlegendentry{safe limit 1.0}
\addplot[only marks,red,mark=*,mark size=1.2pt] table[col sep=comma,x=t_target_s,y=alert_upper]{\evdir/runtime_gradual_c2.csv};
\addlegendentry{risk flag}
\addplot[only marks,orange,mark=x] table[col sep=comma,x=t_target_s,y=ood_u_hat]{\evdir/runtime_gradual_c2.csv};
\addlegendentry{OOD fallback}
\addplot[only marks,black,mark=triangle*] table[col sep=comma,x=t_s,y=measured_overload_u]{\evdir/runtime_gradual_c2.csv};
\addlegendentry{measured $U\ge 1$}
\end{groupplot}
\end{tikzpicture}"""),
}

FIGURE_CAPTIONS = {
    "fig1_dataset_distributions.tex": "Hình 1. Phân bố các biến chính của dataset (mẫu hợp lệ).",
    "fig2_workload_profiles.tex": "Hình 2. U(t) của năm workload (một run mỗi loại, controller c1).",
    "fig3_validation_models.tex": "Hình 3. So sánh năm mô hình trên Validation (RMSE).",
    "fig4_test_rmse.tex": "Hình 4. RMSE và P95 sai số tuyệt đối trên Test.",
    "fig5_per_workload_rmse.tex": "Hình 5. RMSE theo workload (Test; hot-switch là Unseen).",
    "fig6_test_vs_hotswitch.tex": "Hình 6. Test (ID) và Hot-switch (OOD): XGBoost có/không cổng OOD so với Persistence.",
    "fig7_inference_p95.tex": "Hình 7. Độ trễ suy luận một mẫu (P50, P95) khi testbed chạy.",
    "fig8_quality_sensitivity.tex": "Hình 8. Quality Score theo bốn bộ trọng số trên lượt mất telemetry; vạch đỏ: Hard Gate = invalid.",
    "fig9_gradual_timeline.tex": "Hình 9. Dự báo runtime trên workload gradual: $U$ hiện tại, $\\hat U$ tại $t+h$, $\\hat U + m_f$, ngưỡng 1,0 và cờ rủi ro.",
}


def latex_evidence(ev: Evidence) -> None:
    out = ev.out / "latex"
    out.mkdir(exist_ok=True)
    body: list[str] = []
    # tables
    test = _csv_rows(ev.out / "model_comparison_test.csv")
    if test:
        latex_table(out / "tab_model_comparison_test.tex", "Năm mô hình trên Test", "tab:w10-test",
                    ["Model", "MAE", "RMSE", "MAPE (\\%)", "$R^2$", "P95 $|e|$"],
                    [[r["model"], _num(r.get("mae")), _num(r.get("rmse")), _num(r.get("mape")),
                      _num(r.get("r2")), _num(r.get("p95_absolute_error"))] for r in test])
    sel = _csv_rows(ev.out / "model_selection.csv")
    if sel:
        latex_table(out / "tab_model_selection.tex", "Chọn mô hình trên Validation ($S_m$ nhỏ hơn là tốt hơn)",
                    "tab:w10-selection", ["Hạng", "Model", "RMSE", "P95 $|e|$", "RMSE/Persistence", "P95 latency (ms)", "$S_m$"],
                    [[int(r["rank"]), r["model"], _num(r.get("rmse")), _num(r.get("p95_absolute_error")),
                      _num(r.get("rmse_ratio_vs_persistence")), _num(r.get("inference_p95_ms")),
                      _num(r.get("selection_score"))] for r in sel], align="llrrrrr")
    stats = _csv_rows(ev.out / "dataset_statistics.csv")
    if stats:
        latex_table(out / "tab_dataset_statistics.tex", "Thống kê dataset (mẫu hợp lệ)", "tab:w10-dataset",
                    ["Biến", "n", "mean", "std", "min", "median", "p95", "max", "missing \\%"],
                    [[r["variable"], int(float(r["count"])), _num(r["mean"]), _num(r["std"]), _num(r["min"]),
                      _num(r["median"]), _num(r["p95"]), _num(r["max"]), _num(r["missing_pct"])] for r in stats], 3)
    qs = _csv_rows(ev.out / "quality_sensitivity.csv")
    if qs:
        latex_table(out / "tab_quality_sensitivity.tex", "Độ nhạy Quality Score theo bộ trọng số", "tab:w10-quality",
                    ["Biến thể", "mean", "min", "max $|\\Delta|$", "Spearman", "Kendall", "valid", "eligible"],
                    [[r["variant"], _num(r["score_mean"]), _num(r["score_min"]), _num(r["max_abs_diff_vs_base"]),
                      _num(r["spearman_vs_base"]), _num(r["kendall_vs_base"]), int(r["hard_valid"]),
                      int(r["runtime_eligible"])] for r in qs], 3)
    rt = _csv_rows(ev.out / "runtime_summary.csv")
    if rt:
        latex_table(out / "tab_runtime.tex", "Kịch bản runtime (khung workload)", "tab:w10-runtime",
                    ["Kịch bản", "READY", "OOD", "RMSE", "RMSE pers.", "Coverage", "Quá tải", "Báo trúng", "Hụt", "Nhầm"],
                    [[r["scenario"], _num(r["ready_ratio"]), _num(r["ood_fallback_ratio"]), _num(r["rmse"]),
                      _num(r["rmse_persistence"]), _num(r["coverage_upper_bound"]),
                      int(float(r["actual_overloads"])), int(float(r["overload_hits"])),
                      int(float(r["overload_misses"])), int(float(r["false_alarms"]))] for r in rt], 3)
    hz = _csv_rows(ev.out / "horizon_v1.csv")
    if hz:
        h = hz[0]
        latex_table(out / "tab_horizon.tex", "Horizon v1: $T_{detect}+T_{decision}+T_{migrate}\\le h$", "tab:w10-horizon",
                    ["Thành phần", "ms"],
                    [[Tex(r"$T_{detect}$ (P95 thu telemetry)"), _num(h["t_detect_ms"])],
                     [Tex(r"$T_{decision}$ (suy luận + bước dự báo)"), _num(h["t_decision_ms"])],
                     [Tex(r"$T_{migrate}$ (P95 migration)"), _num(h["t_migrate_ms"])],
                     ["Ngân sách cho quyết định", _num(h["t_decision_budget_ms"])],
                     ["Dư", _num(h["slack_ms"])]], 3)
    for tab in sorted(out.glob("tab_*.tex")):
        body.append(f"\\input{{{tab.name}}}")
    for name, (needs, tikz) in FIGURES.items():
        if not (ev.out / needs).is_file():
            continue
        (out / name).write_text(tikz.strip() + "\n", encoding="utf-8")
        body.append("\\begin{figure}[H]\\centering\n\\input{" + name + "}\n\\caption*{"
                    + FIGURE_CAPTIONS[name] + "}\n\\end{figure}")
    (out / "main.tex").write_text(LATEX_MAIN.replace("%BODY%", "\n\n".join(body)), encoding="utf-8")
    ev.written.append("latex/")


# ------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--training-version", default=MODEL_VERSION,
                        help="training run with *_metrics.json (default %(default)s)")
    parser.add_argument("--raw-root", type=Path, default=Path("data/experiment_runs"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("data/raw/forecast"))
    parser.add_argument("--snapshots-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--nominal-samples", type=int, default=9000)
    parser.add_argument("--scenario", action="append", default=[],
                        help="scenario=RUN_ID, overrides the default map (repeatable)")
    parser.add_argument("--ci", action="store_true", help="run make ci and git diff --check -> ci.txt")
    args = parser.parse_args()

    scenarios = dict(DEFAULT_SCENARIOS)
    for item in args.scenario:
        name, _, run_id = item.partition("=")
        scenarios[name.strip()] = run_id.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    ev = Evidence(args.out)
    training_dir = MODELS_ROOT / args.training_version

    dataset_evidence(ev, args.raw_root, args.nominal_samples)
    quality_evidence(ev, args.snapshots_dir, args.runtime_dir, scenarios)
    model_evidence(ev, training_dir)
    calibration_evidence(ev)
    runtime = runtime_evidence(ev, args.runtime_dir, scenarios)
    if args.ci:
        run_ci(ev)
    latex_evidence(ev)
    dod(ev, training_dir, runtime)
    print(f"\n{len(set(ev.written))} files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
