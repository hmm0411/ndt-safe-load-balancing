#!/usr/bin/env python3
"""Why is the coverage of forecast-models-v4 below 0.95 on test? (read-only diagnosis, 26/11)

The v4 run (25/11) reported, on test: coverage 0.881 without the OOD gate, 0.811 with it, and
the gate on for 26.9 % of test samples (76 % of oscillating). Two facts make those offline
numbers differ from what the runtime does:

* the offline gate (compute_forecast_margin.py) watches every feature, while the runtime
  config ignores ``ood_ignore_features`` (process_memory_rss_mb, out of range on 14 % of
  test samples);
* coverage there is two-sided (|U - U_hat| <= m_f); the Safety Gate only needs the upper
  side (U <= U_hat + m_f): an over-prediction is a false alarm, not a missed overload.

This script recomputes, per split and per RUN (one test run per workload, so a workload
number is a single run): coverage two-sided / upper, without gate, with the offline gate and
with the runtime gate; which features put each run out of range; and, for those features,
the train range next to the run's values. It changes no model, margin or artifact.

    NDT_FORECAST_PROFILE=4c40s python3 scripts/analysis/forecast_ood_diagnosis.py \
        --runtime-config configs/forecasting/runtime_4c40s.yaml \
        --out evidence/week16/forecast_v4_ood_diagnosis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.compute_forecast_margin import (  # noqa: E402
    load_predictor,
    time_ordered,
)
from src.twin.forecasting.baselines import persistence_predict  # noqa: E402
from src.twin.forecasting.linear_model import records_to_matrix  # noqa: E402
from src.twin.forecasting.margin import margin_for  # noqa: E402
from src.twin.forecasting.ood import FeatureRanges, gate_mask  # noqa: E402
from src.twin.forecasting.splitter import split_records  # noqa: E402
from src.twin.forecasting.versions import (  # noqa: E402
    MODEL_VERSION,
    MODELS_ROOT,
    PROCESSED_ROOT,
    dataset_version,
)

Predict = Callable[[list[dict[str, Any]]], Any]
GATES = ("no_gate", "offline_gate", "runtime_gate")


def _cov(y: np.ndarray, pred: np.ndarray, m: np.ndarray) -> dict[str, float]:
    err = y - pred
    return {"two_sided": float(np.mean(np.abs(err) <= m)), "upper": float(np.mean(err <= m))}


def _summary(values: np.ndarray) -> dict[str, float]:
    return {"min": float(values.min()), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def diagnose(rows: Sequence[Mapping[str, Any]], predict: Predict, margins: Mapping[str, Any],
             ranges: FeatureRanges, ignore: Sequence[str]) -> dict[str, Any]:
    """Coverage and gate activity of one split, overall and per run."""
    ordered = time_ordered([dict(r) for r in rows])
    if not ordered:
        raise ValueError("no rows")
    y = np.asarray([float(r["future_utilization"]) for r in ordered])
    current = np.asarray(persistence_predict(ordered), dtype=float)
    model = np.asarray(predict(ordered), dtype=float)
    m = np.asarray([margin_for(dict(margins), str(r["controller_id"])) for r in ordered])
    names = list(ranges.feature_names)
    x = records_to_matrix(ordered, names)
    lo, hi = ranges._bounds()
    outside = ~((x >= lo) & (x <= hi))                       # (rows, features), NaN = outside
    keep = np.asarray([n not in set(ignore) for n in names])
    sequences = [str(r["sequence_id"]) for r in ordered]
    active = {
        "no_gate": np.zeros(len(ordered), dtype=bool),
        "offline_gate": gate_mask(outside.any(axis=1), sequences, ranges.consecutive_steps),
        "runtime_gate": gate_mask((outside & keep).any(axis=1), sequences,
                                  ranges.consecutive_steps),
    }
    preds = {g: np.where(active[g], current, model) for g in GATES}
    runs = np.asarray([str(r["run_id"]) for r in ordered])

    def block(mask: np.ndarray) -> dict[str, Any]:
        out: dict[str, Any] = {"samples": int(mask.sum())}
        for g in GATES:
            out[g] = {**_cov(y[mask], preds[g][mask], m[mask]),
                      "gate_on": float(active[g][mask].mean())}
        out["persistence"] = _cov(y[mask], current[mask], m[mask])
        share = outside[mask].mean(axis=0)
        out["out_of_range_share"] = {n: float(s) for n, s in zip(names, share) if s > 0}
        out["out_of_range_values"] = {
            n: {"train_min": float(ranges.minimum[i]), "train_max": float(ranges.maximum[i]),
                "run": _summary(x[mask][:, i][np.isfinite(x[mask][:, i])])}
            for i, n in enumerate(names) if share[i] > 0}
        return out

    return {"overall": block(np.ones(len(ordered), dtype=bool)),
            "by_run": {run: block(runs == run) for run in sorted(set(runs))}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model-version", default=MODEL_VERSION)
    ap.add_argument("--runtime-config", type=Path, required=True,
                    help="orchestrator forecast config (its ood_ignore_features)")
    ap.add_argument("--splits", nargs="+", default=["validation", "test", "unseen"])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    model_dir = MODELS_ROOT / args.model_version
    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    if Path(str(runtime["model_dir"])).name != args.model_version:
        raise SystemExit(f"{args.runtime_config} serves {runtime['model_dir']}, "
                         f"not {args.model_version}")
    ignore = [str(n) for n in runtime.get("ood_ignore_features") or ()]
    features = list(json.loads((model_dir / "feature_schema.json").read_text())["feature_names"])
    model_name = json.loads((model_dir / "model_selection.json").read_text())["selected"]
    margins = json.loads((model_dir / "error_margin.json").read_text())
    ranges = FeatureRanges.from_dict(
        json.loads((model_dir / "feature_ranges.json").read_text()), features)
    predict = load_predictor(model_dir, model_name, features)
    dataset = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
    with dataset.open(encoding="utf-8", newline="") as handle:
        splits = split_records([dict(row) for row in csv.DictReader(handle)])
    report: dict[str, Any] = {
        "model_version": args.model_version, "model": model_name,
        "dataset": str(dataset), "runtime_config": str(args.runtime_config),
        "runtime_ignore": ignore, "margin": "per-controller m_f of error_margin.json",
        "coverage": {"two_sided": "|U - U_hat| <= m_f (as in coverage.json)",
                     "upper": "U <= U_hat + m_f (what the Safety Gate needs)"},
        "splits": {s: diagnose(splits[s], predict, margins, ranges, ignore)
                   for s in args.splits},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{'split / run':<38}{'n':>5}  {'2-sided':>16}  {'upper':>16}  {'gate on':>14}")
    print(f"{'':<38}{'':>5}  {'none off  rt':>16}  {'none off  rt':>16}  {'off   rt':>14}")
    for split, data in report["splits"].items():
        for label, b in [(split, data["overall"])] + [
                (f"  {run}", v) for run, v in data["by_run"].items()]:
            two = " ".join(f"{b[g]['two_sided']:.2f}" for g in GATES)
            up = " ".join(f"{b[g]['upper']:.2f}" for g in GATES)
            gate = f"{b['offline_gate']['gate_on']:.2f}  {b['runtime_gate']['gate_on']:.2f}"
            print(f"{label:<38}{b['samples']:>5}  {two:>16}  {up:>16}  {gate:>14}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
