#!/usr/bin/env python3
"""One-sided coverage of the deployed forecast margin m_f with its companions (protocol 6).

Coverage P(U <= U_hat + m_f) on test (per workload) and hot-switch (unseen), next to:
the margin width, the two-sided reference |U - U_hat| <= m_f, coverage in the decision
region U >= 0.8, false alarm / missed overload of U_hat + m_f >= 1, and a 95 % bootstrap
interval that resamples whole runs (per-second samples of one run are correlated).
Nothing is refit: the model and m_f are read from the frozen artifact.

    NDT_FORECAST_PROFILE=4c40s python3 scripts/analysis/margin_coverage_report.py \
        --out results/coverage/forecast_v4_coverage.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.forecasting.margin import margin_for, overload_confusion  # noqa: E402

DECISION_REGION = 0.8


def _cov(e: np.ndarray, m: np.ndarray) -> float | None:
    return float(np.mean(e <= m)) if e.size else None


def bootstrap_by_run(e: np.ndarray, m: np.ndarray, runs: Sequence[str], n_boot: int = 2000,
                     seed: int = 20261127) -> tuple[float, float] | None:
    """95 % percentile interval of one-sided coverage, resampling whole runs."""
    ids = sorted(set(runs))
    if len(ids) < 2:
        return None
    index = {r: [i for i, x in enumerate(runs) if x == r] for r in ids}
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        rows = [i for r in (rng.choice(ids) for _ in ids) for i in index[r]]
        stats.append(float(np.mean(e[rows] <= m[rows])))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def report(y: np.ndarray, pred: np.ndarray, m: np.ndarray, runs: Sequence[str],
           threshold: float = 1.0, n_boot: int = 2000) -> dict[str, Any]:
    if not (y.shape == pred.shape == m.shape) or y.size != len(runs) or y.size == 0:
        raise ValueError("y, pred, m and runs must be non-empty and of one length")
    e = y - pred
    region = y >= DECISION_REGION
    return {
        "samples": int(y.size), "runs": len(set(runs)),
        "coverage_one_sided": _cov(e, m),
        "coverage_one_sided_ci95_by_run": bootstrap_by_run(e, m, runs, n_boot),
        "coverage_two_sided_reference": float(np.mean(np.abs(e) <= m)),
        f"coverage_one_sided_given_U_ge_{DECISION_REGION}": _cov(e[region], m[region]),
        f"samples_U_ge_{DECISION_REGION}": int(region.sum()),
        "margin_mean": float(np.mean(m)),
        "overload": overload_confusion(y, pred + m, threshold),
    }


def grouped(rows: Sequence[Mapping[str, Any]], pred: np.ndarray, margins: Mapping[str, Any],
            key: str, n_boot: int) -> dict[str, Any]:
    y = np.asarray([float(r["future_utilization"]) for r in rows])
    m = np.asarray([margin_for(dict(margins), str(r["controller_id"])) for r in rows])
    runs = [str(r["run_id"]) for r in rows]
    out = {"all": report(y, pred, m, runs, n_boot=n_boot)}
    for g in sorted({str(r[key]) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r[key]) == g]
        out[g] = report(y[idx], pred[idx], m[idx], [runs[i] for i in idx], n_boot=n_boot)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    from scripts.experiments.compute_forecast_margin import load_predictor, time_ordered
    from src.twin.forecasting.splitter import split_records
    from src.twin.forecasting.versions import (
        MODEL_VERSION,
        MODELS_ROOT,
        PROCESSED_ROOT,
        dataset_version,
    )

    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--model-version", default=MODEL_VERSION)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    model_dir = MODELS_ROOT / args.model_version
    margins = json.loads((model_dir / "error_margin.json").read_text(encoding="utf-8"))
    names = list(json.loads((model_dir / "feature_schema.json").read_text())["feature_names"])
    predict = load_predictor(model_dir, str(margins["model_name"]), names)
    dataset = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
    with dataset.open(encoding="utf-8", newline="") as handle:
        splits = split_records([dict(row) for row in csv.DictReader(handle)])
    result: dict[str, Any] = {
        "model_version": args.model_version, "model_name": margins["model_name"],
        "margin_method": margins.get("method"), "m_f_global": margins["global"],
        "one_sided_q95_validation": margins.get("one_sided_q95"),
        "definition": "coverage = P(U <= U_hat + m_f); two-sided only as reference",
        "splits": {}}
    for split in ("validation", "test", "unseen"):
        rows = time_ordered(list(splits[split]))
        pred = np.asarray(predict(rows), dtype=float)
        result["splits"][split] = grouped(rows, pred, margins, "workload_type", args.n_boot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for split, groups in result["splits"].items():
        for g, r in groups.items():
            ci = r["coverage_one_sided_ci95_by_run"]
            print(f"{split:10} {g:12} one-sided {r['coverage_one_sided']:.3f} "
                  f"{'' if ci is None else f'[{ci[0]:.3f}-{ci[1]:.3f}]'}  "
                  f"two-sided {r['coverage_two_sided_reference']:.3f}  n={r['samples']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
