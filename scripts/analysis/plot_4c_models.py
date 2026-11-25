#!/usr/bin/env python3
"""Thesis figures of the two 4C40S models (chapter 5, "Độ trung thực của bản sao số mạng").

Every number is read from committed files, nothing is recomputed:
  fig_4c_forecast_skill   data/models/forecasting/forecast-models-v4/skill_vs_persistence.json
                          (RMSE / RMSE_persistence on test, hot-switch = unseen run)
  fig_4c_outcome          evidence/week15/outcome_v2/final_test_metrics.csv (test RMSE) and
                          final_coverage_metrics.json (coverage of m_U on test, by group)

    python3 scripts/analysis/plot_4c_models.py --out evidence/week16/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FORECAST = ROOT / "data/models/forecasting/forecast-models-v4/skill_vs_persistence.json"
OUTCOME = ROOT / "evidence/week15/outcome_v2"
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot-switch")
FORECASTERS = (("moving_average_w3", "MA3"), ("linear_regression", "LR"),
               ("random_forest", "RF"), ("xgboost", "XGB"))
OUTCOME_MODELS = (("analytical", "Analytical"), ("direct_rf", "Direct-RF"),
                  ("hybrid_rf", "Hybrid-RF"))
# flat, print-safe grays plus one accent; no gradients or shadows
COLORS = ("#bdbdbd", "#8c8c8c", "#4d4d4d", "#1f4e79")
STYLE = {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight"}


def skill_table(path: Path = FORECAST) -> dict[str, dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    table = {**data["test"], **data["unseen"]}
    missing = [w for w in WORKLOADS if w not in table]
    if missing:
        raise ValueError(f"{path}: no workload {missing}")
    return {w: {m: float(table[w][m]) for m, _ in FORECASTERS} for w in WORKLOADS}


def outcome_rmse(path: Path = OUTCOME / "final_test_metrics.csv",
                 targets: Sequence[str] = ("u_src_post", "u_dst_post", "i_post")
                 ) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {m: {} for m, _ in OUTCOME_MODELS}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test" and row["model"] in out and row["target"] in targets:
                out[row["model"]][row["target"]] = float(row["rmse"])
    for model, values in out.items():
        if set(values) != set(targets):
            raise ValueError(f"{path}: {model} lacks {set(targets) - set(values)}")
    return out


def mu_coverage(path: Path = OUTCOME / "final_coverage_metrics.json",
                group: str = "destination_load_level") -> tuple[str, float, list[Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    deployed = json.loads((OUTCOME / "final_uncertainty_margin.json").read_text())[
        "deployed_model"]
    entry = data["by_model"][deployed]["m_U"]
    rows = [r for r in entry["test"] if r["group"] == group]
    return deployed, float(entry["margin"]), sorted(rows, key=lambda r: float(r["value"]))


def plot_forecast(out: Path) -> Path:
    table = skill_table()
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    width = 0.19
    for k, (model, label) in enumerate(FORECASTERS):
        xs = [i + (k - 1.5) * width for i in range(len(WORKLOADS))]
        ax.bar(xs, [table[w][model] for w in WORKLOADS], width, label=label,
               color=COLORS[k], edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(len(WORKLOADS) - 1.5, color="black", linewidth=0.5, linestyle=":")
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels([w if w != "hot-switch" else "hot-switch\n(unseen)" for w in WORKLOADS])
    ax.set_ylabel("RMSE / RMSE persistence")
    ax.set_ylim(0, max(max(v.values()) for v in table.values()) * 1.12)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=8)
    path = out / "fig_4c_forecast_skill"
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path


def plot_outcome(out: Path) -> Path:
    rmse = outcome_rmse()
    deployed, margin, cov = mu_coverage()
    targets = ("u_src_post", "u_dst_post", "i_post")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.3, 2.6),
                                   gridspec_kw={"width_ratios": [1.3, 1]})
    width = 0.26
    for k, (model, label) in enumerate(OUTCOME_MODELS):
        xs = [i + (k - 1) * width for i in range(len(targets))]
        color = {"analytical": COLORS[0], "hybrid_rf": COLORS[1]}.get(model, COLORS[3])
        ax1.bar(xs, [rmse[model][t] for t in targets], width, label=label,
                color=color, edgecolor="black", linewidth=0.4)   # accent = deployed model
    ax1.set_xticks(range(len(targets)))
    ax1.set_xticklabels([r"$U_{src}$", r"$U_{dst}$", r"$I$"])
    ax1.set_ylabel("RMSE (test)")
    ax1.legend(frameon=False, fontsize=8)
    levels = [f"{float(r['value']):.2f}" for r in cov]
    ax2.bar(levels, [r["coverage"] for r in cov], 0.8, color=COLORS[3], edgecolor="black",
            linewidth=0.4)
    for i, r in enumerate(cov):
        ax2.text(i, 0.53, f"{r['covered']}/{r['n']}", ha="center", fontsize=6.5, color="white")
    ax2.axhline(0.95, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylim(0.5, 1.08)
    ax2.set_xlabel("destination load level")
    ax2.set_ylabel(f"coverage of $m_U$ = {margin:.3f}")
    ax2.set_title(deployed.replace("direct_rf", "Direct-RF"), fontsize=9)
    path = out / "fig_4c_outcome"
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/week16/figures")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    for path in (plot_forecast(args.out), plot_outcome(args.out)):
        print(f"wrote {path}.png / .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
