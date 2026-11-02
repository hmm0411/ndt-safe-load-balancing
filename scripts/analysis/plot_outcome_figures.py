#!/usr/bin/env python3
"""Week 12 figures fig_w12_01..08 of the Post-Migration Outcome Model (Day 6).

Every number comes from the frozen Day 4 choice (model_selection.json + completed
test_lock.json, files checked byte for byte) and the saved models; nothing is re-chosen.
Test-split figures show Analytical, Best Direct and Best Hybrid; the residual figure uses
the Analytical residual of all 300 rows (no trained model involved).

    python3 scripts/analysis/plot_outcome_figures.py            # -> evidence/week12/figures/

  01 actual vs predicted u_src_post (3 panels)   05 P95 |error| per target
  02 actual vs predicted u_dst_post, saturated    06 Analytical residual by workload
     and overload attempts marked                 07 |error| vs switch load share
  03 actual vs predicted r95_post                 08 error of u_dst_post vs destination U
  04 RMSE per target with bootstrap 95 % CI          before the migration
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome.artifact import read_csv  # noqa: E402
from src.twin.outcome.dataset import load_dataset  # noqa: E402
from src.twin.outcome.frozen import ROLES, FinalPaths, FrozenModels, load_frozen  # noqa: E402
from src.twin.outcome.metrics import regression_metrics  # noqa: E402

GREY, DARK, ACCENT = "#A6A6A6", "#555555", "#1F6FB5"
LABEL = {"analytical": "Analytical", "direct_rf": "Direct-RF", "direct_xgb": "Direct-XGB",
         "hybrid_rf": "Hybrid-RF", "hybrid_xgb": "Hybrid-XGB"}
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot_switch")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "savefig.dpi": 200, "savefig.bbox": "tight"})


def colours(names: Sequence[str]) -> dict[str, str]:
    return dict(zip(names, (GREY, DARK, ACCENT)))


def actual_vs_pred(out: Path, rows: list[dict[str, Any]], pred: dict[str, dict[str, list]],
                   names: list[str], target: str, unit: str, number: str,
                   mark: bool = False) -> Path:
    truth = [float(r[target]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), sharex=True, sharey=True)
    values = truth + [v for n in names for v in pred[n][target]]
    lo, hi = min(values), max(values)
    pad = (hi - lo) * 0.05
    for ax, name in zip(axes, names):
        p = pred[name][target]
        rmse = regression_metrics(truth, p)["rmse"]
        normal = [i for i, r in enumerate(rows)
                  if not (mark and (r["saturated_before"] or r["saturated_after"]
                                    or r["overload_after"]))]
        ax.scatter([truth[i] for i in normal], [p[i] for i in normal], s=12,
                   color=colours(names)[name], alpha=0.8, linewidths=0)
        if mark:
            sat = [i for i, r in enumerate(rows) if r["saturated_before"] or r["saturated_after"]]
            ovl = [i for i, r in enumerate(rows) if r["overload_after"]]
            ax.scatter([truth[i] for i in sat], [p[i] for i in sat], s=26, marker="x",
                       color="black", label="saturated")
            ax.scatter([truth[i] for i in ovl], [p[i] for i in ovl], s=40, facecolors="none",
                       edgecolors="black", label="overload after (U >= 1)")
            ax.axvline(1.0, color=GREY, lw=0.8, ls="--")
            ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{LABEL[name]}  (RMSE {rmse:.4f})")
        ax.set_xlabel(f"actual {target} ({unit})")
    axes[0].set_ylabel(f"predicted ({unit})")
    if mark:
        axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle(f"Test split (n = {len(rows)}): actual vs predicted {target}; "
                 "line y = x", y=1.02)
    path = out / f"fig_w12_{number}_actual_vs_predicted_{target}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def rmse_ci(out: Path, boot: list[dict[str, Any]], names: list[str],
            targets: Sequence[str]) -> Path:
    fig, axes = plt.subplots(1, len(targets), figsize=(3.4 * len(targets), 3.0))
    fig.subplots_adjust(wspace=0.55)
    for ax, target in zip(axes, targets):
        for k, name in enumerate(names):
            row = next(r for r in boot if r["kind"] == "rmse" and r["model"] == name
                       and r["target"] == target)
            est, low, high = float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])
            ax.errorbar([k], [est], yerr=[[est - low], [high - est]], fmt="o",
                        color=colours(names)[name], capsize=3, ms=5)
            ax.annotate(f"{est:.4f}", (k, est), textcoords="offset points", xytext=(6, 0),
                        fontsize=7, va="center")
        ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=20)
        ax.set_xlim(-0.5, len(names) - 0.3)
        ax.set_ylim(bottom=0)
        ax.set_title(target)
    axes[0].set_ylabel("RMSE on test")
    fig.suptitle("RMSE per target on test with paired bootstrap 95 % CI", y=1.03)
    path = out / "fig_w12_04_rmse_per_target_ci.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def p95_error(out: Path, test_rows: list[dict[str, Any]], names: list[str],
              targets: Sequence[str]) -> Path:
    fig, axes = plt.subplots(1, len(targets), figsize=(3.4 * len(targets), 3.0))
    fig.subplots_adjust(wspace=0.55)
    for ax, target in zip(axes, targets):
        vals = [next(float(r["p95_abs_error"]) for r in test_rows
                     if r["model"] == n and r["target"] == target) for n in names]
        ax.bar(range(len(names)), vals, color=[colours(names)[n] for n in names], width=0.6)
        for k, v in enumerate(vals):
            ax.annotate(f"{v:.4f}", (k, v), textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), [LABEL[n] for n in names], rotation=20)
        ax.set_title(target)
    axes[0].set_ylabel("P95 |error| on test")
    fig.suptitle("Tail error: 95th percentile of |predicted - actual| on test", y=1.03)
    path = out / "fig_w12_05_p95_abs_error.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def residual_by_workload(out: Path, rows: list[dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=False)
    for ax, target in zip(axes, ("u_src_post", "u_dst_post")):
        data = [[float(r[target]) - float(r[f"analytical_{target}"]) for r in rows
                 if r["workload"] == w] for w in WORKLOADS]
        ax.boxplot(data, widths=0.5, showfliers=False,
                   medianprops={"color": ACCENT}, boxprops={"color": DARK},
                   whiskerprops={"color": DARK}, capprops={"color": DARK})
        for k, values in enumerate(data, start=1):
            jitter = [k + ((i % 7) - 3) * 0.04 for i in range(len(values))]
            ax.scatter(jitter, values, s=6, color=GREY, alpha=0.7, linewidths=0)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(range(1, len(WORKLOADS) + 1), WORKLOADS, rotation=15)
        ax.set_title(f"{target}: actual - Analytical")
        ax.set_ylabel("residual (U)")
    fig.suptitle(f"Analytical residual by workload (all {len(rows)} attempts); a pattern "
                 "is what the residual learner can learn", y=1.03)
    path = out / "fig_w12_06_residual_by_workload.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def error_vs(out: Path, rows: list[dict[str, Any]], pred: dict[str, dict[str, list]],
             names: list[str], feature: str, xlabel: str, number: str, slug: str,
             targets: Sequence[str], signed: bool) -> Path:
    fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 3.2))
    axes_list = list(axes) if len(targets) > 1 else [axes]
    xs = [float(r[feature]) for r in rows]
    for ax, target in zip(axes_list, targets):
        for name in names:
            err = [p - float(r[target]) for p, r in zip(pred[name][target], rows)]
            ys = err if signed else [abs(e) for e in err]
            ax.scatter(xs, ys, s=12, color=colours(names)[name], alpha=0.8, linewidths=0,
                       label=LABEL[name])
        if signed:
            ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(("predicted - actual " if signed else "|error| of ") + target)
        ax.set_title(target)
    axes_list[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(f"Test split (n = {len(rows)}): error vs {xlabel}", y=1.03)
    path = out / f"fig_w12_{number}_{slug}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--models-config", type=Path, default=Path("configs/outcome/models.yaml"))
    parser.add_argument("--eval-config", type=Path,
                        default=Path("configs/outcome/evaluation.yaml"))
    parser.add_argument("--direct-dir", type=Path, default=Path("data/outcome/reports/direct"))
    parser.add_argument("--hybrid-dir", type=Path, default=Path("data/outcome/reports/hybrid"))
    parser.add_argument("--final-dir", type=Path, default=Path("data/outcome/reports/final"))
    parser.add_argument("--out", type=Path, default=Path("evidence/week12/figures"))
    args = parser.parse_args()
    try:
        paths = FinalPaths(args.dataset, args.config, args.models_config, args.eval_config,
                           args.direct_dir, args.hybrid_dir, args.final_dir)
        frozen = load_frozen(paths)
        rows = load_dataset(args.dataset / "outcome_dataset.csv")
        test = [r for r in rows if r["split"] == "test"]
        models = FrozenModels(paths, frozen)
        pred = models.predict_roles(test)
        names = [frozen.name_of(role) for role in ROLES]
        boot = read_csv(args.final_dir / "bootstrap_ci.csv")
        test_metrics = read_csv(args.final_dir / "test_metrics.csv")
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    targets = ("u_src_post", "u_dst_post", "i_post", "r95_post")
    written = [
        actual_vs_pred(args.out, test, pred, names, "u_src_post", "U", "01"),
        actual_vs_pred(args.out, test, pred, names, "u_dst_post", "U", "02", mark=True),
        actual_vs_pred(args.out, test, pred, names, "r95_post", "ms", "03"),
        rmse_ci(args.out, boot, names, targets),
        p95_error(args.out, test_metrics, names, targets),
        residual_by_workload(args.out, rows),
        error_vs(args.out, test, pred, names, "switch_control_load_share",
                 "switch load share before the migration", "07", "error_vs_switch_share",
                 ("u_src_post", "u_dst_post"), signed=False),
        error_vs(args.out, test, pred, names, "dst_u_mean",
                 "destination U before the migration", "08", "error_vs_destination_u",
                 ("u_dst_post",), signed=True),
    ]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
