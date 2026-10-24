#!/usr/bin/env python3
"""Week 11 migration-outcome dataset report: manifest, split, quality, statistics, figures.

    python scripts/analysis/migration_dataset_report.py ROOT [--out DIR]

ROOT = an attempt root (the pilot now, data/experiment_runs/migration-outcome-v1 on
Day 6). Reads only; writes into --out (default results/week11/<ROOT name>):

  migration_outcome_manifest.json  split_manifest.json  dataset_quality_report.json
  migration_latency.json  analytic_check.json
  attempt_summary.csv  committed_summary.csv  rollback_summary.csv
  workload_distribution.csv  noop_vs_migrate.csv
  figures/fig_w11_01 ... fig_w11_08 (.png)

Plan sections 27, 28, 36, 44, 45. Every number in the figures comes from
attempt_summary.csv (same rows).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.experiments.migration.config import load_outcome_config  # noqa: E402
from src.experiments.migration.dataset_report import (  # noqa: E402
    STATIONARY_WORKLOADS,
    build_dataset_report,
    write_report,
)
from src.experiments.migration.pilot import load_pilot_settings  # noqa: E402

WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot_switch")
GREY = "#9a9a9a"
DARK = "#404040"
ACCENT = "#1f6fb4"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
        "legend.frameon": False,
    })


def _save(fig: Any, path: Path, written: list[Path]) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    written.append(path)


def _values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)]


def _present(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen = {r.get("workload") for r in rows}
    return [w for w in WORKLOADS if w in seen]


def _strip(ax: Any, rows: Sequence[dict[str, Any]], key: str, ylabel: str) -> None:
    """One column of points per workload (stationary ones in the accent colour)."""
    names = _present(rows)
    for i, workload in enumerate(names):
        ys = _values([r for r in rows if r.get("workload") == workload], key)
        xs = [i + 0.12 * ((j % 5) - 2) / 2 for j in range(len(ys))]
        color = ACCENT if workload in STATIONARY_WORKLOADS else GREY
        ax.scatter(xs, ys, s=18, color=color, alpha=0.8, linewidths=0)
        if ys:
            ys_sorted = sorted(ys)
            ax.hlines(ys_sorted[len(ys_sorted) // 2], i - 0.25, i + 0.25, color=DARK, linewidth=1)
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right")
    ax.set_ylabel(ylabel)


def draw_figures(report: dict[str, Any], out_dir: Path) -> list[Path]:
    style()
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = [r for r in report["tables"]["attempt_summary"]]
    committed = [r for r in rows if r.get("transaction_status") == "COMMITTED"
                 and r.get("training_eligible")]
    executed = [r for r in rows if r.get("transaction_status") in ("COMMITTED", "RESTORED", "FAILED")]

    # 1. U before / after, source and destination (COMMITTED, eligible)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, side, title in ((axes[0], "src", "Controller nguồn"), (axes[1], "dst", "Controller đích")):
        before = _values(committed, f"{side}_u_before_mean")
        after = _values(committed, f"{side}_u_after_mean")
        bins = [i / 20 for i in range(0, 25)]
        ax.hist(before, bins=bins, color=GREY, alpha=0.7, label="trước")
        ax.hist(after, bins=bins, histtype="step", color=ACCENT, linewidth=1.4, label="sau")
        ax.axvline(1.0, color=DARK, linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("U (trung bình cửa sổ 5 snapshot)")
    axes[0].set_ylabel("số attempt COMMITTED")
    axes[1].legend()
    _save(fig, figures / "fig_w11_01_u_before_after.png", written)

    # 2. control-load share of the moved switch
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    hot = _values([r for r in committed if r.get("target_hot")], "switch_share_before_mean")
    other = _values([r for r in committed if not r.get("target_hot")], "switch_share_before_mean")
    bins = [i / 40 for i in range(0, 29)]
    ax.hist(other, bins=bins, color=GREY, label="switch chọn ngẫu nhiên")
    ax.hist(hot, bins=bins, color=ACCENT, label="switch nóng (hot_switch)")
    ax.set_xlabel("control-load share của switch được migrate (cửa sổ trước)")
    ax.set_ylabel("số attempt")
    ax.legend()
    _save(fig, figures / "fig_w11_02_switch_share.png", written)

    # 3. Delta U source / destination per workload
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    _strip(axes[0], committed, "delta_src_u_mean", "ΔU nguồn (sau − trước)")
    _strip(axes[1], committed, "delta_dst_u_mean", "ΔU đích (sau − trước)")
    for ax in axes:
        ax.axhline(0.0, color=DARK, linewidth=0.8)
    _save(fig, figures / "fig_w11_03_delta_u.png", written)

    # 4. imbalance before / after (paired, identity line)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    pairs = [(float(r["imbalance_before_mean"]), float(r["imbalance_after_mean"])) for r in committed
             if r.get("imbalance_before_mean") is not None and r.get("imbalance_after_mean") is not None]
    if pairs:
        top_value = max(max(p) for p in pairs) * 1.1
        ax.plot([0, top_value], [0, top_value], color=DARK, linestyle="--", linewidth=0.8)
        ax.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=18, color=ACCENT, linewidths=0)
    ax.set_xlabel("imbalance trước")
    ax.set_ylabel("imbalance sau")
    ax.set_title("Dưới đường chéo = cân bằng hơn")
    _save(fig, figures / "fig_w11_04_imbalance.png", written)

    # 5. migration latency and post-window settle time
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    _strip(axes[0], executed, "transaction_s", "thời gian giao dịch (s)")
    _strip(axes[1], executed, "settle_s", "chờ đủ cửa sổ sau (s)")
    _save(fig, figures / "fig_w11_05_latency.png", written)

    # 6. quality score before / after (minimum over each window)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    q = [(float(r["before_quality_min"]), float(r["after_quality_min"])) for r in executed
         if r.get("before_quality_min") is not None and r.get("after_quality_min") is not None]
    if q:
        low = min(min(p) for p in q) - 0.02
        ax.plot([low, 1.0], [low, 1.0], color=DARK, linestyle="--", linewidth=0.8)
        ax.scatter([p[0] for p in q], [p[1] for p in q], s=18, color=ACCENT, linewidths=0)
    ax.set_xlabel("Quality Score nhỏ nhất, cửa sổ trước")
    ax.set_ylabel("Quality Score nhỏ nhất, cửa sổ sau")
    _save(fig, figures / "fig_w11_06_quality.png", written)

    # 7. share vs -Delta U_src (the evidence for the Week 12 analytic model)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    check = report["analytic_check"]["stationary_workloads"]["share_vs_neg_delta_src"]
    for subset, color, label in (
        ([r for r in committed if r.get("workload") not in STATIONARY_WORKLOADS], GREY,
         "burst, gradual, oscillating"),
        ([r for r in committed if r.get("workload") in STATIONARY_WORKLOADS], ACCENT,
         "stable, hot_switch"),
    ):
        xs = _values(subset, "switch_share_before_mean")
        ys = [-v for v in _values(subset, "delta_src_u_mean")]
        if len(xs) == len(ys):
            ax.scatter(xs, ys, s=20, color=color, linewidths=0, label=label)
    if check.get("slope") is not None:
        xs_all = _values(committed, "switch_share_before_mean")
        x_line = [0.0, max(xs_all) if xs_all else 0.6]
        ax.plot(x_line, [check["intercept"] + check["slope"] * x for x in x_line], color=ACCENT,
                linewidth=1, label=f"fit stable+hot_switch, r = {check['pearson_r']:.2f}"
                if check.get("pearson_r") is not None else "fit stable+hot_switch")
    ax.axhline(0.0, color=DARK, linewidth=0.8)
    ax.set_xlabel("control-load share của switch (cửa sổ trước)")
    ax.set_ylabel("−ΔU nguồn")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    _save(fig, figures / "fig_w11_07_share_vs_neg_delta_src.png", written)

    # 8. analytic estimate (rate / C_safe) vs observed -Delta U_src
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    est = [(-float(r["expected_delta_src"]), -float(r["delta_src_u_mean"])) for r in committed
           if r.get("expected_delta_src") is not None and r.get("delta_src_u_mean") is not None]
    if est:
        hi = max(max(abs(v) for v in p) for p in est) * 1.1
        ax.plot([0, hi], [0, hi], color=DARK, linestyle="--", linewidth=0.8, label="y = x")
        for subset, color in (
            ([r for r in committed if r.get("workload") not in STATIONARY_WORKLOADS], GREY),
            ([r for r in committed if r.get("workload") in STATIONARY_WORKLOADS], ACCENT),
        ):
            xs = [-v for v in _values(subset, "expected_delta_src")]
            ys = [-v for v in _values(subset, "delta_src_u_mean")]
            if len(xs) == len(ys):
                ax.scatter(xs, ys, s=20, color=color, linewidths=0)
    ax.set_xlabel("ước lượng giải tích: tốc độ switch / C_safe")
    ax.set_ylabel("−ΔU nguồn đo được")
    ax.set_title("Xanh: stable, hot_switch; xám: workload còn lại")
    ax.legend(loc="upper left")
    _save(fig, figures / "fig_w11_08_expected_vs_observed.png", written)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="attempt root (pilot or migration-outcome-v1)")
    parser.add_argument("--out", type=Path, help="output directory (default results/week11/<root>)")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/migration_outcome.yaml"))
    parser.add_argument("--pilot-config", type=Path,
                        default=Path("configs/experiments/migration_pilot.yaml"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    out = args.out or Path("results/week11") / args.root.name
    report = build_dataset_report(args.root, load_outcome_config(args.config),
                                  load_pilot_settings(args.pilot_config), args.config)
    written = write_report(report, out)
    if not args.no_figures:
        written += draw_figures(report, out)
    quality = report["quality_report"]
    print(f"{quality['total_attempts']} attempts ({quality['planned_attempts']} planned, "
          f"{quality['missing_attempts']} missing), valid {quality['valid_count']}, "
          f"eligible COMMITTED {quality['training_eligible_committed_count']}, "
          f"NO_OP {quality['no_op_count']}, freeze consistent {quality['freeze_consistent']}")
    for path in written:
        print("wrote", path)
    return 0 if quality["invalid_count"] == 0 and quality["freeze_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
