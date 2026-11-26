#!/usr/bin/env python3
"""Tables, tests and figures of one campaign scenario (protocol v1, sections 2 and 6).

Input: the episode CSV of ``campaign_episodes.py collect --prefix B-`` (or C-). Output in
--out: one stats JSON per metric (campaign_stats / stats.compare), summary.csv (median
[min-max] per method x workload x metric), report.md (English) and two figures.

  B (main comparison): Full vs Static, Threshold, Forecast-only, NDT without feedback. The
     confirmatory test is M2 = overload_time_s on the 50 workload x seed pairs, Wilcoxon
     signed-rank (zero pairs dropped), Holm over the 4 baselines; every other metric and
     every per-workload number is descriptive (k/10 seeds won, ties are not wins).
  B2 (2C20S): Full vs Static and Threshold, descriptive (it confirms the trend of B).
  C (ablation): Full vs each ablation, all descriptive (protocol 6).

    python3 scripts/analysis/campaign_report.py --scenario B \
        --episodes results/campaign/episodes_B.csv --out results/campaign/B
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation.stats import Episode, compare  # noqa: E402

PRIMARY = "overload_time_s"
# metric -> (label, lower is better); all from campaign.episode_row
METRICS: dict[str, tuple[str, bool]] = {
    "overload_time_s": ("M2 overload time (s)", True),
    "overload_exposure_us": ("Overload exposure (U-s above 1)", True),
    "max_u": ("Peak U", True),
    "mean_imbalance": ("M3 mean imbalance (std of U)", True),
    "migrations": ("Migrations executed", True),
    "unsafe_migrations": ("Unsafe migrations (U_dst post > 1)", True),
    "ping_pongs": ("Ping-pong migrations", True),
    "decision_p95_ms": ("M8 decision time P95 (ms)", True),
}
SCENARIOS: dict[str, dict[str, Any]] = {
    "B": {"full": "full", "others": ["static", "threshold", "forecast_only", "ndt_no_feedback"],
          "confirmatory": True},
    # 2C20S confirms the trend of B (protocol 4.B): descriptive, no confirmatory test
    "B2": {"full": "full", "others": ["static", "threshold"], "confirmatory": False},
    "C": {"full": "full", "others": ["full_analytical", "full_zero_margin", "full_random_rank",
                                     "full_no_cooldown"], "confirmatory": False},
}
LABELS = {"full": "Full", "static": "Static", "threshold": "Threshold",
          "forecast_only": "Forecast-only", "ndt_no_feedback": "NDT no feedback",
          "full_analytical": "No learned outcome", "full_zero_margin": "Zero margin",
          "full_random_rank": "Random ranking", "full_no_cooldown": "No cooldown"}
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot_switch")
# print-safe: baselines in grays, Full in the one accent (identity is on the axis labels)
GRAY, ACCENT = "#7f7f7f", "#1f4e79"
STYLE = {"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight"}


def _value(cell: str | None) -> float | None:
    text = (cell or "").strip()
    if text in ("", "None", "nan"):
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def episodes(rows: Sequence[Mapping[str, str]], metric: str) -> list[Episode]:
    return [Episode(r["method"], r["workload"], int(r["seed"]), _value(r.get(metric)))
            for r in rows]


def check_complete(rows: Sequence[Mapping[str, str]], methods: Sequence[str]) -> list[str]:
    """Missing (method, workload, seed) cells, so a partial campaign is never read as whole."""
    seeds = sorted({int(r["seed"]) for r in rows})
    have = {(r["method"], r["workload"], int(r["seed"])) for r in rows}
    return [f"{m}/{w}/{s}" for m in methods for w in WORKLOADS for s in seeds
            if (m, w, s) not in have]


def summary(rows: Sequence[Mapping[str, str]], methods: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for metric in METRICS:
        for m in methods:
            for w in WORKLOADS:
                vals = [v for r in rows if r["method"] == m and r["workload"] == w
                        for v in [_value(r.get(metric))] if v is not None]
                out.append({"metric": metric, "method": m, "workload": w, "n": len(vals),
                            "median": statistics.median(vals) if vals else None,
                            "min": min(vals) if vals else None,
                            "max": max(vals) if vals else None})
    return out


def _cell(d: Mapping[str, Any]) -> str:
    if d.get("median") is None:
        return "-"
    return f"{d['median']:.3g} [{d['min']:.3g}-{d['max']:.3g}]"


def _p(p: float | None) -> str:
    return "-" if p is None else (f"{p:.1e}" if p < 1e-3 else f"{p:.4f}")


def report_md(scenario: str, results: Mapping[str, Mapping[str, Any]],
              table: Sequence[Mapping[str, Any]], missing: Sequence[str]) -> str:
    spec = SCENARIOS[scenario]
    full, others = spec["full"], spec["others"]
    lines = [f"# Scenario {scenario}: Full vs {', '.join(LABELS[o] for o in others)}", ""]
    if missing:
        lines += [f"**Incomplete campaign: {len(missing)} episodes missing** "
                  f"(first: {', '.join(missing[:5])}). Numbers below are provisional.", ""]
    kind = ("confirmatory (Wilcoxon signed-rank, zero pairs dropped, Holm over the "
            f"{len(others)} comparisons)" if spec["confirmatory"] else "descriptive only")
    prim = results[PRIMARY]
    lines += [f"## {METRICS[PRIMARY][0]}: {kind}", "",
              "Improvement = baseline - Full (positive = Full better).", "",
              "| vs | pairs | zero pairs (dropped) | median improvement | r_rb | p | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
    for o in others:
        c = prim["comparisons"][o]
        t = c["test"]
        med = c["improvement"]["median"]
        rb = t["rank_biserial"]
        lines.append(f"| {LABELS[o]} | {t['n_pairs']} | {t['n_zero']} | "
                     f"{'-' if med is None else f'{med:.3g}'} | "
                     f"{'-' if rb is None else f'{rb:.2f}'} | {_p(t['p_value'])} | "
                     f"{_p(c['p_holm']) if spec['confirmatory'] else '(descriptive)'} |")
    lines += ["", f"## Per workload, {METRICS[PRIMARY][0]}: median [min-max], seeds won by Full",
              "", "| workload | " + " | ".join(LABELS[m] for m in (full, *others)) + " |",
              "|---|" + "---|" * (len(others) + 1)]
    for w in WORKLOADS:
        cells = []
        for m in (full, *others):
            d = next(r for r in table if r["metric"] == PRIMARY and r["method"] == m
                     and r["workload"] == w)
            cell = _cell(d)
            if m != full:
                pw = prim["comparisons"][m]["per_workload"].get(w)
                if pw:
                    cell += f" (Full won {pw['seeds_won']}/{pw['seeds']}, tied {pw['seeds_tied']})"
            cells.append(cell)
        lines.append(f"| {w} | " + " | ".join(cells) + " |")
    lines += ["", "## Secondary metrics (descriptive): median of the five per-workload medians", "",
              "| metric | " + " | ".join(LABELS[m] for m in (full, *others)) + " |",
              "|---|" + "---|" * (len(others) + 1)]
    for metric, (label, _) in METRICS.items():
        cells = []
        for m in (full, *others):
            vals = [r["median"] for r in table if r["metric"] == metric and r["method"] == m
                    and r["median"] is not None]
            cells.append(f"{statistics.median(vals):.3g}" if vals else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Per-workload median [min-max] of every metric: summary.csv.",
              "Percent improvements are not reported without the absolute values above."]
    return "\n".join(lines) + "\n"


def plot_primary(rows: Sequence[Mapping[str, str]], methods: Sequence[str], path: Path) -> None:
    """One panel per workload; 10 episodes per method as dots, median as a bar."""
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(7.0, 2.4), sharey=True)
    for ax, w in zip(axes, WORKLOADS):
        for i, m in enumerate(methods):
            vals = sorted(v for r in rows if r["method"] == m and r["workload"] == w
                          for v in [_value(r.get(PRIMARY))] if v is not None)
            color = ACCENT if m == methods[0] else GRAY
            xs = [i + 0.18 * ((k % 5) - 2) / 2 for k in range(len(vals))]
            ax.scatter(xs, vals, s=9, color=color, linewidths=0, zorder=2)
            if vals:
                ax.hlines(statistics.median(vals), i - 0.32, i + 0.32, color="black",
                          linewidth=1.0, zorder=3)
        ax.set_title(w.replace("_", "-"), fontsize=8)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([LABELS[m] for m in methods], rotation=60, ha="right", fontsize=7)
        ax.grid(axis="y", linewidth=0.4, color="#dddddd", zorder=0)
    axes[0].set_ylabel(METRICS[PRIMARY][0])
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def plot_paired(result: Mapping[str, Any], rows: Sequence[Mapping[str, str]], full: str,
                others: Sequence[str], path: Path) -> None:
    """Paired improvement (baseline - Full) per workload x seed, one row per baseline."""
    table = {(r["method"], r["workload"], int(r["seed"])): _value(r.get(PRIMARY)) for r in rows}
    fig, ax = plt.subplots(figsize=(5.2, 0.55 * len(others) + 0.8))
    for i, o in enumerate(others):
        diffs = []
        for (m, w, s), v in table.items():
            b = table.get((o, w, s))
            if m == full and v is not None and b is not None:
                diffs.append(b - v)
        ys = [i + 0.25 * (((k * 7) % 11) / 10 - 0.5) for k in range(len(diffs))]
        ax.scatter(diffs, ys, s=8, color=GRAY, linewidths=0, zorder=2)
        if diffs:
            ax.vlines(statistics.median(diffs), i - 0.3, i + 0.3, color=ACCENT, linewidth=1.6,
                      zorder=3)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(others)))
    ax.set_yticklabels([LABELS[o] for o in others])
    ax.invert_yaxis()
    ax.set_xlabel(f"{METRICS[PRIMARY][0]}: baseline - Full (> 0: Full better)")
    ax.grid(axis="x", linewidth=0.4, color="#dddddd", zorder=0)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    spec = SCENARIOS[args.scenario]
    methods = [spec["full"], *spec["others"]]
    rows = [r for r in read_rows(args.episodes) if r["method"] in methods]
    if not rows:
        raise SystemExit(f"{args.episodes}: no episode of {methods}")
    missing = check_complete(rows, methods)
    args.out.mkdir(parents=True, exist_ok=True)
    plt.style.use(STYLE)
    results = {}
    for metric, (_, lower) in METRICS.items():
        results[metric] = compare(episodes(rows, metric), spec["full"], spec["others"], lower)
        (args.out / f"stats_{metric}.json").write_text(
            json.dumps({"metric": metric, "scenario": args.scenario, "missing": missing,
                        "confirmatory": spec["confirmatory"] and metric == PRIMARY,
                        **results[metric]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = summary(rows, methods)
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (args.out / "report.md").write_text(report_md(args.scenario, results, table, missing),
                                        encoding="utf-8")
    plot_primary(rows, methods, args.out / f"fig_{args.scenario}_overload_time")
    plot_paired(results[PRIMARY], rows, spec["full"], spec["others"],
                args.out / f"fig_{args.scenario}_paired_improvement")
    print(f"{len(rows)} episodes, {len(missing)} missing -> {args.out}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
