"""Week 13 figures (My, Day 7): ranking stability and closed-loop decision latency.

Inputs are the JSON files of this week (missing ones are skipped and reported):
  evidence/week13/ranking_sensitivity.json   -> fig_w13_ranking_top1.png
  evidence/week14/closed_loop_2c_analysis.json -> fig_w14_decision_latency.png
Flat academic style (no gradient, no shadow), same rcParams as plot_outcome_figures.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("forecast", "risk", "candidate", "outcome", "safety", "ranking", "revalidation")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "savefig.dpi": 200, "savefig.bbox": "tight"})


def ranking_figure(report: dict[str, Any], out: Path) -> None:
    names = [n for n in report["policies"] if n != "base"]
    agree = [report["policies"][n]["top1_agreement"] or 0.0 for n in names]
    taus = [report["policies"][n]["kendall_tau_mean"] or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    x = range(len(names))
    ax.bar([i - 0.18 for i in x], agree, width=0.36, color="#4C72B0", label="top-1 = base")
    ax.bar([i + 0.18 for i in x], taus, width=0.36, color="#BBBBBB", label="Kendall tau")
    ax.set_xticks(list(x), names)
    ax.set_ylim(-0.05 if min(taus, default=0) >= 0 else -1.05, 1.05)
    ax.set_ylabel("vs base weights")
    n = report["policies"][names[0]]["epochs_compared"] if names else 0
    ax.set_title(f"Ranking stability ({n} epochs with >= 2 safe candidates)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def latency_figure(report: dict[str, Any], out: Path) -> None:
    lat = report["latency_ms"]
    stages = [s for s in STAGES if lat.get(s, {}).get("n", 0)]
    p50 = [lat[s]["p50"] for s in stages]
    p95 = [lat[s]["p95"] for s in stages]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    y = range(len(stages))
    ax.barh([i + 0.18 for i in y], p50, height=0.36, color="#4C72B0", label="p50")
    ax.barh([i - 0.18 for i in y], p95, height=0.36, color="#BBBBBB", label="p95")
    ax.set_yticks(list(y), stages)
    ax.invert_yaxis()
    ax.set_xlabel("ms")
    total = lat["decision_total"]
    ax.set_title(f"Decision latency, {report['latency_source']} "
                 f"(total p95 {total.get('p95', float('nan')):.1f} ms, n={total.get('n', 0)})")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    jobs = (
        ("evidence/week13/ranking_sensitivity.json", "evidence/week13/fig_w13_ranking_top1.png",
         ranking_figure),
        ("evidence/week14/closed_loop_2c_analysis.json",
         "evidence/week14/fig_w14_decision_latency.png", latency_figure),
    )
    for src, dst, draw in jobs:
        path = args.root / src
        if not path.exists():
            print(f"SKIP {src} (not there yet)")
            continue
        out = args.root / dst
        out.parent.mkdir(parents=True, exist_ok=True)
        draw(json.loads(path.read_text(encoding="utf-8")), out)
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
