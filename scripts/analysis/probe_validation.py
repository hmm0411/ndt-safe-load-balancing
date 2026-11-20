#!/usr/bin/env python3
"""Week 15 Day 3: judge the flow-setup probe and draw latency vs load per controller.

  python3 scripts/analysis/probe_validation.py

Reads  evidence/week15/probe_sweep_summary.json  (scripts/experiments/export_4c40s_sweeps.py)
       configs/experiments/probe_acceptance_v1.yaml (criteria fixed before the sweep was read)
Writes evidence/week15/probe_validation.json and evidence/week15/figures/probe_latency_vs_u.png
Exit code 0 when every criterion passes (Huong 3), 1 otherwise (Huong 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.experiments.capacity.probe_acceptance import CONTROLLERS, judge
from src.experiments.common.config import load_yaml


def draw(summary: dict[str, Any], scale: float, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(CONTROLLERS), figsize=(11, 2.8), sharey=True)
    for ax, target in zip(axes, CONTROLLERS):
        by_rate = summary["curves"].get(target, {})
        rates = sorted(by_rate, key=float)
        for cid in CONTROLLERS:
            xs, ys = [], []
            for rate in rates:
                cell = by_rate[rate].get(cid) or {}
                pps, p95 = cell.get("processed_pps_mean"), (cell.get("setup_p95_ms") or {})
                if pps is not None and p95.get("mean") is not None:
                    xs.append(pps / scale)
                    ys.append(p95["mean"])
            style = {"color": "#1f4e79", "lw": 2, "marker": "o"} if cid == target else \
                {"color": "#9e9e9e", "lw": 1, "marker": ".", "alpha": 0.8}
            ax.plot(xs, ys, label=cid, **style)
        ax.set_title(f"target {target}", fontsize=9)
        ax.set_xlabel("U (processed pps / scale)", fontsize=8)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("flow-setup P95 (ms)", fontsize=8)
    fig.suptitle("Flow-setup latency vs load: loaded controller (blue) and the other three",
                 fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("evidence/week15/probe_sweep_summary.json"))
    parser.add_argument("--rules", type=Path,
                        default=Path("configs/experiments/probe_acceptance_v1.yaml"))
    parser.add_argument("--out", type=Path, default=Path("evidence/week15/probe_validation.json"))
    parser.add_argument("--figure", type=Path,
                        default=Path("evidence/week15/figures/probe_latency_vs_u.png"))
    args = parser.parse_args(argv)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = judge(summary, load_yaml(args.rules))
    result["summary_file"] = str(args.summary)
    result["scale_pps"] = summary.get("scale_pps")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    draw(summary, float(summary["scale_pps"]), args.figure)
    for name, criterion in result["criteria"].items():
        failed = [row for row in criterion["rows"] if not row["pass"]]
        print(f"{name}: {'PASS' if criterion['pass'] else 'FAIL'}"
              + (f" ({len(failed)} failing rows, first: {failed[0]})" if failed else ""))
    print(f"decision: {result['decision']}")
    print(f"wrote {args.out} and {args.figure}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
