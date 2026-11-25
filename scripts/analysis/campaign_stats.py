#!/usr/bin/env python3
"""Full vs the baselines on one campaign metric (evaluation protocol v1, section 6).

Input: one CSV row per episode with columns ``method``, ``workload``, ``seed`` and one column
per metric (an empty cell = the metric could not be computed, never 0). Output: JSON with the
Wilcoxon test per baseline (Holm-adjusted), the rank-biserial effect size and, per workload,
median [min - max] and "k/n seeds won"; a short table on stdout.

    python3 scripts/analysis/campaign_stats.py --episodes results/campaign/episodes_4c.csv \
        --metric overload_seconds --lower-is-better --out results/campaign/stats_m2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation.stats import (  # noqa: E402
    ZERO_METHODS,
    Episode,
    compare,
)

BASELINES = ("Static", "Threshold", "Forecast-only", "NDT-no-feedback")


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "-"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


def read_episodes(path: Path, metric: str) -> list[Episode]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        need = {"method", "workload", "seed", metric}
        if not need <= set(reader.fieldnames or ()):
            raise SystemExit(f"{path}: needs columns {sorted(need)}")
        out = []
        for row in reader:
            cell = (row[metric] or "").strip()
            out.append(Episode(row["method"], row["workload"], int(row["seed"]),
                               float(cell) if cell not in ("", "None", "nan") else None))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--metric", required=True)
    side = ap.add_mutually_exclusive_group(required=True)
    side.add_argument("--lower-is-better", action="store_true")
    side.add_argument("--higher-is-better", action="store_true")
    ap.add_argument("--full", default="Full")
    ap.add_argument("--baselines", nargs="+", default=list(BASELINES))
    ap.add_argument("--zero-method", choices=ZERO_METHODS, default="wilcox")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        result = compare(read_episodes(args.episodes, args.metric), args.full, args.baselines,
                         args.lower_is_better, args.zero_method, args.alpha)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    result = {"metric": args.metric, "episodes": str(args.episodes), **result}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.metric}: {args.full} vs baseline (improvement > 0 = {args.full} better)")
    print(f"{'baseline':<18}{'pairs':>6}{'zero':>6}{'median impr':>13}{'r_rb':>7}"
          f"{'p':>9}{'p_Holm':>9}")
    for base, c in result["comparisons"].items():
        t = c["test"]
        p = _fmt_p(t["p_value"])
        ph = _fmt_p(c["p_holm"])
        rb = "-" if t["rank_biserial"] is None else f"{t['rank_biserial']:.2f}"
        med = c["improvement"]["median"]
        print(f"{base:<18}{t['n_pairs']:>6}{t['n_zero']:>6}"
              f"{'-' if med is None else f'{med:.4g}':>13}{rb:>7}{p:>9}{ph:>9}")
        for w, s in c["per_workload"].items():
            print(f"    {w:<14} won {s['seeds_won']}/{s['seeds']} (tied {s['seeds_tied']}), "
                  f"median {args.full} {s[args.full]['median']:.4g} "
                  f"[{s[args.full]['min']:.4g}-{s[args.full]['max']:.4g}] vs "
                  f"{s[base]['median']:.4g} [{s[base]['min']:.4g}-{s[base]['max']:.4g}]")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
