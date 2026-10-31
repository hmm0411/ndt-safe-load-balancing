#!/usr/bin/env python3
"""Error of the Analytical outcome model on the train or validation split (Week 12, Day 1).

The test split is not available here: it is opened once, on Day 4, by the model
selection step. Writes one CSV row per (group, target) and prints the overall table.

    python3 scripts/experiments/evaluate_outcome_analytical.py --split validation
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome.dataset import load_dataset, load_dataset_config  # noqa: E402
from src.twin.outcome.metrics import regression_metrics, selection_score  # noqa: E402
from src.twin.outcome.targets import ALL_TARGETS  # noqa: E402

GROUP_KEYS = ("workload", "destination_load_level", "direction", "switch_share_group",
              "saturated_before", "saturated_after")
COLUMNS = ("split", "group", "value", "target", "n", "mae", "rmse", "p95_abs_error", "bias", "r2")


def evaluate(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("all", "all", list(rows))]
    for key in GROUP_KEYS:
        for value in sorted({str(r[key]) for r in rows}):
            groups.append((key, value, [r for r in rows if str(r[key]) == value]))
    out = []
    for key, value, members in groups:
        for target in ALL_TARGETS:
            m = regression_metrics([r[target] for r in members],
                                   [r[f"analytical_{target}"] for r in members])
            if key != "all":
                m["r2"] = float("nan")  # R^2 only in the overall table
            out.append({"split": split, "group": key, "value": value, "target": target, **m})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--dataset", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    rows = [r for r in load_dataset(args.dataset / "outcome_dataset.csv")
            if r["split"] == args.split]
    if not rows:
        parser.error(f"no {args.split} rows in {args.dataset}")
    results = evaluate(rows, args.split)
    out = args.out or Path("data/outcome/reports") / f"analytical_{args.split}_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows({c: r[c] for c in COLUMNS} for r in results)

    def score_with(weights: dict[str, float]) -> dict[str, float]:
        return selection_score({t: [r[t] for r in rows] for t in weights},
                               {t: [r[f"analytical_{t}"] for r in rows] for t in weights},
                               weights, config.selection_epsilon)

    score = score_with(config.effective_selection_weights)
    plan = score_with(config.selection_weights)
    print(f"Analytical on {args.split} (n = {len(rows)})")
    print(f"{'target':<14}{'MAE':>9}{'RMSE':>9}{'P95|e|':>9}{'bias':>9}{'R2':>8}")
    for r in results:
        if r["group"] == "all":
            print(f"{r['target']:<14}{r['mae']:>9.4f}{r['rmse']:>9.4f}{r['p95_abs_error']:>9.4f}"
                  f"{r['bias']:>9.4f}{r['r2']:>8.3f}")
    terms = ", ".join(f"{k} {v:.3f}" for k, v in plan.items() if k != "score")
    excluded = ", ".join(config.selection_exclude) or "none"
    print(f"selection score S = {score['score']:.4f} (without {excluded}; used to choose models)")
    print(f"plan score S_plan = {plan['score']:.4f} ({terms})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
