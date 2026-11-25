#!/usr/bin/env python3
"""Day 5-6: 4C probe-off outcome attempts -> outcome rows JSONL with the frozen 2C predictions.

    python3 scripts/experiments/build_transfer_outcome_rows.py \
        --raw-root ~/w15-day34/data/experiment_runs/migration-outcome-4c-noprobe-v1 \
        --capacity ~/w15-day34/data/benchmarks/controller_capacity_v5_noprobe.json
Then evaluate_transfer_4c.py --forecast-rows <...> --outcome-rows <out>.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation import transfer_outcome_rows as tor  # noqa: E402
from src.twin.outcome.dataset import load_dataset_config  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/experiments/transfer_4c_v1.yaml")
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "configs/outcome/dataset.yaml")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--capacity", type=Path, required=True)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evidence/week15/transfer_4c_outcome_rows.jsonl")
    args = ap.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    levels = [float(x) for x in config["outcome"]["destination_load_levels"]]
    rows, excluded = tor.collect(args.raw_root.expanduser(), args.capacity.expanduser(),
                                 load_dataset_config(args.dataset_config), levels)
    out_rows = tor.outcome_rows(rows, tor.frozen_predictors(ROOT / config["outcome"]["models"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    reasons = Counter(r.split(":")[0] for r in excluded.values())
    print(f"{len(out_rows)} MIGRATE rows at levels {levels} -> {args.out}")
    print(f"excluded over the whole collection: {dict(reasons)}")
    print("by workload:", dict(Counter(r["workload"] for r in out_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
