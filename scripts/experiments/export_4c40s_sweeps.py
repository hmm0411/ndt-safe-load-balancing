#!/usr/bin/env python3
"""Week 15 Day 3-4 outputs from the 4C40S run directories (no Mininet, no sudo).

  python3 scripts/experiments/export_4c40s_sweeps.py            # CSV + Day 3 summary
  python3 scripts/experiments/export_4c40s_sweeps.py --derive   # + capacity-v5 JSON

Writes
  data/benchmarks/capacity_runs_4c40s.csv     one row per run of the four sweeps
  evidence/week15/probe_sweep_summary.json    Day 3 curves and probe-induced load (no verdict)
  data/benchmarks/controller_capacity_v5.json with --derive, by configs/experiments/
                                              capacity_v5_rules.yaml (v4 files untouched)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.experiments.capacity.export_4c40s import collect, probe_sweep_summary
from src.experiments.capacity.rules_v5 import Rules, derive
from src.experiments.common.config import load_yaml

V4_CAPACITY = Path("data/benchmarks/controller_capacity.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("data/experiment_runs"))
    parser.add_argument("--csv", type=Path, default=Path("data/benchmarks/capacity_runs_4c40s.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("evidence/week15/probe_sweep_summary.json"))
    parser.add_argument("--rules", type=Path,
                        default=Path("configs/experiments/capacity_v5_rules.yaml"))
    parser.add_argument("--capacity-out", type=Path,
                        default=Path("data/benchmarks/controller_capacity_v5.json"))
    parser.add_argument("--derive", action="store_true")
    args = parser.parse_args(argv)

    rows = collect(args.runs)
    if not rows:
        print(f"no 4C40S sweep run in {args.runs}", file=sys.stderr)
        return 1
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, list[int]] = {}
    for row in rows:
        valid_invalid = counts.setdefault(row["experiment"], [0, 0])
        valid_invalid[0 if row["valid"] else 1] += 1
    print(f"wrote {len(rows)} rows to {args.csv}: "
          + ", ".join(f"{e} {v} valid/{i} invalid" for e, (v, i) in sorted(counts.items())))

    # Until capacity-v5 exists, delta U of the probe uses the capacity-v4 scale (c1).
    scale = float(json.loads(V4_CAPACITY.read_text(encoding="utf-8"))
                  ["controllers"]["c1"]["c_safe_pps"])
    summary = probe_sweep_summary(rows, scale)
    summary["scale_source"] = f"{V4_CAPACITY} c1 c_safe_pps (capacity-v4)"
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.summary}")

    if args.derive:
        capacity = derive(rows_as_text(rows), Rules.from_config(load_yaml(args.rules)))
        args.capacity_out.write_text(json.dumps(capacity, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(capacity["method"], indent=2))
        for cid, value in capacity["controllers"].items():
            print(f"{cid}: c_safe {value['c_safe_pps']:.1f} pps at "
                  f"{value['safe_reference']['target_rate_fps']:g} fps; knees setup "
                  f"{value['setup_knee_rate_fps']} / throughput {value['throughput_knee_rate_fps']}")
        print(f"wrote {args.capacity_out}")
    return 0


def rows_as_text(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """The rules read CSV text, as for capacity_runs.csv of v4."""
    return [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]


if __name__ == "__main__":
    sys.exit(main())
