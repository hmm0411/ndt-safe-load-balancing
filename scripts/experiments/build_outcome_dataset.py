#!/usr/bin/env python3
"""Build the processed Post-Migration Outcome dataset (Week 12, Day 1).

Run on the VM from the repo root (the raw attempts live there):

    python3 scripts/experiments/build_outcome_dataset.py \
        --raw-root ~/w12-cap4/data/experiment_runs/migration-outcome-v2 \
        --split evidence/week12/split_manifest.json \
        --saturation evidence/week12/attempt_saturation.csv \
        --capacity data/benchmarks/controller_capacity.json \
        --out data/outcome/processed/migration-outcome-processed-v1

--saturation is a cross-check: flags are recomputed from the raw CPU and must
equal the committed evidence file.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.twin.outcome.dataset import (  # noqa: E402
    build_rows,
    capacity_info,
    load_dataset_config,
    load_saturation_csv,
    load_split_map,
    saturation_flags,
    write_dataset,
)


def _git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=False)
    return proc.stdout.strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/outcome/dataset.yaml"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="split_manifest.json of the raw collection")
    parser.add_argument("--capacity", type=Path,
                        default=Path("data/benchmarks/controller_capacity.json"))
    parser.add_argument("--saturation", type=Path,
                        help="attempt_saturation.csv to cross-check the recomputed flags")
    parser.add_argument("--out", type=Path,
                        default=Path("data/outcome/processed/migration-outcome-processed-v1"))
    parser.add_argument("--force", action="store_true", help="overwrite an existing build")
    args = parser.parse_args()

    try:
        config = load_dataset_config(args.config)
        capacity = capacity_info(args.capacity)
        split_map = load_split_map(args.split)
        flags = saturation_flags(args.raw_root.expanduser())
        if args.saturation is not None:
            committed = load_saturation_csv(args.saturation)
            if committed != flags:
                diff = sorted(k for k in set(committed) | set(flags)
                              if committed.get(k) != flags.get(k))
                parser.error(f"saturation flags differ from {args.saturation}: {diff[:5]}")
        rows, noop_rows, excluded = build_rows(
            args.raw_root.expanduser(), split_map, capacity["c_safe_pps"], config, flags)
        result = write_dataset(
            args.out, rows, noop_rows, excluded, config,
            provenance={
                "raw_root": str(args.raw_root),
                "split_source": str(args.split),
                "capacity_version": capacity["capacity_version"],
                "capacity_sha256": capacity["sha256"],
                "c_safe_pps": capacity["c_safe_pps"],
                "saturation_rule": "CPU p95 >= 95 % in the window "
                                   "(src/experiments/migration/saturation.py)",
                "git_commit": _git_commit(),
            },
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        parser.error(str(exc))

    report = result["report"]
    split = report["rows_by_split"]
    print(f"{report['rows']} rows (train {split['train']}, validation {split['validation']}, "
          f"test {split['test']}), NO_OP reference {report['noop_reference_rows']}, "
          f"excluded {report['excluded']} {report['excluded_by_reason']}")
    print(f"capacity {capacity['capacity_version']} c_safe_pps {capacity['c_safe_pps']}")
    print(f"saturated_before {report['saturated_before']}, saturated_after "
          f"{report['saturated_after']}, overload_after {report['overload_after']} "
          f"{report['overload_after_by_split']}")
    print(f"{report['candidate_features']} candidate features, {report['model_features']} model "
          f"features; audit_only: {json.dumps(report['audit_only'])}")
    print(f"one row per attempt: {report['one_row_per_attempt']}; leakage check OK")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
