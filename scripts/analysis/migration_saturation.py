#!/usr/bin/env python3
"""Controller saturation check of a migration-outcome collection (CPU p95 >= 95 %).

Writes attempt_saturation.csv, per_attempt_cpu.csv and the CPU-by-U report into --out.
Run on the VM from the repo root, against the raw attempts:

    python3 scripts/analysis/migration_saturation.py \
        --raw-root <.../data/experiment_runs/migration-outcome-v2> --out evidence/week12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.migration.saturation import write_outputs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="directory holding one sub-directory per attempt")
    parser.add_argument("--out", type=Path, default=Path("evidence/week12"))
    parser.add_argument("--report-name", default="cpu_vs_u_v2.txt")
    args = parser.parse_args()
    try:
        summary = write_outputs(args.raw_root, args.out, args.report_name)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}/attempt_saturation.csv, per_attempt_cpu.csv, {args.report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
