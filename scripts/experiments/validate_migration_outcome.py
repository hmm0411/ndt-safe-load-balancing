#!/usr/bin/env python3
"""Validate a migration-outcome dataset root (Week 11).

    python scripts/experiments/validate_migration_outcome.py data/experiment_runs/migration-pilot
    python scripts/experiments/validate_migration_outcome.py data/experiment_runs/migration-outcome-v1 \
        --config configs/experiments/migration_outcome.yaml --out /tmp/w11-check.json

Exit code 1 when any attempt is invalid (``valid = false``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.migration.config import load_outcome_config  # noqa: E402
from src.experiments.migration.validator import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="directory holding attempt directories")
    parser.add_argument("--config", type=Path, help="matrix config, to list missing attempts")
    parser.add_argument("--out", type=Path, help="also write the summary JSON here")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")
    config = load_outcome_config(args.config) if args.config else None
    summary = validate_dataset(args.root, config)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    short = dict(summary)
    for key in ("missing", "unplanned"):
        if isinstance(short.get(key), list) and len(short[key]) > 10:
            short[key] = short[key][:10] + [f"... {len(summary[key]) - 10} more (see --out)"]
    print(json.dumps(short, indent=2, ensure_ascii=False))
    return 1 if summary["invalid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
