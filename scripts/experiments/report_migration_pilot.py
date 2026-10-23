#!/usr/bin/env python3
"""Day 3 pilot report of the migration-outcome collection (Week 11).

    python scripts/experiments/report_migration_pilot.py
    python scripts/experiments/report_migration_pilot.py data/experiment_runs/migration-pilot \
        --out-json /tmp/pilot_report.json --out-md /tmp/pilot_report.md

Runs the validator, the Day 3 bug checks and the time estimate; prints the
Markdown summary. Exit code 1 when the Day 3 gate does not pass.
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
from src.experiments.migration.pilot import (  # noqa: E402
    build_pilot_report,
    load_pilot_settings,
    render_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", type=Path,
                        help="pilot root (default: pilot_root of --pilot-config)")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/migration_outcome.yaml"))
    parser.add_argument("--pilot-config", type=Path,
                        default=Path("configs/experiments/migration_pilot.yaml"))
    parser.add_argument("--out-json", type=Path, help="full report (with one row per attempt)")
    parser.add_argument("--out-md", type=Path, help="the Markdown summary")
    args = parser.parse_args()
    settings = load_pilot_settings(args.pilot_config)
    root = args.root or Path(settings.pilot_root)
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    report = build_pilot_report(root, load_outcome_config(args.config), settings)
    text = render_markdown(report)
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
