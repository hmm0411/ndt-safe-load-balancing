#!/usr/bin/env python3
"""Run scripted migration attempts (Week 11). Run as the normal user, not root.

  python3 scripts/experiments/run_migration_outcome.py --dry-run
  python3 scripts/experiments/run_migration_outcome.py \
      --only stable:early:dst45:c1-c2:r01 --output-root data/experiment_runs/migration-smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.migration.config import (  # noqa: E402
    dst_tag,
    expand_attempts,
    load_outcome_config,
)
from src.experiments.migration.runner import (  # noqa: E402
    RunnerSettings,
    attempt_dir_name,
    run_attempt,
)

DEFAULT_CONFIG = "configs/experiments/migration_outcome.yaml"
DEFAULT_WORK_ROOT = "data/experiment_runs/migration-work"


def cfg(config, *names):
    """First attribute of the config that exists (the field names are My's)."""
    for name in names:
        if hasattr(config, name):
            return getattr(config, name)
    raise SystemExit(f"OutcomeConfig has none of {names}: adjust cfg() calls in this script")


def parse_only(spec: str):
    parts = spec.split(":")
    if len(parts) != 5 or not parts[2].startswith("dst") or "-" not in parts[3]:
        raise SystemExit(f"--only expects workload:phase:dstNN:src-dst:rNN, got {spec!r}")
    workload, phase, dst, direction, repeat = parts
    source, target = direction.split("-", 1)
    return workload, phase, dst, source, target, repeat


def matches(plan, key) -> bool:
    workload, phase, dst, source, target, repeat = key
    return (plan.workload, plan.phase, dst_tag(plan.destination_load_level),
            plan.source_controller, plan.target_controller, plan.repeat) == (
        workload, phase, dst, source, target, repeat)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--only", action="append", default=[], metavar="W:PHASE:dstNN:SRC-DST:rNN",
                   help="run only this matrix cell (repeatable)")
    p.add_argument("--action", choices=("MIGRATE", "NO_OP"), default=None)
    p.add_argument("--simulate-failure", choices=("none", "flow_mod"), default="none")
    p.add_argument("--output-root", default=None)
    p.add_argument("--work-root", default=DEFAULT_WORK_ROOT)
    p.add_argument("--duration", type=float, default=None, help="workload seconds (default per workload)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true", help="continue after a failed attempt")
    p.add_argument("--orchestrator-url", default="http://127.0.0.1:9000")
    p.add_argument("--capacity-json", default="data/benchmarks/controller_capacity.json")
    p.add_argument("--capacity-runs", default="data/benchmarks/capacity_runs.csv")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if os.geteuid() == 0 and not args.dry_run:
        print("run this as the normal user (the runner calls sudo itself)", file=sys.stderr)
        return 2
    os.chdir(REPO_ROOT)
    config_path = Path(args.config)
    config = load_outcome_config(config_path)
    plans = expand_attempts(config)
    if args.only:
        keys = [parse_only(s) for s in args.only]
        plans = [p for p in plans if any(matches(p, k) for k in keys)]
        if len(plans) < len(keys):
            print(f"warning: {len(keys)} --only given, {len(plans)} matched the matrix", file=sys.stderr)
    if args.action:
        plans = [p for p in plans if p.action_type == args.action]
    if args.limit is not None:
        plans = plans[: args.limit]
    if not plans:
        print("no attempt selected", file=sys.stderr)
        return 2
    if args.simulate_failure != "none" and any(p.action_type != "MIGRATE" for p in plans):
        print("--simulate-failure needs MIGRATE attempts only (use --action MIGRATE)", file=sys.stderr)
        return 2

    output_root = Path(args.output_root or cfg(config, "output_root"))
    settings = RunnerSettings(
        repo_root=REPO_ROOT,
        output_root=output_root if output_root.is_absolute() else REPO_ROOT / output_root,
        work_root=(Path(args.work_root) if Path(args.work_root).is_absolute() else REPO_ROOT / args.work_root),
        config_path=config_path if config_path.is_absolute() else REPO_ROOT / config_path,
        controllers_config=str(cfg(config, "controllers_config")),
        capacity_json=REPO_ROOT / args.capacity_json,
        capacity_runs=REPO_ROOT / args.capacity_runs,
        topology=str(cfg(config, "topology")),
        orchestrator_url=args.orchestrator_url,
        required_window=int(cfg(config, "pre_window_valid_snapshots", "pre_window")),
        post_window_timeout_s=float(cfg(config, "post_window_timeout_s")),
    )

    print(f"{len(plans)} attempt(s) -> {settings.output_root}")
    if args.dry_run:
        for p in plans:
            done = (settings.output_root / attempt_dir_name(p, args.simulate_failure)).exists()
            print(f"  {'done ' if done else 'todo '} {attempt_dir_name(p, args.simulate_failure)}"
                  f"  [{p.action_type}, split={p.split}]")
        return 0

    settings.output_root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for number, plan in enumerate(plans, 1):
        print(f"--- {number}/{len(plans)} {attempt_dir_name(plan, args.simulate_failure)}")
        try:
            result = run_attempt(plan, settings, failure=args.simulate_failure, duration=args.duration)
        except Exception as exc:  # noqa: BLE001 - the attempt is quarantined already
            failures += 1
            print(f"    FAILED: {exc}", file=sys.stderr)
            if not args.keep_going:
                return 1
            continue
        if result.status == "exists":
            print("    already done, skipped")
        else:
            print(f"    {result.transaction_status or 'NO_OP'}  eligible={result.training_eligible}  "
                  f"{result.seconds:.0f}s  {result.note}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
