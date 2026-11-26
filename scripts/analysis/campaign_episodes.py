#!/usr/bin/env python3
"""Closed-loop campaign: the episode plan, and one metrics row per finished episode.

  python3 scripts/analysis/campaign_episodes.py plan --scenario B --out results/campaign/plan_B.csv
  python3 scripts/analysis/campaign_episodes.py collect --runs <campaign dir> \
      --out results/campaign/episodes_4c.csv

Layout of a finished episode (written by the VM runner):
  <runs>/<episode_id>/meta.json        {"episode_id", "method", "workload", "seed", "order",
                                        "workload_started_at": ISO time}
  <runs>/<episode_id>/snapshots.jsonl  the orchestrator's snapshots of the episode
  <runs>/<episode_id>/decisions/<decision_id>/record.json
The CSV feeds scripts/analysis/campaign_stats.py (one column per metric).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation import campaign as cp  # noqa: E402
from src.experiments.final_evaluation.faults import plan_fault_episodes  # noqa: E402

CONFIG = ROOT / "configs/experiments/campaign_4c_v1.yaml"


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: not a mapping")
    return data


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(k for row in rows for k in row))  # D rows add columns
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.scenario == "D":  # fault scenarios: their own grid (configs ``faults``)
        plan = plan_fault_episodes(cfg["faults"], int(cfg["faults"]["order_seed"]))
        write_csv(args.out, [vars(e) for e in plan])
        print(f"{len(plan)} fault episodes (D1-D6 Full, D7 levels x methods) -> {args.out}")
        return 0
    methods = cfg["scenarios"][args.scenario]
    plan = cp.plan_episodes(methods, cfg["workloads"], cfg["seeds"],
                            int(cfg["order_seed"]) + ord(args.scenario), f"{args.scenario}")
    write_csv(args.out, [vars(e) for e in plan])
    print(f"{len(plan)} episodes ({len(methods)} methods x {len(cfg['workloads'])} workloads"
          f" x {len(cfg['seeds'])} seeds) -> {args.out}")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    capacity_file = ROOT / str(cfg["capacity"])
    capacity = {str(k): float(v["c_safe_pps"]) for k, v in
                json.loads(capacity_file.read_text(encoding="utf-8"))["controllers"].items()}
    rows, problems = [], []
    for meta_path in sorted(args.runs.glob(f"{args.prefix}*/meta.json")):
        run = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status", "OK") != "OK":
            problems.append(f"{run.name}: episode status {meta.get('status')} (not collected)")
            continue
        ep = cp.PlannedEpisode(int(meta["order"]), str(meta["episode_id"]), str(meta["method"]),
                               str(meta["workload"]), int(meta["seed"]))
        try:
            records = [json.loads(p.read_text(encoding="utf-8"))
                       for p in sorted(run.glob("decisions/*/record.json"))]
            row = cp.episode_row(ep, cp.read_jsonl(run / "snapshots.jsonl"), records,
                                 capacity, cp._time(meta["workload_started_at"]),
                                 float(cfg["warmup_s"]))
            if meta.get("fault_check") is not None:  # D episodes
                row["fault_pass"] = meta["fault_check"].get("pass")
                row["fault_check"] = json.dumps(meta["fault_check"], sort_keys=True)
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{run.name}: {exc}")
    if not rows:
        raise SystemExit("no finished episode under " + str(args.runs))
    write_csv(args.out, sorted(rows, key=lambda r: r["order"]))
    print(f"{len(rows)} episodes -> {args.out}")
    for line in problems:
        print("!! " + line)
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", type=Path, default=CONFIG)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scenario", required=True)
    p.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--runs", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--prefix", default="", help="episode ids starting with this (B-, C-, D-)")
    args = ap.parse_args(argv)
    return cmd_plan(args) if args.cmd == "plan" else cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
