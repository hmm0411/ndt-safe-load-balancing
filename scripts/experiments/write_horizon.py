#!/usr/bin/env python3
"""Horizon feasibility check (Week 10, V3): T_detect + T_decision + T_migrate <= h.

T_detect  = P95 telemetry collection latency (collection benchmark)
T_migrate = P95 switch-migration latency (migration benchmark)
T_decision = forecast inference p95 (forecast_inference_latency.json)
             + the rest of the decision path, measured on the running
               orchestrator (--decision-overhead-ms; planner comes in Week 11)

Writes data/benchmarks/horizon_v1.json. feasible = false is a finding to
report to the team, not something to fix by changing h silently.

    python scripts/experiments/write_horizon.py --decision-overhead-ms 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.twin.forecasting.versions import FORECAST_CONFIG

# P95 values measured on kltn-sdn (e2-standard-8). Pass the numbers from the
# benchmark reports explicitly if they are re-measured.
DEFAULT_T_DETECT_MS = 568.579
DEFAULT_T_MIGRATE_MS = 2132.906


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--t-detect-ms", type=float, default=DEFAULT_T_DETECT_MS)
    parser.add_argument("--t-migrate-ms", type=float, default=DEFAULT_T_MIGRATE_MS)
    parser.add_argument("--latency", type=Path,
                        default=Path("data/benchmarks/forecast_inference_latency.json"))
    parser.add_argument("--decision-overhead-ms", type=float, required=True,
                        help="measured p95 of snapshot -> forecast stored, minus inference")
    parser.add_argument("--output", type=Path, default=Path("data/benchmarks/horizon_v1.json"))
    args = parser.parse_args()

    config = yaml.safe_load(FORECAST_CONFIG.read_text(encoding="utf-8")) or {}
    horizon_s = float(config["horizon"]["provisional"]["effective_horizon_seconds"])
    latency = json.loads(args.latency.read_text(encoding="utf-8"))
    inference_p95 = float(latency["selected_p95_ms"])

    budget = horizon_s * 1000.0 - args.t_detect_ms - args.t_migrate_ms
    t_decision = inference_p95 + args.decision_overhead_ms
    result = {
        "schema_version": "1.0",
        "horizon_seconds": horizon_s,
        "t_detect_ms": args.t_detect_ms,
        "t_migrate_ms": args.t_migrate_ms,
        "t_decision_budget_ms": round(budget, 3),
        "inference_p95_ms": inference_p95,
        "inference_source": str(args.latency),
        "inference_measured_at": latency.get("environment", {}).get("measured_at"),
        "decision_overhead_ms": args.decision_overhead_ms,
        "t_decision_ms": round(t_decision, 3),
        "slack_ms": round(budget - t_decision, 3),
        "feasible": bool(t_decision <= budget),
        "planner_included": False,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for key in ("horizon_seconds", "t_decision_budget_ms", "t_decision_ms", "slack_ms", "feasible"):
        print(f"{key}: {result[key]}")
    print("Wrote", args.output)
    return 0 if result["feasible"] else 1


if __name__ == "__main__":
    sys.exit(main())
