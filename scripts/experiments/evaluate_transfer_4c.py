#!/usr/bin/env python3
"""Week 15 Day 5: apply configs/experiments/transfer_4c_v1.yaml -> evidence/week15/transfer_4c.json.

Inputs are the 4C predictions of the FROZEN 2C models (nothing is refit here):
  --forecast-rows JSONL: workload_type, controller_id, utilization, future_utilization,
                         prediction (forecast-models-v3 deployed model, U on capacity-v5)
  --outcome-rows  JSONL: workload, u_dst_post, pred_analytical, pred_direct_xgb,
                         pred_hybrid_xgb (post-migration-v1)
Margins come from the artifacts named in the config (error_margin.json, uncertainty_margin.json).
Producing the two JSONL files needs the 4C runs of Day 5; until then this script only runs on
the unit-test fixtures. Usage, from the repo root:
    python3 scripts/experiments/evaluate_transfer_4c.py --forecast-rows F --outcome-rows O
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation import transfer_4c as t4c  # noqa: E402

CONFIG = Path("configs/experiments/transfer_4c_v1.yaml")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def run(config_path: Path, forecast_rows: Path, outcome_rows: Path | None,
        root: Path = ROOT) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rules = t4c.Rules.from_config(config)
    margins = read_json(root / config["forecast"]["models"] / "error_margin.json")
    forecast = t4c.forecast_transfer(read_jsonl(forecast_rows), margins,
                                     config["forecast"]["workloads"])
    outcome = None
    if outcome_rows is not None:
        m_u = float(read_json(root / config["outcome"]["models"]
                              / "uncertainty_margin.json")["m_U"])
        outcome = t4c.outcome_transfer(read_jsonl(outcome_rows), m_u,
                                       config["outcome"]["variants"])
    return {"schema_version": "1.0", "config": str(config_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "inputs": {"forecast_rows": str(forecast_rows),
                       "outcome_rows": None if outcome_rows is None else str(outcome_rows)},
            "forecast": forecast, "outcome": outcome,
            "decision": t4c.decide(forecast, outcome, rules)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=ROOT / CONFIG)
    ap.add_argument("--forecast-rows", type=Path, required=True)
    ap.add_argument("--outcome-rows", type=Path, default=None,
                    help="Day 6 attempts; omit to score the forecast part only")
    ap.add_argument("--out", type=Path, default=None, help="default: output in the config")
    args = ap.parse_args(argv)
    report = run(args.config, args.forecast_rows, args.outcome_rows)
    out = args.out or ROOT / yaml.safe_load(args.config.read_text(encoding="utf-8"))["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    d = report["decision"]
    print(f"forecast: macro ratio {report['forecast']['macro_rmse_ratio_seen']:.3f}, "
          f"coverage {report['forecast']['coverage_seen']:.3f} -> reuse={d['forecast_reuse']}")
    if report["outcome"] is None:
        print("outcome: pending (no --outcome-rows)")
    else:
        print(f"outcome: coverage {report['outcome']['coverage_seen']['coverage']:.3f} -> "
              f"reuse={d['outcome_reuse']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
