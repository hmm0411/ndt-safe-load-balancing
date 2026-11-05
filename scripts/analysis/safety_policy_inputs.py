#!/usr/bin/env python3
"""Evidence behind every number in configs/decision/safety_policy_v1.yaml (Week 13 Day 1).

Reads only files in the repo (capacity artifact, outcome margins, raw-v2 attempt
summary) and writes evidence/week13/safety_policy_inputs.json.

    python3 scripts/analysis/safety_policy_inputs.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from src.twin.decision.safety_policy import (
    POLICY_PATH, cooldown_lower_bound, load_margins, load_safety_policy,
)
from src.twin.forecasting.margin import finite_sample_quantile

CAPACITY = Path("data/benchmarks/controller_capacity.json")
ATTEMPTS = Path("evidence/week12/attempt_summary.csv")
OUT = Path("evidence/week13/safety_policy_inputs.json")


def committed(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows
            if r["action_type"] == "MIGRATE" and r["transaction_status"] == "COMMITTED"]


def inputs(capacity: dict[str, Any], rows: list[dict[str, str]], policy_path: Path) -> dict:
    policy = load_safety_policy(policy_path)
    m_u, m_r = load_margins(policy)
    done = committed(rows)
    tx = [float(r["transaction_s"]) for r in done]
    settle = [float(r["settle_s"]) for r in done]
    p95 = cooldown_lower_bound(tx, settle)
    rose = [r for r in done if float(r["src_u_after_mean"]) > float(r["src_u_before_mean"])]
    val = [r for r in done if r["split"] == "validation"]
    ctrl = capacity["controllers"]
    return {
        "policy_version": policy.version,
        "utilization": {"capacity_file": str(CAPACITY),
                        "c_safe_pps": {c: v["c_safe_pps"] for c, v in ctrl.items()}},
        "response": {"c_response_pps": {c: v["c_response_pps"] for c, v in ctrl.items()},
                     "response_limits_capacity": any(v["c_response_pps"] for v in ctrl.values())},
        "margins": {"file": policy.margins_file, "m_U": m_u, "m_R": m_r},
        "cooldown": {"source": str(ATTEMPTS), "n": len(done), "level": 0.95,
                     "transaction_plus_settle_p95_s": p95, "value_s": math.ceil(p95),
                     "transaction_p95_s": finite_sample_quantile(tx, 0.95),
                     "settle_p95_s": finite_sample_quantile(settle, 0.95),
                     "max_s": max(a + b for a, b in zip(tx, settle))},
        "source_relief": {
            "n_committed": len(done), "source_rose_after": len(rose),
            "validation_n": len(val),
            "validation_source_rose_after": sum(1 for r in val if r in rose),
            "min_switch_share": min(float(r["switch_share_before_mean"]) for r in done)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    capacity = json.loads(CAPACITY.read_text(encoding="utf-8"))
    with ATTEMPTS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = inputs(capacity, rows, args.policy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(data["cooldown"], sort_keys=True))
    print(json.dumps(data["source_relief"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
