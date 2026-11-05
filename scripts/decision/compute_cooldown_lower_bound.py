"""Cooldown lower bound = P95(T_migration) + P95(T_post_observation).

Both terms come from committed raw-v2 evidence (COMMITTED MIGRATE attempts):
  transaction_s = orchestrator transaction started_at -> finished_at
  settle_s      = transaction finished_at -> 5th Hard-Gate-valid post snapshot
Nearest-rank P95. The result is a lower bound, not a chosen policy value.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

SOURCE = Path("evidence/week12/attempt_summary.csv")
OUTPUT = Path("evidence/week13/cooldown_lower_bound.json")


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def main() -> None:
    with open(SOURCE, encoding="utf-8", newline="") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r["action_type"] == "MIGRATE" and r["transaction_status"] == "COMMITTED"
        ]
    t_migration = p95([float(r["transaction_s"]) for r in rows])
    t_post = p95([float(r["settle_s"]) for r in rows])
    result = {
        "source": str(SOURCE),
        "n_attempts": len(rows),
        "percentile": "P95 nearest-rank",
        "t_migration_p95_s": round(t_migration, 3),
        "t_post_observation_p95_s": round(t_post, 3),
        "cooldown_lower_bound_s": round(t_migration + t_post, 3),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
