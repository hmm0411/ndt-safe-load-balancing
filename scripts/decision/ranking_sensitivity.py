"""Ranking sensitivity on recorded pre-migration snapshots (Week 13, Day 5; runs on the VM).

Usage (from the repo root, PYTHONPATH=$PWD):
  python3 scripts/decision/ranking_sensitivity.py \
      --runs data/experiment_runs/safety-stress-v1 data/experiment_runs/<calibration-set>

Each attempt directory gives one decision epoch: its before_samples.jsonl fills the
Outcome Model history, the last valid snapshot is the pinned state. Never point this at
the outcome TEST split. Writes evidence/week13/ranking_sensitivity.json and a CSV of
the top-1 choice of every epoch under every weight set.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.twin.decision.ranking_policy import load_ranking_policy  # noqa: E402
from src.twin.decision.ranking_sensitivity import Epoch, rank_epoch, summarize  # noqa: E402
from src.twin.decision.safety_gate import SafetyGate  # noqa: E402
from src.twin.decision.safety_policy import load_safety_policy  # noqa: E402
from src.twin.outcome.artifact import capacity_version  # noqa: E402
from src.twin.outcome.history import OutcomeHistory  # noqa: E402
from src.twin.outcome.predictor import PostMigrationOutcomePredictor  # noqa: E402

NOW = datetime(2026, 11, 12, tzinfo=timezone.utc)  # fixed: a replay has no clock


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows.append(item.get("payload", item) if isinstance(item, dict) else item)
    return [r for r in rows if isinstance(r, dict) and (r.get("quality") or {}).get("valid")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", type=Path, required=True,
                    help="collection roots holding attempt directories")
    ap.add_argument("--artifact", type=Path,
                    default=ROOT / "data/models/outcome/post-migration-v1")
    ap.add_argument("--capacity", type=Path,
                    default=ROOT / "data/benchmarks/controller_capacity.json")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/week13/ranking_sensitivity.json")
    args = ap.parse_args(argv)

    ranking = load_ranking_policy()
    policies = {name: ranking.with_weights(name, w) for name, w in ranking.sensitivity}
    gate = SafetyGate(load_safety_policy())
    predictor = PostMigrationOutcomePredictor(args.artifact, capacity_version(args.capacity))
    epochs: list[Epoch] = []
    skipped: dict[str, int] = {"no_before_samples": 0, "short_history": 0}
    for root in args.runs:
        if "test" in root.name:
            raise SystemExit(f"refusing {root}: the test split is never used to tune ranking")
        for attempt in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))):
            path = attempt / "before_samples.jsonl"
            if not path.exists():
                skipped["no_before_samples"] += 1
                continue
            rows = load_rows(path)
            predictor.history = OutcomeHistory()
            for row in rows:
                predictor.observe(row)
            if not rows or not predictor.history.ready():
                skipped["short_history"] += 1
                continue
            epochs.append(rank_epoch(f"{root.name}/{attempt.name}", rows[-1], predictor,
                                     gate, policies, NOW))
    report = {
        "ranking_policy": ranking.version,
        "safety_policy": gate.policy.version,
        "outcome_artifact": str(args.artifact.relative_to(ROOT)) if args.artifact.is_relative_to(ROOT) else str(args.artifact),
        "runs": [str(r) for r in args.runs],
        "skipped": skipped,
        "note": "stability only: one observed outcome per attempt, so no regret is computed",
        **summarize(epochs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "source", "n_candidates", "n_safe", "policy", "status", "selected"])
        for e in epochs:
            for name, r in e.results.items():
                writer.writerow([e.epoch_id, e.source, e.n_candidates, e.n_safe, name,
                                 r.status, r.selected_candidate_id or ""])
    print(json.dumps({k: report[k] for k in ("n_epochs", "base_status", "policies")}, indent=1))
    print(f"wrote {args.out} and {args.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
