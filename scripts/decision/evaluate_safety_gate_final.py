"""Final Safety Gate evaluation (Week 13, Day 4): Gate A and Gate B on the TEST split and on
safety-stress-v1, ONCE, with the frozen safety-policy-v1. Gate A uses no margin, Gate B the
margins (m_U, m_R) of the outcome artifact. Nothing is tuned here: the two configurations and
the metrics are fixed before the data is scored.

Rehearsal on the validation split only (no lock; test and stress stay closed):
    PYTHONPATH=$PWD python3 scripts/decision/evaluate_safety_gate_final.py \
        --dataset DIR --dry-run --out-dir /tmp/safety-rehearsal
The real run creates evidence/week13/safety_test_lock.json before the Gate is scored and
refuses to run again while that file exists:
    PYTHONPATH=$PWD python3 scripts/decision/evaluate_safety_gate_final.py \
        --dataset DIR --stress-root DIR2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.twin.decision import safety_report as rep  # noqa: E402
from src.twin.decision.safety_policy import POLICY_PATH, load_safety_policy  # noqa: E402
from src.twin.decision.safety_replay import ReplayResult, run_gate  # noqa: E402
from src.twin.outcome import hybrid, predictor  # noqa: E402
from src.twin.outcome.dataset import (  # noqa: E402
    attempt_dirs,
    build_rows,
    capacity_info,
    load_dataset,
    load_dataset_config,
    saturation_flags,
)

FROZEN_POLICY_SHA12 = "8c548830861b"
ARTIFACT = REPO / "data" / "models" / "outcome" / "post-migration-v1"
CAPACITY = REPO / "data" / "benchmarks" / "controller_capacity.json"
CONFIG = REPO / "configs" / "outcome" / "dataset.yaml"
OUT = REPO / "evidence" / "week13"
LOCK_NAME = "safety_test_lock.json"
RULE = "test and safety-stress-v1 scored once with the frozen policy; nothing selected on them"
EXPECTED_ROWS = {"test": 60, "stress": 60}


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stress_rows(root: Path) -> list[dict[str, Any]]:
    """Rows of the raw safety-stress-v1 attempts; all of them belong to the label 'stress'."""
    split_map = {}
    for directory in attempt_dirs(root):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        split_map[str(metadata["attempt_id"])] = "stress"
    rows, noop, excluded = build_rows(
        root, split_map, capacity_info(CAPACITY)["c_safe_pps"],
        load_dataset_config(CONFIG), saturation_flags(root))
    if noop or excluded:
        raise SystemExit(f"stress set: {len(noop)} NO_OP rows, excluded {excluded}")
    return rows


def fmt(rate: Any) -> str:
    if rate.value is None:
        return "n/a"
    return f"{rate.numerator}/{rate.denominator}={rate.value:.3f} [{rate.low:.3f},{rate.high:.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="folder with outcome_dataset.csv")
    parser.add_argument("--stress-root", type=Path, help="raw safety-stress-v1 attempts")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--dry-run", action="store_true", help="validation split only, no lock")
    args = parser.parse_args()
    out: Path = args.out_dir.expanduser()
    lock = out / LOCK_NAME
    if args.dry_run and out.resolve() == OUT.resolve():
        parser.error("--dry-run needs --out-dir outside evidence/week13")
    if not args.dry_run:
        if args.stress_root is None:
            parser.error("the real run needs --stress-root")
        if lock.exists():
            raise SystemExit(f"already evaluated: {lock} exists. Test and stress are used once.")
    policy = load_safety_policy()
    policy_sha = sha12(POLICY_PATH)
    if policy_sha != FROZEN_POLICY_SHA12:
        raise SystemExit(f"policy sha {policy_sha} is not the frozen {FROZEN_POLICY_SHA12}")
    csv_path = args.dataset.expanduser() / "outcome_dataset.csv"
    every = load_dataset(csv_path)
    sets = {"validation": [r for r in every if r["split"] == "validation"]}
    if not args.dry_run:
        sets["test"] = [r for r in every if r["split"] == "test"]
        sets["stress"] = stress_rows(args.stress_root.expanduser())
        for name, expected in EXPECTED_ROWS.items():
            if len(sets[name]) != expected:
                raise SystemExit(f"{name}: {len(sets[name])} rows, expected {expected}")
    runtime = predictor._load_runtime(args.artifact)
    values = {name: hybrid.predict(runtime.models, rows, list(runtime.hybrid_features))
              for name, rows in sets.items()}
    out.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        rep.open_lock(lock, policy_sha, RULE)  # after the data is prepared, before it is scored
    gates = {"gate_a": (0.0, 0.0), "gate_b": (runtime.m_u, runtime.m_r)}
    results: dict[tuple[str, str], ReplayResult] = {}
    for split, rows in sets.items():
        for gate, (m_u, m_r) in gates.items():
            results[(split, gate)] = run_gate(rows, values[split], policy, m_u, m_r)
    u_limit = policy.utilization_safe_limit.value
    assert u_limit is not None
    files = {
        "safety_confusion_table.csv": rep.write_csv(
            out / "safety_confusion_table.csv",
            [rep.confusion_row(s, g, r.metrics) for (s, g), r in results.items()],
            rep.CONFUSION_COLUMNS),
        "safety_margin_ablation.csv": rep.write_csv(
            out / "safety_margin_ablation.csv",
            [rep.ablation_row(s, g, *gates[g], r.metrics) for (s, g), r in results.items()],
            rep.ABLATION_COLUMNS),
        "safety_reason_distribution.csv": rep.write_csv(
            out / "safety_reason_distribution.csv",
            [row for (s, g), r in results.items() for row in rep.reason_rows(s, g, r)],
            rep.REASON_COLUMNS),
    }
    report: dict[str, Any] = {
        "dry_run": args.dry_run,
        "rows": {name: len(rows) for name, rows in sets.items()},
        "inputs": {"dataset_sha12": sha12(csv_path), "policy_sha12": policy_sha,
                   "policy_version": policy.version, "m_U": runtime.m_u, "m_R": runtime.m_r},
        "limits": {"u_limit": u_limit, "r95_max": policy.response_p95_max_ms.value},
        "unsafe_by_label": {name: results[(name, "gate_a")].metrics.unsafe for name in sets},
        "notes": ["Gate A and Gate B both use the frozen policy (minimum_source_relief included);"
                  " they differ only in the uncertainty margins",
                  "rows are executed migrations: rejected candidates have no observed outcome",
                  "unsafe = observed U_dst_post above the utilization limit; R95_max is null",
                  "validation is in-sample for m_U; conclusions use test and stress only"],
        "results": {f"{s}/{g}": {"metrics": r.metrics.to_dict(),
                                 "failed_reasons": r.failed_reasons,
                                 "skipped_checks": r.skipped_checks}
                    for (s, g), r in results.items()},
    }
    (out / "safety_final_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files["safety_final_evaluation.json"] = hashlib.sha256(
        (out / "safety_final_evaluation.json").read_bytes()).hexdigest()
    if not args.dry_run:
        rep.complete_lock(lock, files)
    for (split, gate), res in results.items():
        m = res.metrics
        print(f"{split}/{gate}: rows={m.records} unsafe={m.unsafe} accepted={m.accepted} "
              f"UCRR {fmt(m.ucrr)} UAR {fmt(m.uar)} FRR {fmt(m.frr)} SAP {fmt(m.sap)}")
    print("wrote", out, "dry_run" if args.dry_run else "locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
