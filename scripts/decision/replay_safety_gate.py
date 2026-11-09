"""Replay the Safety Gate on the VALIDATION rows of the processed outcome dataset.

Only split == "validation" is read: the test split and safety-stress-v1 stay closed until the
policy is frozen. Four configurations: Gate A (no margin) and Gate B (artifact margins), each
without and with the proposed minimum_source_relief (= m_U, a noise floor). Nothing here changes
the policy file; the proposal is only evaluated.
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

from src.twin.decision.safety_metrics import Rate  # noqa: E402
from src.twin.decision.safety_policy import POLICY_PATH, load_safety_policy  # noqa: E402
from src.twin.decision.safety_replay import (  # noqa: E402
    ReplayResult,
    run_gate,
    with_minimum_relief,
)
from src.twin.outcome import hybrid, predictor  # noqa: E402
from src.twin.outcome.dataset import load_dataset  # noqa: E402

ARTIFACT = REPO / "data" / "models" / "outcome" / "post-migration-v1"
OUT = REPO / "evidence" / "week13" / "safety_replay_validation.json"
RELIEF_SOURCE = "proposal: m_U of post-migration-v1 (noise floor of the model)"


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def fmt(rate: Rate | None) -> str:
    if rate is None or rate.value is None or rate.low is None or rate.high is None:
        return "n/a"
    return f"{rate.numerator}/{rate.denominator}={rate.value:.3f} [{rate.low:.3f},{rate.high:.3f}]"


def summary(result: ReplayResult) -> dict[str, Any]:
    return {"metrics": result.metrics.to_dict(), "failed_reasons": result.failed_reasons,
            "skipped_checks": result.skipped_checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="folder with outcome_dataset.csv")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    csv_path = args.dataset / "outcome_dataset.csv"
    policy = load_safety_policy()
    runtime = predictor._load_runtime(args.artifact)
    rows = [r for r in load_dataset(csv_path) if r["split"] == "validation"]
    values = hybrid.predict(runtime.models, rows, list(runtime.hybrid_features))
    relief = with_minimum_relief(policy, runtime.m_u, RELIEF_SOURCE)
    plan = {"gate_a": (policy, 0.0, 0.0), "gate_b": (policy, runtime.m_u, runtime.m_r),
            "gate_a_relief": (relief, 0.0, 0.0),
            "gate_b_relief": (relief, runtime.m_u, runtime.m_r)}
    results = {name: run_gate(rows, values, pol, mu, mr) for name, (pol, mu, mr) in plan.items()}
    report: dict[str, Any] = {
        "split": "validation", "rows": len(rows),
        "inputs": {"dataset_sha12": sha12(csv_path), "policy_sha12": sha12(POLICY_PATH),
                   "policy_version": policy.version, "m_U": runtime.m_u, "m_R": runtime.m_r,
                   "minimum_source_relief_proposal": runtime.m_u},
        "limits": {"u_limit": policy.utilization_safe_limit.value,
                   "r95_max": policy.response_p95_max_ms.value},
        "notes": ["validation rows only; test and safety-stress-v1 not read",
                  "rows are executed migrations: rejected candidates have no observed outcome"],
        "results": {name: summary(res) for name, res in results.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("rows", len(rows), "dataset", report["inputs"]["dataset_sha12"],
          "policy", report["inputs"]["policy_sha12"])
    for name, res in results.items():
        m = res.metrics
        print(f"{name}: unsafe={m.unsafe} accepted={m.accepted} UCRR {fmt(m.ucrr)}"
              f" UAR {fmt(m.uar)} FRR {fmt(m.frr)} SAP {fmt(m.sap)} DOAR {fmt(m.doar)}")
        print("   failed", res.failed_reasons, "skipped", res.skipped_checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
