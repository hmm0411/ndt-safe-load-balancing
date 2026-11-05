"""Write the provenance of every safety-policy-v1 threshold.

Policy values are read from the policy file, never retyped, so the manifest
cannot disagree with it. A threshold without a data-backed source is
reported as NOT_CONFIGURED with the reason, never filled with a guess.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from src.schemas.decision import STRUCTURAL_REASONS, ReasonCode
from src.twin.decision.safety_policy import DEFAULT_POLICY_PATH, load_safety_policy

OUT_DIR = Path("data/decision/safety-v1")
CAPACITY = Path("data/benchmarks/controller_capacity.json")
MARGINS = Path("evidence/week12/uncertainty_margin.json")
COOLDOWN = Path("evidence/week13/cooldown_lower_bound.json")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _margin(margins: dict, key: str) -> dict:
    entry = margins["by_model"][margins["deployed_model"]][key]
    return {
        "value": entry["value"],
        "status": "DERIVED",
        "source": str(MARGINS),
        "method": entry["method"],
        "n": entry["n"],
        "rank": entry["rank"],
        "level": entry["level"],
    }


def _not_configured(reason: str, pending: str, **extra: object) -> dict:
    return {"value": None, "status": "NOT_CONFIGURED", "reason": reason, "pending": pending, **extra}


def build_manifest() -> dict:
    policy = load_safety_policy()
    margins, capacity, cooldown = _load(MARGINS), _load(CAPACITY), _load(COOLDOWN)
    c_safe = {c: v["c_safe_pps"] for c, v in capacity["controllers"].items()}
    c_resp = {c: v["c_response_pps"] for c, v in capacity["controllers"].items()}
    thresholds = {
        "utilization_safe_limit": {
            "value": policy.utilization_safe_limit,
            "status": "CONVENTION",
            "source": "U = processed_packet_in_rate / c_safe_pps; U = 1 is safe capacity",
            "comparison": "U_hat + m_U <= limit",
        },
        "m_U": _margin(margins, "m_U"),
        "m_R": _margin(margins, "m_R"),
        "response_p95_max_ms": _not_configured(
            "capacity-v4 never found response limiting capacity (c_response_pps is "
            f"{c_resp}); no frozen operational response requirement exists",
            "supervisor chooses one of three options in the Week 13-14 plan",
        ),
        "minimum_quality_score": _not_configured(
            "quality-score-v1 defines per-component good/hard thresholds only; no calibrated "
            "minimum score exists. The Hard Gate (valid) stays mandatory.",
            "needs a calibrated threshold",
        ),
        "cooldown_seconds": _not_configured(
            "only a lower bound exists; the policy value is not chosen yet",
            "confirm the policy value",
            lower_bound_s=cooldown["cooldown_lower_bound_s"],
            lower_bound_source=str(COOLDOWN),
        ),
        "minimum_source_relief": _not_configured(
            "must be chosen on validation raw-v2, not set by hand",
            "proposal from validation (My, Day 1)",
        ),
    }
    return {
        "policy_version": policy.version,
        "policy_file": str(DEFAULT_POLICY_PATH),
        "policy_sha256": hashlib.sha256(DEFAULT_POLICY_PATH.read_bytes()).hexdigest(),
        "git_commit": _git_commit(),
        "capacity_version": "capacity-v4",
        "c_safe_pps": c_safe,
        "outcome_model": {"version": margins["model_version"], "deployed": margins["deployed_model"]},
        "quality_score": "quality-score-v1",
        "forecast_model": {"version": None, "status": "PENDING", "reason": "forecast-models-v3 not trained yet"},
        "thresholds": thresholds,
        "not_configured": list(policy.not_configured()),
    }


def build_reason_codes() -> dict:
    return {
        "reason_codes": [r.value for r in ReasonCode],
        "structural": sorted(r.value for r in STRUCTURAL_REASONS),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {"policy_manifest.json": build_manifest(), "reason_codes.json": build_reason_codes()}
    for name, payload in files.items():
        (OUT_DIR / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("wrote", OUT_DIR / name)


if __name__ == "__main__":
    main()
