"""Closed-loop 2C analysis (Week 13, Day 6, My): latency, horizon, prediction error, gates.

Reads data/decision_runs/<decision_id>/record.json. The record is written by Vy's
closed loop (Day 5); this file fixes the fields the analysis needs (the contract):

  decision_id, scenario, mode ("dry_run" | "execute"), status, snapshot_id,
  timings_ms: {forecast, risk, candidate, outcome, safety, ranking, revalidation},
  collection_latency_ms,               # of the pinned snapshot
  funnel: {raw, eligible, evaluated, safe},
  reasons: {REASON_CODE: count},       # Gate + pre-check rejections of this decision
  selected: null | {candidate_id, predicted_src_utilization, predicted_dst_utilization,
                    predicted_imbalance, utilization_margin},
  transaction: null | {transaction_id, status, duration_s},
  actual_post: null | {u_src, u_dst, i}   # mean of the post window, execute mode only

Usage: python3 scripts/decision/analyze_closed_loop.py [--runs data/decision_runs]
Writes evidence/week14/closed_loop_2c_analysis.json. Final tables use execute records
only; dry-run records are reported separately and labelled.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.twin.forecasting.horizon import RuntimeLatency, estimate_horizon  # noqa: E402
from src.twin.forecasting.margin import finite_sample_quantile  # noqa: E402

STAGES = ("forecast", "risk", "candidate", "outcome", "safety", "ranking", "revalidation")
REQUIRED = ("decision_id", "scenario", "mode", "status", "timings_ms", "funnel", "reasons")
CURRENT_H = 3  # configs/forecasting/forecast.yaml (provisional)
POLICY_INPUTS = ROOT / "evidence" / "week13" / "safety_policy_inputs.json"
Record = Mapping[str, Any]


def _executed(r: Record, status: str) -> bool:
    return r["mode"] == "execute" and (r.get("transaction") or {}).get("status") == status


# Scenarios whose workload may stay below the risk threshold (U + m_f >= 1). When every
# record ever flags a controller at risk the result is NOT_TRIGGERED: neither PASS nor FAIL, reported as is.
MAY_NOT_TRIGGER = ("gradual", "hot_switch")
NOT_TRIGGERED = "NOT_TRIGGERED"

# Expected outcome per scenario (plan Day 6). A scenario with no record is MISSING.
GATES: dict[str, tuple[str, Callable[[Sequence[Record]], bool]]] = {
    "stable": ("no migration is executed",
               lambda rs: not any(r.get("transaction") for r in rs)),
    "gradual": ("at least one migration is COMMITTED",
                lambda rs: any(_executed(r, "COMMITTED") for r in rs)),
    "hot_switch": ("at least one migration is COMMITTED",
                   lambda rs: any(_executed(r, "COMMITTED") for r in rs)),
    "sustained_overload": ("at least one migration is COMMITTED",
                           lambda rs: any(_executed(r, "COMMITTED") for r in rs)),
    "verification_failure": ("the transaction ends RESTORED",
                             lambda rs: any(_executed(r, "RESTORED") for r in rs)),
    "no_safe_action": ("every decision with a controller at risk is NO_SAFE_ACTION, no migration",
                       lambda rs: any(r.get("at_risk") for r in rs)
                       and all(r["status"] == "NO_SAFE_ACTION" for r in rs if r.get("at_risk"))
                       and not any(r.get("transaction") for r in rs)),
    "dst_near_capacity": ("the Gate rejects at least one candidate for the destination",
                          lambda rs: any(r["reasons"].get("DESTINATION_ALREADY_AT_RISK", 0)
                                         + r["reasons"].get("DST_UTILIZATION_UNSAFE", 0) > 0
                                         for r in rs)),
    "invalid_snapshot": ("no migration on an invalid snapshot",
                         lambda rs: not any(r.get("transaction") for r in rs)),
    "cooldown": ("COOLDOWN_ACTIVE or DEFERRED is recorded",
                 lambda rs: any(r["status"] == "DEFERRED"
                                or r["reasons"].get("COOLDOWN_ACTIVE", 0) for r in rs)),
    "conflict": ("ACTIVE_TRANSACTION_CONFLICT or DEFERRED is recorded",
                 lambda rs: any(r["status"] == "DEFERRED"
                                or r["reasons"].get("ACTIVE_TRANSACTION_CONFLICT", 0)
                                for r in rs)),
}


def load_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.glob("*/record.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = [k for k in REQUIRED if k not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing} (see the contract in this script)")
        records.append(data)
    return records


def stats(values: Sequence[Any]) -> dict[str, Any]:
    clean = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not clean:
        return {"n": 0}
    mid = len(clean) // 2
    p50 = clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2
    return {"n": len(clean), "mean": sum(clean) / len(clean), "p50": p50,
            "p95": finite_sample_quantile(clean, 0.95), "max": clean[-1]}


def latency(records: Sequence[Record]) -> dict[str, Any]:
    """Stage latency; decision_total only over decisions that ran the whole pipeline
    (candidates, outcome, Gate, ranking): a NO_RISK decision stops after the forecast
    and would make the total look two orders of magnitude faster than it is."""
    out = {s: stats([r["timings_ms"].get(s) for r in records if s in r["timings_ms"]])
           for s in STAGES}
    full = [r for r in records if "ranking" in r["timings_ms"]]
    out["decision_total"] = stats(
        [sum(float(v) for k, v in r["timings_ms"].items() if k in STAGES) for r in full])
    out["decision_total_all_records"] = stats(
        [sum(float(v) for k, v in r["timings_ms"].items() if k in STAGES) for r in records
         if r["timings_ms"]])
    return out


def _durations_ms(records: Sequence[Record], statuses: tuple[str, ...]) -> list[float]:
    return sorted(1000.0 * float(r["transaction"]["duration_s"]) for r in records
                  if (r.get("transaction") or {}).get("duration_s") is not None
                  and r["transaction"].get("status") in statuses)


def horizon(records: Sequence[Record], decision_p95_ms: float | None) -> dict[str, Any]:
    """Lead time = collection + decision + migration (P95 each) -> horizon steps of 1 s.

    The migration term is the duration of COMMITTED transactions: the horizon answers
    "how early must the risk be seen for a migration to finish in time". A rolled-back
    transaction takes longer (verification timeout); the same estimate with rollbacks
    included is reported next to it, not hidden.
    """
    collection = stats([r.get("collection_latency_ms") for r in records])
    committed = _durations_ms(records, ("COMMITTED",))
    with_rollback = _durations_ms(records, ("COMMITTED", "RESTORED"))
    if committed:
        migration_p95, source = finite_sample_quantile(committed, 0.95), "closed-loop runs"
    else:
        data = json.loads(POLICY_INPUTS.read_text(encoding="utf-8"))
        migration_p95 = 1000.0 * float(data["cooldown"]["transaction_p95_s"])
        source = f"{POLICY_INPUTS.relative_to(ROOT)} (raw-v2 transaction P95)"
    if decision_p95_ms is None or collection.get("n", 0) == 0:
        return {"status": "NOT_AVAILABLE", "reason": "no decision or collection latency"}
    h = estimate_horizon(RuntimeLatency(collection["p95"], decision_p95_ms, migration_p95), 1.0)
    out = {"status": "OK", "collection_p95_ms": collection["p95"],
           "decision_p95_ms": decision_p95_ms, "migration_p95_ms": migration_p95,
           "migration_source": source, "n_committed": len(committed), **h.to_dict(),
           "current_h": CURRENT_H, "current_h_sufficient": h.horizon_steps <= CURRENT_H,
           "margin_seconds": CURRENT_H - h.required_lead_time_seconds}
    if len(with_rollback) > len(committed):
        worst = finite_sample_quantile(with_rollback, 0.95)
        hr = estimate_horizon(RuntimeLatency(collection["p95"], decision_p95_ms, worst), 1.0)
        out["including_rollbacks"] = {"n": len(with_rollback), "migration_p95_ms": worst,
                                      "required_lead_time_seconds": hr.required_lead_time_seconds,
                                      "horizon_steps": hr.horizon_steps}
    return out


def prediction_errors(records: Sequence[Record]) -> dict[str, Any]:
    pairs = [(r["selected"], r["actual_post"]) for r in records
             if _executed(r, "COMMITTED") and r.get("selected") and r.get("actual_post")]
    names = (("u_src", "predicted_src_utilization"), ("u_dst", "predicted_dst_utilization"),
             ("i", "predicted_imbalance"))
    out: dict[str, Any] = {"n": len(pairs)}
    for actual_key, pred_key in names:
        errors = [float(a[actual_key]) - float(p[pred_key]) for p, a in pairs]
        out[f"{actual_key}_error_actual_minus_pred"] = stats(errors)
    covered = [float(a["u_dst"]) <= float(p["predicted_dst_utilization"])
               + float(p["utilization_margin"]) + 1e-9 for p, a in pairs]
    out["u_dst_upper_bound_coverage"] = (f"{sum(covered)}/{len(covered)}" if covered else None)
    out["migrations"] = [
        {"decision_id": r["decision_id"], "scenario": r["scenario"],
         "candidate_id": r["selected"]["candidate_id"],
         "u_before": {str(c["controller_id"]): float(c["utilization"])
                      for c in (r.get("twin_state") or {}).get("controllers", [])},
         "predicted_dst": float(r["selected"]["predicted_dst_utilization"]),
         "dst_upper_bound": float(r["selected"]["predicted_dst_utilization"])
         + float(r["selected"]["utilization_margin"]),
         "actual_dst": float(r["actual_post"]["u_dst"]),
         "actual_src": float(r["actual_post"]["u_src"]),
         "dst_above_safe_limit": float(r["actual_post"]["u_dst"]) > 1.0}
        for r in records
        if _executed(r, "COMMITTED") and r.get("selected") and r.get("actual_post")]
    out["n_committed_without_actual_post"] = sum(
        1 for r in records if _executed(r, "COMMITTED") and not r.get("actual_post"))
    return out


def analyze(records: Sequence[Record]) -> dict[str, Any]:
    by_mode: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)
    execute = by_mode.get("execute", [])
    lat = latency(execute or records)
    decision_p95 = lat["decision_total"].get("p95")
    scenarios: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        scenarios[r["scenario"]].append(r)
    gates = {}
    for name, (expect, check) in GATES.items():
        rs = scenarios.get(name, [])
        if not rs:
            result = "MISSING"
        elif name in MAY_NOT_TRIGGER and not any(
                r.get("at_risk") or r.get("transaction") for r in rs):
            result = NOT_TRIGGERED
        else:
            result = "PASS" if check(rs) else "FAIL"
        peak = max((float(c["utilization"]) for r in rs
                    for c in (r.get("twin_state") or {}).get("controllers", [])), default=None)
        gates[name] = {"expect": expect, "n": len(rs), "result": result,
                       "peak_utilization": peak,
                       "status_counts": dict(sorted(Counter(
                           f"{r['mode']}:{r['status']}" for r in rs).items()))}
    reasons: Counter[str] = Counter()
    for r in records:
        reasons.update({k: int(v) for k, v in r["reasons"].items()})
    return {
        "n_records": len(records),
        "n_by_mode": {k: len(v) for k, v in sorted(by_mode.items())},
        "latency_source": "execute" if execute else "dry_run (no execute record yet)",
        "status_counts": dict(sorted(Counter(r["status"] for r in records).items())),
        "reason_counts": dict(sorted(reasons.items())),
        "latency_ms": lat,
        "horizon": horizon(execute or records, decision_p95),
        "prediction_errors": prediction_errors(execute),
        "scenario_gates": gates,
        "all_gates_pass": all(g["result"] in ("PASS", NOT_TRIGGERED) for g in gates.values()),
        "not_triggered": sorted(k for k, g in gates.items() if g["result"] == NOT_TRIGGERED),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=ROOT / "data" / "decision_runs")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evidence" / "week14" / "closed_loop_2c_analysis.json")
    args = ap.parse_args(argv)
    report = analyze(load_records(args.runs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, gate in report["scenario_gates"].items():
        print(f"{gate['result']:13s} {name:22s} n={gate['n']:<3d} {gate['expect']}")
        print(f"{'':13s} {gate['status_counts']}")
    print("horizon:", json.dumps(report["horizon"]))
    for m in report["prediction_errors"].get("migrations", []):
        before = " ".join(f"{k}={v:.3f}" for k, v in sorted(m["u_before"].items()))
        print(f"migration {m['scenario']:20s} {m['candidate_id'].split(':', 1)[1]:12s} before {before}"
              f"  dst pred {m['predicted_dst']:.3f} bound {m['dst_upper_bound']:.3f}"
              f" actual {m['actual_dst']:.3f}{'  > 1' if m['dst_above_safe_limit'] else ''}")
    print(f"wrote {args.out}")
    return 0 if report["all_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
