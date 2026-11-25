"""Week 15 Day 5: do the 2C20S models transfer to 4C40S? Pure part of the decision.

The rule is configs/experiments/transfer_4c_v1.yaml, declared before any 4C run:
  forecast reuse  <=> macro RMSE ratio vs persistence over the SEEN workloads < max
                      and coverage of the frozen m_f on the 4C samples >= min;
  outcome reuse   <=> RMSE(hybrid_xgb) < RMSE(analytical) on u_dst_post
                      and one-sided coverage of the frozen m_U >= min.
hot_switch is unseen: computed and reported, never part of the decision.

Inputs are rows that already carry the predictions of the frozen models (the CLI
scripts/experiments/evaluate_transfer_4c.py reads them from JSONL). Nothing is refit here
and no margin is recomputed: the margins are the 2C validation ones from the artifacts.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.twin.forecasting import margin as fmargin
from src.twin.outcome import margins as omargin

UNSEEN = "hot_switch"


def norm_workload(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


@dataclass(frozen=True)
class Rules:
    rmse_ratio_max: float
    forecast_coverage_min: float
    hybrid_beats_analytical: bool
    outcome_coverage_min: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Rules:
        d = config["decision"]
        if d.get("hot_switch") != "report_only":
            raise ValueError("transfer rule v1 defines hot_switch: report_only only")
        rules = cls(
            rmse_ratio_max=float(d["forecast_reuse_if"]["rmse_ratio_vs_persistence_max"]),
            forecast_coverage_min=float(d["forecast_reuse_if"]["margin_coverage_min"]),
            hybrid_beats_analytical=bool(d["outcome_reuse_if"]["hybrid_beats_analytical"]),
            outcome_coverage_min=float(d["outcome_reuse_if"]["margin_coverage_min"]),
        )
        if rules.rmse_ratio_max <= 0:
            raise ValueError("rmse_ratio_vs_persistence_max must be positive")
        for value in (rules.forecast_coverage_min, rules.outcome_coverage_min):
            if not 0.0 < value <= 1.0:
                raise ValueError("coverage minimums must be in (0, 1]")
        return rules


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("actual and predicted must be non-empty and the same length")
    return math.sqrt(sum((float(a) - float(p)) ** 2 for a, p in zip(actual, predicted))
                     / len(actual))


def _number(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite: {row[key]!r}")
    return value


def forecast_transfer(rows: Sequence[Mapping[str, Any]], margins: Mapping[str, Any],
                      expected_workloads: Sequence[str]) -> dict[str, Any]:
    """Rows: workload_type, controller_id, utilization (U(t), the persistence forecast),
    future_utilization (U(t+h)), prediction (model forecast of U(t+h))."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[norm_workload(row["workload_type"])].append(row)
    expected = [norm_workload(w) for w in expected_workloads]
    missing = [w for w in expected if w not in groups]
    if missing:
        raise ValueError(f"no forecast rows for workloads {missing}")
    per: dict[str, dict[str, Any]] = {}
    for workload in expected:
        own = groups[workload]
        actual = [_number(r, "future_utilization") for r in own]
        model = [_number(r, "prediction") for r in own]
        persistence = [_number(r, "utilization") for r in own]
        m = [fmargin.margin_for(dict(margins), str(r["controller_id"])) for r in own]
        r_model, r_pers = rmse(actual, model), rmse(actual, persistence)
        per[workload] = {
            "n": len(own),
            "rmse_model": r_model,
            "rmse_persistence": r_pers,
            "rmse_ratio_vs_persistence": r_model / r_pers if r_pers > 0 else None,
            "coverage": fmargin.coverage(actual, model, m),
            "overload": fmargin.overload_confusion(actual, [p + k for p, k in zip(model, m)]),
            "role": "report_only" if workload == UNSEEN else "decides",
        }
    seen = [w for w in expected if w != UNSEEN]
    ratios = [per[w]["rmse_ratio_vs_persistence"] for w in seen]
    if any(r is None for r in ratios):
        raise ValueError("persistence RMSE is 0 on a seen workload: ratio undefined")
    seen_rows = [r for w in seen for r in groups[w]]
    seen_cov = fmargin.coverage(
        [_number(r, "future_utilization") for r in seen_rows],
        [_number(r, "prediction") for r in seen_rows],
        [fmargin.margin_for(dict(margins), str(r["controller_id"])) for r in seen_rows])
    return {"per_workload": per, "seen_workloads": seen,
            "macro_rmse_ratio_seen": sum(ratios) / len(ratios),
            "coverage_seen": seen_cov,
            "skill": "RMSE ratio model / persistence (< 1 = better than persistence)",
            "margin": {"source": "frozen 2C validation m_f", "controllers": dict(
                margins.get("controllers", {})), "global": margins.get("global")}}


def outcome_transfer(rows: Sequence[Mapping[str, Any]], m_u: float,
                     variants: Sequence[str]) -> dict[str, Any]:
    """Rows: workload, u_dst_post (actual) and pred_<variant> for every variant."""
    if m_u < 0 or not math.isfinite(m_u):
        raise ValueError("m_U must be finite and >= 0")
    if not rows:
        raise ValueError("no outcome rows")
    actual = [_number(r, "u_dst_post") for r in rows]
    per_variant = {}
    for v in variants:
        pred = [_number(r, f"pred_{v}") for r in rows]
        per_variant[v] = {"rmse_u_dst_post": rmse(actual, pred)}
    hybrid = [_number(r, "pred_hybrid_xgb") for r in rows]
    seen = [i for i, r in enumerate(rows) if norm_workload(r["workload"]) != UNSEEN]
    unseen = [i for i, r in enumerate(rows) if norm_workload(r["workload"]) == UNSEEN]

    def cov(index: list[int]) -> dict[str, Any] | None:
        if not index:
            return None
        return omargin.coverage([actual[i] for i in index], [hybrid[i] for i in index], m_u)

    by_workload = omargin.coverage_by_group(
        rows, "u_dst_post", hybrid, m_u, {"workload": lambda r: norm_workload(r["workload"])})
    return {"n": len(rows), "per_variant": per_variant,
            "coverage_seen": cov(seen), "coverage_hot_switch": cov(unseen),
            "coverage_by_workload": by_workload,
            "gate_confusion_seen": omargin.gate_confusion(
                [actual[i] for i in seen], [hybrid[i] for i in seen], m_u) if seen else None,
            "margin": {"source": "frozen 2C validation m_U (hybrid_xgb)", "m_U": m_u}}


def decide(forecast: Mapping[str, Any], outcome: Mapping[str, Any] | None,
           rules: Rules) -> dict[str, Any]:
    """outcome None = forecast part only (the outcome attempts come with Day 6)."""
    f_ratio = float(forecast["macro_rmse_ratio_seen"])
    f_cov = float(forecast["coverage_seen"])
    f_checks = {"macro_rmse_ratio_below_max": f_ratio < rules.rmse_ratio_max,
                "coverage_at_least_min": f_cov >= rules.forecast_coverage_min}
    result: dict[str, Any] = {
        "forecast_reuse": all(f_checks.values()), "forecast_checks": f_checks,
        "outcome_reuse": None, "outcome_checks": None,
        "note": "reuse = keep the 2C model for the 4C campaign; otherwise retrain on 4C "
                "data. hot_switch is reported, never decides."}
    if outcome is None:
        result["note"] += " Outcome part pending (Day 6 attempts)."
        return result
    seen_cov = outcome["coverage_seen"]
    if seen_cov is None:
        raise ValueError("no outcome attempt on a seen workload: the rule cannot be applied")
    per = outcome["per_variant"]
    o_checks = {"coverage_at_least_min":
                float(seen_cov["coverage"]) >= rules.outcome_coverage_min}
    if rules.hybrid_beats_analytical:
        o_checks["hybrid_beats_analytical"] = (per["hybrid_xgb"]["rmse_u_dst_post"]
                                               < per["analytical"]["rmse_u_dst_post"])
    result.update(outcome_reuse=all(o_checks.values()), outcome_checks=o_checks)
    return result
