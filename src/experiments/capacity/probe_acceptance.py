"""Day 3: judge the flow-setup probe against configs/experiments/probe_acceptance_v1.yaml.

Input is evidence/week15/probe_sweep_summary.json written by
scripts/experiments/export_4c40s_sweeps.py (curves per target controller and rate, and the
probe-induced load). Every criterion keeps the numbers it was judged on, so the verdict can
be checked by hand. A missing measurement is a failure ("missing"), never a pass.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CONTROLLERS = ("c1", "c2", "c3", "c4")


def _rate_key(rate: float) -> str:
    return f"{float(rate):g}"


def _p95(curves: Mapping[str, Any], target: str, rate: float, cid: str) -> float | None:
    cell = curves.get(target, {}).get(_rate_key(rate), {}).get(cid) or {}
    stats = cell.get("setup_p95_ms") or {}
    value = stats.get("mean")
    return None if value is None else float(value)


def baselines(summary: Mapping[str, Any], rate: float) -> dict[str, float | None]:
    """Per controller: mean P95 over every run where all controllers carry the base rate."""
    curves = summary["curves"]
    out: dict[str, float | None] = {}
    for cid in CONTROLLERS:
        values = [v for t in curves if (v := _p95(curves, t, rate, cid)) is not None]
        out[cid] = sum(values) / len(values) if values else None
    return out


def judge(summary: Mapping[str, Any], rules: Mapping[str, Any]) -> dict[str, Any]:
    curves = summary["curves"]
    base_rate = float(rules["baseline_rate_fps"])
    rates = [float(r) for r in rules["rates_fps"]]
    top = max(rates)
    base = baselines(summary, base_rate)
    criteria: dict[str, Any] = {}

    # P1 isolation
    rows, ok = [], True
    for target in CONTROLLERS:
        for rate in rates:
            if rate == base_rate:
                continue
            for cid in CONTROLLERS:
                if cid == target:
                    continue
                value, ref = _p95(curves, target, rate, cid), base[cid]
                ratio = None if value is None or not ref else value / ref
                passed = ratio is not None and ratio <= float(rules["isolation_max_ratio"])
                ok &= passed
                rows.append({"target": target, "rate": rate, "controller": cid,
                             "p95_ms": value, "baseline_ms": ref, "ratio": ratio,
                             "pass": passed})
    criteria["P1_isolation"] = {"pass": ok, "limit_ratio": rules["isolation_max_ratio"],
                                "rows": rows}

    # P2 monotone
    rows, ok = [], True
    for target in CONTROLLERS:
        values = [_p95(curves, target, r, target) for r in sorted(rates)]
        known = [v for v in values if v is not None]
        passed = len(known) == len(values) and all(a < b for a, b in zip(known, known[1:]))
        ok &= passed
        rows.append({"controller": target, "rates": sorted(rates), "p95_ms": values,
                     "pass": passed})
    criteria["P2_monotone"] = {"pass": ok, "rows": rows}

    # P3 saturation visible at the top rate
    rows, ok = [], True
    for target in CONTROLLERS:
        cell = curves.get(target, {}).get(_rate_key(top), {}).get(target) or {}
        value, ref = _p95(curves, target, top, target), base[target]
        timeouts = int(cell.get("timeouts") or 0)
        ratio = None if value is None or not ref else value / ref
        passed = value is not None and value > 0 and (
            timeouts > 0 or (ratio is not None and ratio >= float(rules["saturation_min_ratio"])))
        ok &= passed
        rows.append({"controller": target, "rate": top, "p95_ms": value, "baseline_ms": ref,
                     "ratio": ratio, "timeouts": timeouts, "pass": passed})
    criteria["P3_saturation_visible"] = {"pass": ok, "min_ratio": rules["saturation_min_ratio"],
                                         "rows": rows}

    # P4 probe-induced load
    rows, ok = [], True
    for cid in CONTROLLERS:
        load = summary.get("probe_load", {}).get(cid) or {}
        delta = load.get("delta_u")
        passed = delta is not None and abs(float(delta)) <= float(rules["probe_load_max_delta_u"])
        ok &= passed
        rows.append({"controller": cid, "delta_u": delta, "delta_pps": load.get("delta_pps"),
                     "pass": passed})
    criteria["P4_probe_load"] = {"pass": ok, "limit_delta_u": rules["probe_load_max_delta_u"],
                                 "scale_source": summary.get("scale_source"), "rows": rows}

    # P5 enough samples per run and controller
    expected = (float(rules["expected_switches_per_controller"])
                * float(rules["probe_rate_per_second"]) * float(rules["measurement_seconds"]))
    floor = float(rules["min_sample_share"]) * expected
    rows, ok = [], True
    for target, by_rate in curves.items():
        for rate_key, cell in by_rate.items():
            repeats = int(cell.get("repeats") or 0)
            for cid in CONTROLLERS:
                samples = int((cell.get(cid) or {}).get("samples") or 0)
                per_run = samples / repeats if repeats else 0.0
                passed = per_run >= floor
                ok &= passed
                rows.append({"target": target, "rate": rate_key, "controller": cid,
                             "samples_per_run": per_run, "pass": passed})
    criteria["P5_samples"] = {"pass": ok, "floor_per_run": floor, "rows": rows}

    all_pass = all(c["pass"] for c in criteria.values())
    return {
        "rules": "configs/experiments/probe_acceptance_v1.yaml",
        "baseline_p95_ms": base,
        "criteria": criteria,
        "all_pass": all_pass,
        "decision": "Huong 3 (R95 = probe flow-setup P95)" if all_pass else "Huong 2",
    }
