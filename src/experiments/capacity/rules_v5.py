"""capacity-v5 derivation rules (configs/experiments/capacity_v5_rules.yaml), pure part.

Input rows have the columns of capacity_runs.csv (capacity-v4) or capacity_runs_4c40s.csv:
run_id, controller, target_new_flow_rate, processed_packet_in_rate_mean, qos_success_ratio,
flow_setup_latency_p95_ms, valid. Only valid rows whose run_id starts with the rule's
experiment are used.
"""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RateStats:
    controller: str
    rate: float
    processed: tuple[float, ...]
    setup_p95_ms: float      # mean over repeats of the 10-hop probe P95
    qos_success: float       # mean over repeats


@dataclass(frozen=True)
class Rules:
    experiment: str
    repeats: int
    t_one_sided_95: float
    setup_ratio_max: float
    qos_success_min: float
    setup_jump_ratio: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Rules:
        if config.get("safe_reference") != "common":
            raise ValueError("only safe_reference: common is defined")
        rules = cls(
            experiment=str(config["experiment"]),
            repeats=int(config["repeats"]),
            t_one_sided_95=float(config["t_one_sided_95"]),
            setup_ratio_max=float(config["healthy"]["setup_ratio_max"]),
            qos_success_min=float(config["healthy"]["qos_success_min"]),
            setup_jump_ratio=float(config["setup_jump_ratio"]),
        )
        if rules.repeats < 2 or rules.setup_ratio_max <= 1 or rules.setup_jump_ratio <= 1:
            raise ValueError("repeats >= 2, setup_ratio_max > 1 and setup_jump_ratio > 1")
        return rules


def group_rows(rows: Iterable[Mapping[str, str]], rules: Rules) -> dict[str, list[RateStats]]:
    """Per controller, the swept rates in increasing order (valid rows of the experiment)."""
    groups: dict[tuple[str, float], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row["run_id"].startswith(rules.experiment + "-") and \
                str(row["valid"]).lower() == "true":
            groups[(row["controller"], float(row["target_new_flow_rate"]))].append(row)
    out: dict[str, list[RateStats]] = defaultdict(list)
    for (controller, rate), group in sorted(groups.items()):
        out[controller].append(RateStats(
            controller=controller,
            rate=rate,
            processed=tuple(float(r["processed_packet_in_rate_mean"]) for r in group),
            setup_p95_ms=statistics.mean(float(r["flow_setup_latency_p95_ms"]) for r in group),
            qos_success=statistics.mean(float(r["qos_success_ratio"]) for r in group),
        ))
    if not out:
        raise ValueError(f"no valid {rules.experiment} rows")
    return dict(out)


def _complete(stats: RateStats, rules: Rules) -> RateStats:
    if len(stats.processed) != rules.repeats:
        raise ValueError(f"{stats.controller} {stats.rate:g} fps: {len(stats.processed)} valid "
                         f"repeats, {rules.repeats} needed")
    return stats


def healthy_rates(sweep: list[RateStats], rules: Rules) -> list[float]:
    """Rates healthy from the baseline up, stopping at the first unhealthy one."""
    baseline = _complete(sweep[0], rules).setup_p95_ms
    if baseline <= 0:
        raise ValueError(f"{sweep[0].controller}: baseline setup latency is not positive")
    healthy: list[float] = []
    for stats in sweep:
        _complete(stats, rules)
        if stats.qos_success < rules.qos_success_min or \
                stats.setup_p95_ms > rules.setup_ratio_max * baseline:
            break
        healthy.append(stats.rate)
    return healthy


def setup_knee(sweep: list[RateStats], rules: Rules) -> float | None:
    for below, stats in zip(sweep, sweep[1:]):
        if stats.qos_success < rules.qos_success_min or \
                stats.setup_p95_ms > rules.setup_jump_ratio * below.setup_p95_ms:
            return stats.rate
    return None


def throughput_knee(sweep: list[RateStats], rules: Rules) -> float | None:
    return next((s.rate for s in sweep if s.qos_success < rules.qos_success_min), None)


def lower_bound(values: tuple[float, ...], t_value: float) -> float:
    return statistics.mean(values) - t_value * statistics.stdev(values) / math.sqrt(len(values))


def derive(rows: Iterable[Mapping[str, str]], rules: Rules) -> dict[str, Any]:
    sweeps = group_rows(rows, rules)
    healthy = {c: healthy_rates(s, rules) for c, s in sweeps.items()}
    if any(not rates for rates in healthy.values()):
        raise ValueError(f"a controller has no healthy rate: {healthy}")
    common = min(rates[-1] for rates in healthy.values())
    controllers: dict[str, Any] = {}
    for controller, sweep in sorted(sweeps.items()):
        by_rate = {s.rate: s for s in sweep}
        safe = by_rate[common]
        knee_setup = setup_knee(sweep, rules)
        knee_throughput = throughput_knee(sweep, rules)
        c_setup = None if knee_setup is None else statistics.mean(by_rate[knee_setup].processed)
        c_throughput = (None if knee_throughput is None
                        else statistics.mean(by_rate[knee_throughput].processed))
        limits = [v for v in (c_setup, c_throughput) if v is not None]
        controllers[controller] = {
            "c_throughput_pps": c_throughput,
            "c_setup_pps": c_setup,
            "c_response_pps": None,
            "c_critical_pps": min(limits) if limits else None,
            "c_safe_pps": lower_bound(safe.processed, rules.t_one_sided_95),
            "setup_knee_rate_fps": knee_setup,
            "throughput_knee_rate_fps": knee_throughput,
            "highest_healthy_rate_fps": healthy[controller][-1],
            "baseline_setup_p95_ms": sweep[0].setup_p95_ms,
            "safe_reference": {
                "target_rate_fps": common,
                "processed_mean_pps": statistics.mean(safe.processed),
                "processed_std_pps": statistics.stdev(safe.processed),
                "sample_count": len(safe.processed),
                "setup_p95_ms": safe.setup_p95_ms,
            },
        }
    return {
        "schema_version": "1.0",
        "method": {
            "experiment": rules.experiment,
            "rules": "configs/experiments/capacity_v5_rules.yaml",
            "repeats_per_rate": rules.repeats,
            "safe_reference_rate_fps": common,
            "safe_definition": "one-sided 95% lower confidence bound at the highest rate "
                               "healthy for every controller",
            "student_t": rules.t_one_sided_95,
        },
        "controllers": controllers,
    }
