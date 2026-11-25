"""Rows and summaries of the Week 15 4C40S sweeps (pure part, no Mininet).

One row per run directory of the four sweep configs. Columns: the capacity-v4 columns of
capacity_runs.csv (from summary.json), then per controller the processed Packet-In rate over
the measurement window (controllers.jsonl) and the flow-setup probe P95 / samples /
timeouts (setup_probe_summary.json), then the machine (run_conditions.json).
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from src.experiments.capacity.aggregator import in_window, parse_dt, read_jsonl

EXPERIMENTS = ("probe-off-4c40s", "probe-sweep-4c40s",
               "capacity-4c40s-coarse", "capacity-4c40s-fine",
               "capacity-4c40s-noprobe-fine")
CONTROLLERS = ("c1", "c2", "c3", "c4")
SUMMARY_FIELDS = (
    "emitted_new_flow_rate_mean", "processed_packet_in_rate_mean",
    "processed_packet_in_rate_p95", "processed_packet_in_rate_max", "flow_mod_rate_mean",
    "cpu_mean", "cpu_p95", "response_p95_ms_mean", "qos_success_ratio",
    "flow_setup_latency_p95_ms", "snapshot_valid_ratio",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def experiment_of(run_id: str) -> str | None:
    return next((e for e in EXPERIMENTS if run_id.startswith(e + "-")), None)


def run_row(run_dir: Path) -> dict[str, Any] | None:
    """None for a directory that is not a finished run of the four sweeps."""
    experiment = experiment_of(run_dir.name)
    metadata = _load(run_dir / "metadata.json")
    summary = _load(run_dir / "summary.json")
    validation = _load(run_dir / "validation.json")
    if experiment is None or metadata is None or summary is None or validation is None:
        return None
    start = parse_dt(metadata["measurement_started_at"])
    end = parse_dt(metadata["measurement_ended_at"])
    processed: dict[str, list[float]] = defaultdict(list)
    for row in read_jsonl(run_dir / "controllers.jsonl"):
        if in_window(row, "observed_at", start, end):
            processed[str(row.get("controller_id"))].append(
                float(row["processed_packet_in_rate"]))
    probe = _load(run_dir / "setup_probe_summary.json") or {}
    conditions = _load(run_dir / "run_conditions.json") or {}
    out: dict[str, Any] = {
        "run_id": run_dir.name,
        "experiment": experiment,
        "controller": metadata["target_controller"],
        "repeat_index": metadata["repeat_index"],
        "target_new_flow_rate": metadata["target_new_flow_rate"],
        "background_rate_fps": conditions.get("background_rate_fps"),
        "setup_probe": bool(conditions.get("setup_probe")),
        "valid": bool(validation.get("valid")),
        "errors": ";".join(validation.get("errors", [])),
        "warnings": ";".join(validation.get("warnings", [])),
    }
    out.update({field: summary.get(field) for field in SUMMARY_FIELDS})
    for cid in CONTROLLERS:
        values = processed.get(cid, [])
        out[f"processed_pps_{cid}"] = statistics.mean(values) if values else None
        stats = probe.get(cid) or {}
        out[f"setup_p95_ms_{cid}"] = stats.get("setup_p95_ms")
        out[f"setup_samples_{cid}"] = stats.get("samples")
        out[f"setup_timeouts_{cid}"] = stats.get("timeouts")
    out["cpu_model"] = conditions.get("cpu_model")
    out["git_commit"] = metadata.get("git_commit")
    return out


def collect(run_root: Path) -> list[dict[str, Any]]:
    rows = [run_row(path) for path in sorted(run_root.iterdir()) if path.is_dir()]
    return [row for row in rows if row is not None]


def _stats(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "n": len(values)}


def probe_sweep_summary(rows: Iterable[Mapping[str, Any]], scale_pps: float) -> dict[str, Any]:
    """Day 3 tables, no verdict (the criteria are in evidence/week15/probe_acceptance.md).

    curves: per target controller and rate, per controller: setup P95 (mean/min/max over the
    repeats), timeouts (sum), processed pps (mean).
    probe_load: per controller, processed pps with every controller at 30 fps, probe ON
    (probe-sweep at its baseline rate) minus probe OFF (probe-off); delta_u divides by
    scale_pps, which the caller names (capacity-v4 933 pps until capacity-v5 exists).
    """
    valid = [r for r in rows if r["valid"]]
    sweep = [r for r in valid if r["experiment"] == "probe-sweep-4c40s"]
    curves: dict[str, dict[str, Any]] = defaultdict(dict)
    for (target, rate), group in sorted(_group(sweep).items()):
        curves[target][f"{rate:g}"] = {
            cid: {
                "setup_p95_ms": _stats([float(r[f"setup_p95_ms_{cid}"]) for r in group
                                        if r[f"setup_p95_ms_{cid}"] is not None]),
                "timeouts": sum(int(r[f"setup_timeouts_{cid}"] or 0) for r in group),
                "samples": sum(int(r[f"setup_samples_{cid}"] or 0) for r in group),
                "processed_pps_mean": _mean_of(group, f"processed_pps_{cid}"),
            }
            for cid in CONTROLLERS
        } | {"repeats": len(group)}
    base_rate = min((float(r["target_new_flow_rate"]) for r in sweep), default=None)
    on = [r for r in sweep if float(r["target_new_flow_rate"]) == base_rate]
    off = [r for r in valid if r["experiment"] == "probe-off-4c40s"]
    probe_load = {}
    for cid in CONTROLLERS:
        mean_on, mean_off = _mean_of(on, f"processed_pps_{cid}"), _mean_of(off, f"processed_pps_{cid}")
        delta = None if mean_on is None or mean_off is None else mean_on - mean_off
        probe_load[cid] = {"pps_probe_on": mean_on, "pps_probe_off": mean_off,
                           "delta_pps": delta,
                           "delta_u": None if delta is None else delta / scale_pps,
                           "runs_on": len(on), "runs_off": len(off)}
    return {"curves": dict(curves), "probe_load": probe_load, "scale_pps": scale_pps,
            "valid_runs": {"probe_sweep": len(sweep), "probe_off": len(off)}}


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, float], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["controller"]), float(row["target_new_flow_rate"]))].append(row)
    return groups


def _mean_of(rows: Iterable[Mapping[str, Any]], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) is not None]
    return statistics.mean(values) if values else None
