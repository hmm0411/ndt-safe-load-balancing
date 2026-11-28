"""Fault scenarios D1-D7 of the closed-loop campaign (docs/evaluation-protocol-v1.md, 4.D).

Load of every D episode (configs/experiments/campaign_4c_v1.yaml, ``faults``): c1 holds a
constant level scaled by NDT_WORKLOAD_SCALE (1.0 x 1.2 = 1.2, overloaded for the whole
episode, so Full wants to act as soon as the warm-up ends); c2-c4 hold ``dst_level``.
Faults start right after the 30 s warm-up, times are seconds after the workload start:

  d1 stale snapshot      c2's telemetry frozen for ``window_s`` (POST /api/v1/faults)
  d2 missing data        c2's telemetry dropped for ``window_s``
  d3 disconnection       c3's controller process stopped (SIGSTOP) for ``window_s``
  d4 timeout             c3 stopped, a migration c1 -> c3 is requested, c3 resumes after
                         ``timeout_freeze_s`` (> the 3 s role-request timeout)
  d5 conflict            two migrations of the same switch requested at once
  d6 destination dies    c3 stopped, a migration c1 -> c3 is requested, c3 killed (SIGKILL)
                         while the request waits; c3 stays dead
  d7 no safe destination no fault: c2-c4 at each level of ``d7_levels`` (Full and Threshold)

Expected behaviour (protocol table "Hanh vi dung"): ``check_episode``.
"""
from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.experiments.final_evaluation.campaign import EXECUTED, PlannedEpisode

SCENARIOS = ("d1", "d2", "d3", "d4", "d5", "d6")
ACTION_KINDS = ("telemetry", "signal", "migrate_frozen_target", "migrate_pair")


@dataclass(frozen=True)
class FaultAction:
    at_s: float
    kind: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FaultSpec:
    scenario: str
    dst_level: float
    c1_level: float
    window: tuple[float, float] | None
    actions: tuple[FaultAction, ...]
    # d1-d3: seconds after the fault starts before it can show (a frozen sample turns stale
    # only past freshness_max_age_ms, 2.5 s); decisions before that are not checked
    grace_s: float = 0.0


def _level(value: Any, name: str) -> float:
    level = float(value)
    if not math.isfinite(level) or not 0.0 < level <= 1.5:
        raise ValueError(f"{name} must be in (0, 1.5], got {value}")
    return level


def workload_name(scenario: str, dst_level: float | None = None) -> str:
    """Plan column 'workload' of a D episode: d1..d6, or d7-u080 for c2-c4 at 0.80."""
    if scenario == "d7":
        if dst_level is None:
            raise ValueError("d7 needs a destination level")
        return f"d7-u{round(dst_level * 100):03d}"
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown fault scenario {scenario!r}")
    return scenario


def fault_spec(workload: str, cfg: Mapping[str, Any]) -> FaultSpec:
    """The fault and load of a D episode from its plan 'workload' and the config's faults."""
    c1 = _level(cfg["c1_level"], "c1_level")
    start, length = float(cfg["start_s"]), float(cfg["window_s"])
    at = float(cfg["migrate_at_s"])
    if start < 0 or length <= 0 or at < 0:
        raise ValueError("start_s, migrate_at_s must be >= 0 and window_s > 0")
    if workload.startswith("d7-u"):
        level = int(workload[4:]) / 100.0
        if level not in [round(float(x), 2) for x in cfg["d7_levels"]]:
            raise ValueError(f"{workload}: level not in d7_levels")
        return FaultSpec("d7", _level(level, "d7 level"), c1, None, ())
    grace = float(cfg["effect_grace_s"])
    if not 0.0 <= grace < length:
        raise ValueError("effect_grace_s must be in [0, window_s)")
    dst = _level(cfg["dst_level"], "dst_level")
    end = start + length
    actions: tuple[FaultAction, ...]
    if workload == "d1":
        actions = (FaultAction(start, "telemetry", {"kind": "freeze_telemetry",
                                                    "controller_id": "c2", "seconds": length}),)
    elif workload == "d2":
        actions = (FaultAction(start, "telemetry", {"kind": "drop_telemetry",
                                                    "controller_id": "c2", "seconds": length}),)
    elif workload == "d3":
        actions = (FaultAction(start, "signal", {"controller_id": "c3", "signal": "STOP"}),
                   FaultAction(end, "signal", {"controller_id": "c3", "signal": "CONT"}))
    elif workload == "d4":
        actions = (FaultAction(at, "migrate_frozen_target", {
            "source": "c1", "target": "c3", "then": "CONT",
            "after_s": float(cfg["timeout_freeze_s"])}),)
    elif workload == "d5":
        actions = (FaultAction(at, "migrate_pair", {"source": "c1", "targets": ["c2", "c4"]}),)
    elif workload == "d6":
        actions = (FaultAction(at, "migrate_frozen_target", {
            "source": "c1", "target": "c3", "then": "KILL", "after_s": 1.0}),)
    else:
        raise ValueError(f"unknown fault workload {workload!r}")
    window = (start, end) if workload in ("d1", "d2", "d3") else (at, at + length)
    return FaultSpec(workload, dst, c1, window, actions, grace)


def plan_fault_episodes(cfg: Mapping[str, Any], order_seed: int) -> list[PlannedEpisode]:
    """D1-D6 with Full x seeds, D7 with d7_methods x d7_levels x d7_seeds (default: seeds);
    shuffled order."""
    seeds = [int(s) for s in cfg["seeds"]]
    d7_seeds = [int(s) for s in cfg.get("d7_seeds", seeds)]
    for name, values in (("seeds", seeds), ("d7_seeds", d7_seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"faults.{name} must be non-empty and unique")
    cells = [("full", s, seed) for s in SCENARIOS for seed in seeds]
    cells += [(m, workload_name("d7", float(level)), seed) for m in cfg["d7_methods"]
              for level in cfg["d7_levels"] for seed in d7_seeds]
    random.Random(order_seed).shuffle(cells)
    return [PlannedEpisode(i + 1, f"D-{m}-{w}-s{seed}", m, w, seed)
            for i, (m, w, seed) in enumerate(cells)]


def _t(value: Any, start: float) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() - start


def check_episode(spec: FaultSpec, events: Sequence[Mapping[str, Any]],
                  records: Iterable[Mapping[str, Any]], start: float) -> dict[str, Any]:
    """Expected behaviour of Full under the fault; ``events`` = the runner's faults.jsonl."""
    recs = sorted(records, key=lambda r: _t(r["decided_at"], start))
    executed = [r for r in recs if r.get("status") in EXECUTED]
    out: dict[str, Any] = {"scenario": spec.scenario, "executed": len(executed)}
    if spec.scenario in ("d1", "d2", "d3"):
        assert spec.window is not None
        lo, hi = spec.window[0] + spec.grace_s, spec.window[1]
        inside = [r for r in recs if lo <= _t(r["decided_at"], start) <= hi]
        out["decisions_in_window"] = len(inside)
        out["executed_in_window"] = sum(r.get("status") in EXECUTED for r in inside)
        out["invalid_in_window"] = sum(r.get("status") == "INVALID_SNAPSHOT" for r in inside)
        out["fault_took_effect"] = out["invalid_in_window"] > 0
        ok = out["executed_in_window"] == 0 and out["fault_took_effect"]
        if spec.scenario == "d3":
            to_c3 = [r for r in executed if (r.get("selected") or {}).get(
                "target_controller") == "c3" and _t(r["decided_at"], start) >= lo]
            out["executed_to_c3"] = len(to_c3)
            ok = ok and not to_c3
        out["pass"] = ok
        return out
    if spec.scenario in ("d4", "d6"):
        mig = [e for e in events if e.get("action") == "migrate"]
        post = [e for e in events if e.get("action") == "post_check"]
        out["http_status"] = mig[0].get("http_status") if mig else None
        out["migration_status"] = mig[0].get("status") if mig else None
        source = mig[0].get("source") if mig else None
        state = post[-1] if post else {}
        out["owner_after"] = state.get("owner")
        out["locked_after"] = state.get("locked_switches")
        out["pass"] = bool(mig and out["migration_status"] == "RESTORED"
                           and out["owner_after"] == source and not out["locked_after"])
        return out
    if spec.scenario == "d5":
        codes = sorted(int(e["http_status"]) for e in events
                       if e.get("action") == "migrate" and e.get("http_status") is not None)
        out["http_statuses"] = codes
        out["pass"] = len(codes) == 2 and codes.count(409) == 1 and codes[0] == 200
        return out
    # d7: what the method did; Full must not make a destination overloaded
    post_u = [(r.get("actual_post") or {}).get("u_dst") for r in executed]
    known = [float(u) for u in post_u if isinstance(u, (int, float))]
    out["max_u_dst_post"] = max(known) if known else None
    out["overloaded_destinations"] = sum(u >= 1.0 for u in known)
    out["pass"] = out["overloaded_destinations"] == 0
    return out
