"""Measure candidate generation and structural pre-check latency.

Input is a synthetic TwinState (no live telemetry): it isolates the cost of
the two pure-Python steps. Output: evidence/week13/candidate_latency.json.
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.schemas.snapshot import OwnershipState
from src.schemas.telemetry import SwitchTelemetry
from src.schemas.twin import ControllerTwinState, TwinQuality, TwinState
from src.twin.decision.candidate_generator import generate_candidates
from src.twin.decision.candidate_precheck import PrecheckContext, precheck_all
from src.twin.decision.safety_policy import load_safety_policy

OUTPUT = Path("evidence/week13/candidate_latency.json")
NOW = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
ITERATIONS = 2000
WARMUP = 200
SCENARIOS = {
    "2c20s_one_source": (2, 10, ("c1",)),
    "2c20s_both_sources": (2, 10, ("c1", "c2")),
    "4c20s_one_source": (4, 5, ("c1",)),
    "4c20s_all_sources": (4, 5, ("c1", "c2", "c3", "c4")),
}


def build_twin(n_controllers: int, per_controller: int) -> TwinState:
    cids = [f"c{i + 1}" for i in range(n_controllers)]
    owners = {}
    for i, cid in enumerate(cids):
        for j in range(per_controller):
            owners[f"s{i * per_controller + j + 1}"] = cid
    controllers = [
        ControllerTwinState(cid, 800.0, 933.0, 0.86, 1.0, 0.0, 10.0, 100.0, 1.0, per_controller)
        for cid in cids
    ]
    switches = [
        SwitchTelemetry(sid, cid, NOW, 0, 0.0, 0, 0.0, 0, 0.0, 1.0 / per_controller)
        for sid, cid in owners.items()
    ]
    ownership = [OwnershipState(s, c, "MASTER", "SLAVE", 0, 1) for s, c in owners.items()]
    quality = TwinQuality(10.0, 1.0, 1.0, 1.0, True, True)
    return TwinState("bench", NOW, 1, 1, controllers, switches, ownership, quality)


def build_roles(twin: TwinState) -> dict:
    cids = [c.controller_id for c in twin.controllers]
    return {
        (o.switch_id, c): {"role": "MASTER" if c == o.owner_controller_id else "SLAVE",
                           "connected": True}
        for o in twin.ownership
        for c in cids
    }


def _summary(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": round(statistics.fmean(ordered), 4),
        "p50_ms": round(ordered[len(ordered) // 2], 4),
        "p95_ms": round(ordered[int(0.95 * len(ordered)) - 1], 4),
        "max_ms": round(ordered[-1], 4),
    }


def measure(n_controllers: int, per_controller: int, at_risk: tuple[str, ...]) -> dict:
    twin = build_twin(n_controllers, per_controller)
    cooldown = load_safety_policy().cooldown_seconds.value
    ctx = PrecheckContext(frozenset(at_risk), build_roles(twin), frozenset(), None, cooldown, NOW)
    generate_ms: list[float] = []
    precheck_ms: list[float] = []
    count = 0
    for i in range(WARMUP + ITERATIONS):
        t0 = time.perf_counter()
        candidates = generate_candidates(twin, at_risk, NOW).candidates
        t1 = time.perf_counter()
        precheck_all(candidates, twin, ctx)
        t2 = time.perf_counter()
        count = len(candidates)
        if i >= WARMUP:
            generate_ms.append((t1 - t0) * 1000)
            precheck_ms.append((t2 - t1) * 1000)
    total = [a + b for a, b in zip(generate_ms, precheck_ms)]
    return {
        "raw_candidates": count,
        "generate": _summary(generate_ms),
        "precheck": _summary(precheck_ms),
        "total": _summary(total),
    }


def main() -> None:
    result = {
        "input": "synthetic TwinState, no live telemetry",
        "iterations": ITERATIONS,
        "warmup": WARMUP,
        "python": platform.python_version(),
        "scenarios": {name: measure(*args) for name, args in SCENARIOS.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, data in result["scenarios"].items():
        print(name, data["raw_candidates"], "candidates, total p95", data["total"]["p95_ms"], "ms")


if __name__ == "__main__":
    main()
