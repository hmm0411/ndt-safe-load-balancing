"""Measure T_candidate, T_outcome and T_safety of the dry-run decision pipeline.

Input is a synthetic TwinState (no live telemetry) with the real post-migration-v1 artifact,
the real Safety Gate and the frozen policy. The forecast is a fixed payload: forecast
inference is timed by the forecast service (service_time_ms), not here. The script refuses to
write numbers unless every eligible candidate gets a READY prediction: a NOT_READY path is
faster and would make the latency look better than it is.
Output: evidence/week13/decision_stage_latency.json.
Run: PYTHONPATH=$PWD python3 scripts/decision/benchmark_decision_latency.py
"""
from __future__ import annotations

import dataclasses
import json
import platform
import statistics
from datetime import timedelta
from pathlib import Path
from typing import Any

from benchmark_candidate_latency import NOW, build_roles, build_twin

from src.orchestrator.decision_service import DecisionService
from src.orchestrator.outcome_service import capacity_version
from src.orchestrator.transaction_manager import TransactionManager
from src.twin.outcome.predictor import READY, PostMigrationOutcomePredictor

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "data" / "models" / "outcome" / "post-migration-v1"
CAPACITY = ROOT / "data" / "benchmarks" / "controller_capacity.json"
OUTPUT = ROOT / "evidence" / "week13" / "decision_stage_latency.json"
ITERATIONS = 300
WARMUP = 30
STAGES = ("t_candidate", "t_outcome", "t_safety")
# name: (controllers, switches per controller, at-risk controllers)
SCENARIOS = {
    "2c_3_candidates": (2, 3, ("c1",)),
    "2c_5_candidates": (2, 5, ("c1",)),
    "2c_10_candidates": (2, 10, ("c1",)),
    "2c_15_candidates": (2, 15, ("c1",)),
    "2c_20_candidates_both_sources": (2, 10, ("c1", "c2")),
}


class _Store:
    def __init__(self, twin: Any, roles: Any) -> None:
        self.twin, self.roles = twin, roles

    def twin_state(self) -> Any:
        return self.twin

    def role_matrix(self) -> Any:
        return self.roles


def _summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "mean_ms": round(statistics.fmean(ordered), 4),
        "p50_ms": round(ordered[len(ordered) // 2], 4),
        "p95_ms": round(ordered[int(0.95 * len(ordered)) - 1], 4),
        "max_ms": round(ordered[-1], 4),
    }


def measure(n_controllers: int, per_controller: int, at_risk: tuple[str, ...]) -> dict[str, Any]:
    twin = build_twin(n_controllers, per_controller)
    predictor = PostMigrationOutcomePredictor(ARTIFACT, capacity_version(CAPACITY))
    for i in range(4):  # four earlier snapshots 1 s apart, then the evaluated one
        past = dataclasses.replace(
            twin, snapshot_id=f"bench-{i}", created_at=NOW - timedelta(seconds=4 - i)
        )
        predictor.observe(past.to_dict())
    predictor.observe(twin.to_dict())
    payload = {
        "status": "OK", "snapshot_id": twin.snapshot_id, "model_version": "synthetic",
        "controllers": [
            {"controller_id": c.controller_id, "status": "READY",
             "overload_risk": c.controller_id in at_risk}
            for c in twin.controllers
        ],
    }
    store = _Store(twin, build_roles(twin))
    service = DecisionService(
        store, lambda: payload, predictor, TransactionManager(), clock=lambda: NOW  # type: ignore[arg-type]
    )
    samples: dict[str, list[float]] = {name: [] for name in STAGES}
    totals: list[float] = []
    first: dict[str, Any] | None = None
    for i in range(WARMUP + ITERATIONS):
        out = service.evaluate()
        if out["status"] != "OK":
            raise SystemExit(f"not benchmarking: {out}")
        if first is None:
            first = out
            ready = {e["outcome_status"] for e in out["candidates"] if e["precheck"]["eligible"]}
            if ready != {READY}:
                raise SystemExit(f"not benchmarking: outcome statuses {sorted(map(str, ready))}")
        if i >= WARMUP:
            for name in STAGES:
                samples[name].append(out["timings_ms"][name])
            totals.append(sum(out["timings_ms"][name] for name in STAGES))
    assert first is not None
    return {
        "funnel": first["funnel"],
        "rejections": first["rejections"],
        **{name: _summary(samples[name]) for name in STAGES},
        "total": _summary(totals),
    }


def main() -> None:
    result = {
        "input": "synthetic TwinState, fixed forecast payload, real outcome model and gate",
        "iterations": ITERATIONS,
        "warmup": WARMUP,
        "python": platform.python_version(),
        "capacity_version": capacity_version(CAPACITY),
        "scenarios": {name: measure(*args) for name, args in SCENARIOS.items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, data in result["scenarios"].items():
        print(name, data["funnel"], "total p95", data["total"]["p95_ms"], "ms")


if __name__ == "__main__":
    main()
