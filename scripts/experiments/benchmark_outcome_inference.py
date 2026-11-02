#!/usr/bin/env python3
"""Inference latency of the runtime what-if predictor (Week 12, Day 5).

Measures the path the endpoint uses: PostMigrationOutcomePredictor.predict() on a
TwinState with a full 5-snapshot history: build_features, Analytical, residual models,
Hybrid, derived R95 / imbalance, margins. The model is loaded BEFORE timing (one
untimed call), so file reading is never part of the numbers. One timed unit is the
evaluation of N candidates, one predict() call per candidate (N = 1 and 10):

    python3 scripts/experiments/benchmark_outcome_inference.py

Run it in a quiet testbed window (nobody training or running a workload). It refuses
to write numbers unless every prediction is READY: a NOT_READY path is much faster and
would make the latency look better than it is. The input is a synthetic TwinState of
2 controllers x 10 switches; it fixes the size of the computation, it is not a workload.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.experiments.migration import fixtures as fx
from src.experiments.migration.schema import MigrationAction
from src.orchestrator.outcome_service import capacity_version
from src.twin.forecasting.selection import measure_latency_ms
from src.twin.outcome import predictor as p

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = Path("data/models/outcome/post-migration-v1")
DEFAULT_CAPACITY = Path("data/benchmarks/controller_capacity.json")
DEFAULT_OUTPUT = Path("evidence/week12/outcome_inference_latency.json")
MAX_CANDIDATES = 10  # c1 owns s1..s10 in the 2C20S state used here
MIN_REPEATS = 1000  # plan Week 12, Day 5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--capacity-file", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--candidates", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_average() -> list[float] | None:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _states(count: int = 5) -> list[dict[str, Any]]:
    return [fx.make_twin_state({"c1": 0.8, "c2": 0.3}, index=i) for i in range(count)]


def _actions(count: int) -> list[MigrationAction]:
    return [
        MigrationAction(
            action_type="MIGRATE",
            source_controller="c1",
            target_controller="c2",
            switch_id=f"s{k}",
        )
        for k in range(1, count + 1)
    ]


def build_predictor(
    artifact_dir: Path, capacity_file: Path
) -> tuple[p.PostMigrationOutcomePredictor, dict[str, Any]]:
    """Predictor with a full history and the model loaded (one untimed call)."""
    predictor = p.PostMigrationOutcomePredictor(artifact_dir, capacity_version(capacity_file))
    states = _states()
    for state in states:
        predictor.observe(state)
    first = predictor.predict(states[-1], _actions(1)[0], observe=False)
    if first.status != p.READY:
        raise SystemExit(
            f"not benchmarking: the first prediction is {first.status} / {first.reason} "
            f"(artifact {artifact_dir}, capacity {capacity_file})"
        )
    return predictor, states[-1]


def measure(
    predictor: p.PostMigrationOutcomePredictor,
    state: dict[str, Any],
    candidates: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    if not 1 <= candidates <= MAX_CANDIDATES:
        raise ValueError(f"candidates must be between 1 and {MAX_CANDIDATES}")
    actions = _actions(candidates)

    def evaluate_all() -> None:
        for action in actions:
            result = predictor.predict(state, action, observe=False)
            if result.status != p.READY:  # fail closed: a fast NOT_READY is not a result
                raise SystemExit(f"prediction became {result.status} / {result.reason}")

    stats = measure_latency_ms(evaluate_all, warmup=warmup, repeats=repeats)
    stats["ms_per_candidate_mean"] = stats["mean_ms"] / candidates
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.repeats < MIN_REPEATS:
        raise SystemExit(f"--repeats must be >= {MIN_REPEATS} (plan Week 12, Day 5)")
    artifact_dir = _resolve(args.artifact_dir)
    capacity_file = _resolve(args.capacity_file)
    predictor, state = build_predictor(artifact_dir, capacity_file)
    manifest = json.loads((artifact_dir / "training_manifest.json").read_text(encoding="utf-8"))

    load_before = _load_average()
    results = {
        str(n): measure(predictor, state, n, args.warmup, args.repeats)
        for n in args.candidates
    }
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    payload = {
        "schema_version": "1.0",
        "model_version": manifest.get("model_version"),
        "capacity_version": capacity_version(capacity_file),
        "method": "predict_per_candidate_sequential_perf_counter_ns",
        "model_preloaded": True,
        "input": "synthetic TwinState, 2 controllers x 10 switches, 5-snapshot history",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "candidates": {
            n: {key: round(value, 6) for key, value in stats.items()}
            for n, stats in results.items()
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": _load_average(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_tracked_changes": dirty,
            "measured_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for n, stats in results.items():
        print(
            f"{n:>3} candidate(s): mean={stats['mean_ms']:.4f} ms  p50={stats['p50_ms']:.4f} ms  "
            f"p95={stats['p95_ms']:.4f} ms  p99={stats['p99_ms']:.4f} ms  "
            f"({stats['ms_per_candidate_mean']:.4f} ms per candidate)"
        )
    print("Wrote", output)


if __name__ == "__main__":
    main()
