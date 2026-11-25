"""Forecast workloads on 4C40S (Week 15 Day 5): the pure part of runtime_4c40s.py --forecast.

The five 2C20S runners already take their host tables from NDT_CONTROLLERS_CONFIG
(controller_paths.py); this module adds what is 4C-specific and testable without Mininet:
  - the environment the runners must see BEFORE they are imported (their tables are built
    at import time);
  - the capacity rows they map U to offered rate with: the two capacity-v5 sweeps (coarse
    25-1600 fps and fine 30-... fps, same conditions: probe on, no background), so the Day 3
    probe runs (with a 30 fps background) never enter the mapping. The fine sweep alone starts
    at 30 fps (~352 pps), above the U = 0.40 of gradual (~337 pps): the first Day 5 queue
    failed there (20/11);
  - the run plan of the transfer test (configs/experiments/transfer_4c_v1.yaml).
"""
from __future__ import annotations

import csv
import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# workload -> (module, function) of the 2C20S runner, reused unchanged on 4C40S
RUNNERS: dict[str, tuple[str, str]] = {
    "stable": ("scripts.experiments.run_forecast_stable_2c20s", "run_forecast_stable"),
    "gradual": ("scripts.experiments.run_forecast_gradual_2c20s", "run_forecast_gradual"),
    "burst": ("scripts.experiments.run_forecast_burst_2c20s", "run_forecast_burst"),
    "oscillating": ("scripts.experiments.run_forecast_oscillating_2c20s",
                    "run_forecast_oscillating"),
    "hot_switch": ("scripts.experiments.run_forecast_hot_switch_2c20s",
                   "run_forecast_hot_switch"),
}
CAPACITY_EXPERIMENTS = ("capacity-4c40s-coarse", "capacity-4c40s-fine")
# lowest and highest U the five profiles ask for: gradual 0.40, burst 1.10
U_RANGE = (0.40, 1.10)
TOPOLOGY = "4C40S"


def runner_env(controllers_config: Path) -> dict[str, str]:
    """Set before importing a runner: host tables from the 4C config, and the local sink of
    hot-switch is p<i> (10.0.0.(100+i), the same address hs<i> has in 2C20S)."""
    return {"NDT_CONTROLLERS_CONFIG": str(controllers_config), "NDT_LOCAL_SINK_PREFIX": "p"}


def load_runner(workload: str) -> Callable[..., Path]:
    if workload not in RUNNERS:
        raise ValueError(f"unknown forecast workload {workload!r}; one of {sorted(RUNNERS)}")
    module, name = RUNNERS[workload]
    runner: Callable[..., Path] = getattr(importlib.import_module(module), name)
    return runner


def filter_capacity_rows(source: Path, target: Path,
                         experiments: tuple[str, ...] = CAPACITY_EXPERIMENTS,
                         controllers: tuple[str, ...] = ("c1", "c2", "c3", "c4")) -> int:
    """Copy the valid rows of the given experiments; every controller must keep some rows."""
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [r for r in reader
                if any(str(r.get("run_id", "")).startswith(e + "-") for e in experiments)
                and str(r.get("valid", "")).strip().lower() == "true"]
    missing = [c for c in controllers if not any(r.get("controller") == c for r in rows)]
    if missing:
        raise ValueError(f"{source}: no valid {list(experiments)} rows for {missing}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def check_u_range(capacity_runs: Path, controller_capacity: Path,
                  controllers: tuple[str, ...] = ("c1", "c2", "c3", "c4"),
                  u_range: tuple[float, float] = U_RANGE) -> dict[str, tuple[float, float]]:
    """Offered fps at the lowest and highest U for every controller; raises the mapper's
    ValueError when a target falls outside the measured range (checked before any run)."""
    from src.experiments.workloads.capacity_mapper import CapacityWorkloadMapper

    mapper = CapacityWorkloadMapper(capacity_runs, controller_capacity)
    return {c: (mapper.offered_rate_for_utilization(c, u_range[0]),
                mapper.offered_rate_for_utilization(c, u_range[1])) for c in controllers}


@dataclass(frozen=True)
class ForecastRun:
    workload: str
    seed: int
    repeat_index: int
    run_id: str


def transfer_plan(config: Mapping[str, Any]) -> list[ForecastRun]:
    """One run per workload and seed, in the order of the config (seed-major).
    The run ids start with config["run_id_prefix"] (default transfer4c-)."""
    workloads = [str(w) for w in config["forecast"]["workloads"]]
    unknown = [w for w in workloads if w not in RUNNERS]
    if unknown:
        raise ValueError(f"unknown workloads in the transfer config: {unknown}")
    seeds = [int(s) for s in config["seeds"]]
    if len(set(seeds)) != len(seeds):
        raise ValueError("forecast seeds repeat")
    prefix = str(config.get("run_id_prefix", "transfer4c-"))
    return [ForecastRun(w, seed, k, f"{prefix}{w.replace('_', '-')}-r{k:02d}-s{seed}")
            for k, seed in enumerate(seeds, start=1) for w in workloads]


def runner_kwargs(workload: str, *, capacity_runs: Path, controller_capacity: Path,
                  output_dir: Path, duration: float, sample_interval: float,
                  orchestrator_url: str, run_id: str | None, seed: int | None,
                  repeat_index: int | None, experiment_type: str | None,
                  git_commit: str | None, profiles: dict[str, str] | None,
                  target_utilization: float) -> dict[str, Any]:
    """Keyword arguments of the runner; stable alone takes target_utilization."""
    kwargs: dict[str, Any] = {
        "capacity_runs_path": capacity_runs, "controller_capacity_path": controller_capacity,
        "output_dir": output_dir, "duration_seconds": duration,
        "sampling_interval_seconds": sample_interval, "orchestrator_url": orchestrator_url,
        "run_id": run_id, "experiment_type": experiment_type, "topology": TOPOLOGY,
        "repeat_index": repeat_index, "seed": seed,
        "capacity_artifact": str(controller_capacity), "git_commit": git_commit,
        "profiles": profiles or {},
    }
    if workload == "stable":
        kwargs["target_utilization"] = target_utilization
    return kwargs


def main(argv: list[str] | None = None) -> int:
    """Print the transfer plan as workload|seed|repeat|run_id (read by the VM queue script)."""
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="transfer-test forecast plan")
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/transfer_4c_v1.yaml"))
    parser.add_argument("--check-range", nargs=2, type=Path, metavar=("RUNS_CSV", "CAPACITY"),
                        help="only check that U %s-%s maps inside the measured range" % U_RANGE)
    parser.add_argument("--capacity-experiment", action="append", default=None)
    args = parser.parse_args(argv)
    if args.check_range:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            used = Path(tmp) / "used.csv"
            experiments = tuple(args.capacity_experiment or CAPACITY_EXPERIMENTS)
            n = filter_capacity_rows(args.check_range[0], used, experiments)
            for cid, (low, high) in check_u_range(used, args.check_range[1]).items():
                print(f"RANGE OK {cid}: U {U_RANGE[0]} -> {low:.1f} fps, "
                      f"U {U_RANGE[1]} -> {high:.1f} fps ({n} rows)")
        return 0
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for run in transfer_plan(config):
        print(f"{run.workload}|{run.seed}|{run.repeat_index}|{run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
