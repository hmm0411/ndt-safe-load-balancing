#!/usr/bin/env python3
"""4C40S sweep driver (Week 15 Day 3 probe sweep, Day 4 capacity-v5). VM only, sudo.

One process = one target controller and one rate: it builds the 4C40S topology,
waits for every switch-controller connection, runs init-roles, then runs the
repeats of that (controller, rate) one after another, like capacity-v4 ran them.

A repeat whose run directory already holds a valid run is skipped. A failed or
incomplete one is reported and left untouched (never retried here). A repeat that
fails does not stop the others: the exit code is 1 when any repeat failed.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

from src.experiments.capacity.capacity_runner import run_capacity_experiment
from src.experiments.capacity.sweep_4c40s import (
    background_rate,
    plan_runs,
    run_status,
    setup_probe_spec,
    sweep_controllers,
    validate_sweep_config,
)
from src.experiments.common.config import load_yaml, validate_capacity_config
from src.experiments.topologies.runtime_4c40s import (
    Runtime4C40STopo,
    controller_specs,
    initialize_runtime,
)


def machine_conditions() -> dict[str, Any]:
    """What capacity depends on and the v4 fingerprint did not record (CPU model)."""
    model = "unknown"
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in lscpu.splitlines():
            if line.startswith("Model name:"):
                model = line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    mem_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
    except OSError:
        pass
    return {"cpu_model": model, "cpu_count": os.cpu_count(), "mem_total_kb": mem_kb,
            "kernel": platform.release()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controllers-config", type=Path,
                        default=Path("configs/controllers_4c40s.yaml"))
    parser.add_argument("--controller", required=True)
    parser.add_argument("--rate", type=float, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path.cwd()
    config = load_yaml(repo_root / args.config)
    controllers_config = load_yaml(repo_root / args.controllers_config)
    validate_capacity_config(config)
    validate_sweep_config(config, controllers_config)
    if args.controller not in sweep_controllers(config, controllers_config):
        raise SystemExit(f"{args.controller} is not a target controller of this sweep")
    if float(args.rate) not in [float(r) for r in config["workload"]["rates_fps"]]:
        raise SystemExit(f"rate {args.rate} is not in workload.rates_fps")

    output_root = repo_root / str(config["experiment"]["output_root"])
    todo = []
    for run in plan_runs(config, [args.controller], [args.rate]):
        status = run_status(output_root / run.run_id)
        if status == "missing":
            todo.append(run)
        elif status == "valid":
            print(f"SKIP VALID: {run.run_id}", flush=True)
        else:
            print(f"SKIP {status.upper()} (move it to data/experiment_runs_invalid/ to rerun):"
                  f" {run.run_id}", flush=True)
    if not todo:
        return 0

    conditions = machine_conditions()
    print("RUN_CONDITIONS:", conditions, flush=True)
    setLogLevel("info")
    net = Mininet(topo=Runtime4C40STopo(), controller=None, switch=OVSSwitch,
                  link=TCLink, build=False)
    for cid, (port, _) in controller_specs(controllers_config).items():
        net.addController(cid, controller=RemoteController, ip="127.0.0.1", port=port)
    failed = 0
    try:
        net.build()
        net.start()
        initialize_runtime(controllers_config, str(config["runtime"]["orchestrator_url"]))
        for run in todo:
            try:
                run_dir = run_capacity_experiment(
                    net=net, config=config, controller_id=run.controller, rate=run.rate,
                    repeat_index=run.repeat, repo_root=repo_root,
                    background_rate=background_rate(config),
                    setup_probe=setup_probe_spec(config),
                    run_conditions=conditions,
                )
                print("PASS:", run_dir, flush=True)
            except Exception as exc:  # one failed repeat must not stop the others
                failed += 1
                print(f"FAIL: {run.run_id}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        net.stop()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
