#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from scripts.experiments.run_forecast_stable_2c20s import (
    DEFAULT_ORCHESTRATOR_URL,
    get_twin_state,
    start_host_process,
    terminate_process,
    wait_for_twin_state,
)
from src.experiments.runners.forecast_data_runner import (
    ForecastDataRunner,
    ForecastRunConfig,
)
from src.experiments.workloads.capacity_mapper import (
    CapacityWorkloadMapper,
)
from src.experiments.workloads.oscillating import (
    OscillatingWorkload,
)
from src.experiments.workloads.profile_spec import controller_modes


CONTROLLER_IDS = ("c1", "c2")

SINKS = (
    ("h10", "10.0.0.10", 9000),
    ("h20", "10.0.0.20", 9000),
)

GENERATORS = (
    ("h1", "10.0.0.1", "h10", "10.0.0.10"),
    ("h11", "10.0.0.11", "h20", "10.0.0.20"),
)


def build_rate_map(
    mapper: CapacityWorkloadMapper,
    controller_id: str,
) -> dict[str, float]:
    # Dense map so the dynamic generator can interpolate the
    # continuously changing sinusoidal target.
    rate_map: dict[str, float] = {}

    for step in range(50, 91, 5):
        utilization = step / 100.0

        rate_map[f"{utilization:.2f}"] = (
            mapper.offered_rate_for_utilization(
                controller_id,
                utilization,
            )
        )

    return rate_map


def run_forecast_oscillating(
    net: Any,
    *,
    capacity_runs_path: Path,
    controller_capacity_path: Path,
    output_dir: Path,
    duration_seconds: float = 60.0,
    sampling_interval_seconds: float = 1.0,
    profiles: dict[str, str] | None = None,
    orchestrator_url: str = DEFAULT_ORCHESTRATOR_URL,
    run_id: str | None = None,
    experiment_type: str | None = None,
    topology: str | None = None,
    repeat_index: int | None = None,
    seed: int | None = None,
    capacity_artifact: str | None = None,
    git_commit: str | None = None,
) -> Path:
    if duration_seconds < 60.0:
        raise ValueError(
            "oscillating workload requires "
            "duration_seconds >= 60"
        )

    mapper = CapacityWorkloadMapper(
        capacity_runs_path=capacity_runs_path,
        controller_capacity_path=controller_capacity_path,
        range_policy="reject",
    )

    # Phases (rising/peak/falling) only in the new per-controller mode;
    # without --profile the phase stays "oscillating" as in Week 10.
    profile = OscillatingWorkload(phased=bool(profiles))

    rate_maps = {
        controller_id: build_rate_map(
            mapper,
            controller_id,
        )
        for controller_id in CONTROLLER_IDS
    }

    wait_for_twin_state(orchestrator_url)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = run_id or (
        f"oscillating-{uuid.uuid4().hex[:8]}"
    )

    run_dir = output_dir / run_id

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    logs_dir = run_dir / "logs"

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rate_map_paths: dict[str, Path] = {}

    for controller_id in CONTROLLER_IDS:
        path = (
            run_dir
            / f"rate_map_{controller_id}.json"
        )

        path.write_text(
            json.dumps(
                rate_maps[controller_id],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        rate_map_paths[controller_id] = path

    modes = controller_modes(profiles or {}, "oscillating", CONTROLLER_IDS)

    processes: list[subprocess.Popen[Any]] = []

    try:
        for host_name, bind_ip, port in SINKS:
            process = start_host_process(
                net.get(host_name),
                [
                    "python3",
                    "-m",
                    "src.experiments.workloads.udp_sink",
                    "--bind-ip",
                    bind_ip,
                    "--port",
                    str(port),
                ],
                output_path=(
                    logs_dir
                    / f"sink_{host_name}.log"
                ),
            )

            processes.append(process)

        time.sleep(0.25)

        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    "UDP sink exited early "
                    f"with code {process.returncode}"
                )

        for index, (
            host_name,
            source_ip,
            target_host,
            target_ip,
        ) in enumerate(GENERATORS):
            controller_id = CONTROLLER_IDS[index]
            level = modes[controller_id]
            if level is None:
                generator_module = "src.experiments.workloads.udp_dynamic_flow"
                pattern_args = [
                    "--pattern",
                    "oscillating",
                    "--rate-map",
                    str(rate_map_paths[controller_id]),
                ]
            else:
                generator_module = "src.experiments.workloads.udp_new_flow"
                pattern_args = [
                    "--pattern",
                    "stable",
                    "--rate",
                    str(mapper.offered_rate_for_utilization(controller_id, level)),
                ]

            process = start_host_process(
                net.get(host_name),
                [
                    "python3",
                    "-m",
                    generator_module,
                    "--run-id",
                    f"{run_id}-{controller_id}",
                    "--source-host",
                    host_name,
                    "--source-ip",
                    source_ip,
                    "--target-host",
                    target_host,
                    "--target-ip",
                    target_ip,
                    "--target-port",
                    "9000",
                    *pattern_args,
                    "--duration",
                    str(duration_seconds),
                    "--sample-interval",
                    str(
                        sampling_interval_seconds
                    ),
                    "--output",
                    str(
                        run_dir
                        / (
                            "workload_"
                            f"{controller_id}.jsonl"
                        )
                    ),
                ],
                output_path=(
                    logs_dir
                    / (
                        "generator_"
                        f"{controller_id}.log"
                    )
                ),
            )

            processes.append(process)

        time.sleep(0.25)

        for process in processes[len(SINKS):]:
            if process.poll() is not None:
                raise RuntimeError(
                    "oscillating generator "
                    "exited early "
                    f"with code "
                    f"{process.returncode}"
                )

        runner = ForecastDataRunner(
            config=ForecastRunConfig(
                workload_type="oscillating",
                duration_seconds=(
                    duration_seconds
                ),
                sampling_interval_seconds=(
                    sampling_interval_seconds
                ),
                controller_ids=(
                    CONTROLLER_IDS
                ),
                output_dir=output_dir,
                run_id=run_id,
                experiment_type=experiment_type,
                topology=topology,
                repeat_index=repeat_index,
                seed=seed,
                capacity_artifact=capacity_artifact,
                git_commit=git_commit,
            ),
            twin_state_provider=(
                lambda: get_twin_state(
                    orchestrator_url
                )
            ),
            workload_step=profile.target_at,
        )

        result = runner.run()

        for process in processes[len(SINKS):]:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        return result

    finally:
        for process in reversed(processes):
            terminate_process(process)


if __name__ == "__main__":
    raise SystemExit(
        "Use runtime_2c20s.py "
        "--forecast-oscillating"
    )
