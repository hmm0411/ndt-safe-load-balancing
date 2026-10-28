#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from src.experiments.runners.forecast_data_runner import (
    ForecastDataRunner,
    ForecastRunConfig,
)
from src.experiments.workloads.capacity_mapper import CapacityWorkloadMapper
from src.experiments.workloads.profile_spec import stable_utilizations
from src.experiments.workloads.stable import StableWorkload


CONTROLLER_IDS = ("c1", "c2")

SINKS = (
    ("h10", "10.0.0.10", 9000),
    ("h20", "10.0.0.20", 9000),
)

GENERATORS = (
    ("h1", "10.0.0.1", "h10", "10.0.0.10"),
    ("h11", "10.0.0.11", "h20", "10.0.0.20"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORCHESTRATOR_URL = "http://127.0.0.1:9000"


def get_twin_state(
    orchestrator_url: str = DEFAULT_ORCHESTRATOR_URL,
) -> dict[str, Any]:
    response = requests.get(
        f"{orchestrator_url.rstrip('/')}/api/v1/twin/state",
        timeout=2.0,
    )
    response.raise_for_status()

    payload = response.json()

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Twin State response must be a JSON object"
        )

    if payload.get("status") == "NOT_READY":
        raise RuntimeError("Twin State is not ready")

    return payload


def wait_for_twin_state(
    orchestrator_url: str,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Wait until the orchestrator exposes a valid NDT TwinState."""
    deadline = time.monotonic() + timeout_seconds
    last_error = "Twin State is not ready"

    while time.monotonic() < deadline:
        try:
            state = get_twin_state(orchestrator_url)

            quality = state.get("quality")
            if not isinstance(quality, dict):
                last_error = "Twin State has no quality object"
            elif not bool(quality.get("valid", False)):
                last_error = (
                    "Twin State is not valid: "
                    f"snapshot_id={state.get('snapshot_id')}, "
                    f"twinning_rate={quality.get('twinning_rate')}, "
                    f"completeness_ratio="
                    f"{quality.get('completeness_ratio')}, "
                    f"consistent={quality.get('consistent')}"
                )
            else:
                return state

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
        ) as exc:
            last_error = str(exc)

        time.sleep(0.5)

    raise RuntimeError(
        f"timed out waiting for valid Twin State: {last_error}"
    )


def terminate_process(
    process: subprocess.Popen[Any] | None,
) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def start_host_process(
    host: Any,
    arguments: list[str],
    *,
    output_path: Path | None = None,
) -> subprocess.Popen[Any]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)

    stdout: Any = subprocess.DEVNULL

    if output_path is not None:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        stdout = output_path.open(
            "w",
            encoding="utf-8",
        )

    return host.popen(
        arguments,
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )


def run_forecast_stable(
    net: Any,
    *,
    capacity_runs_path: Path,
    controller_capacity_path: Path,
    output_dir: Path,
    duration_seconds: float = 60.0,
    sampling_interval_seconds: float = 1.0,
    target_utilization: float = 0.60,
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
    mapper = CapacityWorkloadMapper(
        capacity_runs_path=capacity_runs_path,
        controller_capacity_path=controller_capacity_path,
    )

    utilizations = stable_utilizations(
        profiles or {}, target_utilization, CONTROLLER_IDS
    )
    offered_rates = {
        controller_id: mapper.offered_rate_for_utilization(
            controller_id,
            utilizations[controller_id],
        )
        for controller_id in CONTROLLER_IDS
    }

    # Phases (early/middle/late) only in the new per-controller mode;
    # without --profile the phase stays "stable" as in Week 10.
    stable_profile = StableWorkload(
        target_utilization,
        duration_seconds=duration_seconds if profiles else None,
    )

    processes: list[subprocess.Popen[Any]] = []

    wait_for_twin_state(orchestrator_url)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = run_id or f"stable-{uuid.uuid4().hex[:8]}"
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

    try:
        # Start UDP sinks.
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
                    logs_dir /
                    f"sink_{host_name}.log"
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

        # Start one workload generator per controller.
        for index, (
            host_name,
            source_ip,
            target_host,
            target_ip,
        ) in enumerate(GENERATORS):
            controller_id = CONTROLLER_IDS[index]

            process = start_host_process(
                net.get(host_name),
                [
                    "python3",
                    "-m",
                    "src.experiments.workloads.udp_new_flow",
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
                    "--pattern",
                    "stable",
                    "--rate",
                    str(offered_rates[controller_id]),
                    "--duration",
                    str(duration_seconds),
                    "--sample-interval",
                    str(sampling_interval_seconds),
                    "--output",
                    str(
                        run_dir /
                        f"workload_{controller_id}.jsonl"
                    ),
                ],
                output_path=(
                    logs_dir /
                    f"generator_{controller_id}.log"
                ),
            )

            processes.append(process)

        time.sleep(0.25)

        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    "forecast process exited early "
                    f"with code {process.returncode}"
                )

        runner = ForecastDataRunner(
            config=ForecastRunConfig(
                workload_type="stable",
                duration_seconds=duration_seconds,
                sampling_interval_seconds=(
                    sampling_interval_seconds
                ),
                controller_ids=CONTROLLER_IDS,
                output_dir=output_dir,
                run_id=run_id,
                experiment_type=experiment_type,
                topology=topology,
                repeat_index=repeat_index,
                seed=seed,
                capacity_artifact=capacity_artifact,
                git_commit=git_commit,
            ),
            twin_state_provider=lambda: get_twin_state(
                orchestrator_url
            ),
            workload_step=stable_profile.target_at,
        )

        result = runner.run()

        # Give generators a short opportunity to finish cleanly.
        for process in processes[len(SINKS):]:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass

        return result

    finally:
        for process in reversed(processes):
            terminate_process(process)
