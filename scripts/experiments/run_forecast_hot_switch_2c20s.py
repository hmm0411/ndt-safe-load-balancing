#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
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
from src.twin.capacity.model import CapacityModel
from src.experiments.workloads.hot_switch import (
    HotSwitchWorkload,
)


CONTROLLER_IDS = ("c1", "c2")

TARGET_UTILIZATION = 0.80
HOT_SHARE = 0.55

CONTROLLER_HOSTS = {
    "c1": tuple(range(1, 11)),
    "c2": tuple(range(11, 21)),
}

HOT_HOST = {
    "c1": 1,
    "c2": 11,
}


# Packet-In messages generated per new flow on the LOCAL path h_i -> s_i -> hs_i.
#
# CapacityWorkloadMapper must not be used here: it inverts the capacity
# benchmark, whose flows cross the whole 10-switch chain (~10.4 Packet-In per
# flow). A local flow crosses one switch, so the mapper's rate produced only
# ~9% utilization instead of the 80% target.
#
# 1.175 was measured on the local path at low load
# (run forecast-hot-switch-2c20s-r04: 63.2 flows/s -> 74.25 Packet-In/s).
# Re-check it with a 60 s smoke run after any controller change and override
# with NDT_HOT_SWITCH_PIN_PER_FLOW if the achieved utilization is off.
DEFAULT_LOCAL_PACKET_IN_PER_FLOW = 1.175

# Accepted |achieved - target| mean utilization for the run to count.
UTILIZATION_TOLERANCE = 0.10

# Samples skipped at the start of the run before checking achieved load.
ACHIEVED_SKIP_SECONDS = 10


def local_packet_in_per_flow() -> float:
    raw = os.environ.get("NDT_HOT_SWITCH_PIN_PER_FLOW")
    value = (
        float(raw)
        if raw
        else DEFAULT_LOCAL_PACKET_IN_PER_FLOW
    )

    if value <= 0:
        raise ValueError(
            "packet-in per flow must be > 0"
        )

    return value


def build_distribution(
    safe_capacity_pps: float,
    controller_id: str,
    packet_in_per_flow: float,
) -> dict[int, float]:
    """Per-host new-flow rates that load one controller to the target."""
    if safe_capacity_pps <= 0 or packet_in_per_flow <= 0:
        raise ValueError(
            "safe_capacity_pps and packet_in_per_flow must be > 0"
        )

    total_rate = (
        TARGET_UTILIZATION
        * safe_capacity_pps
        / packet_in_per_flow
    )

    hosts = CONTROLLER_HOSTS[controller_id]
    hot_host = HOT_HOST[controller_id]

    cold_hosts = [
        host
        for host in hosts
        if host != hot_host
    ]

    hot_rate = (
        total_rate
        * HOT_SHARE
    )

    cold_rate = (
        total_rate
        * (1.0 - HOT_SHARE)
        / len(cold_hosts)
    )

    return {
        host: (
            hot_rate
            if host == hot_host
            else cold_rate
        )
        for host in hosts
    }


def achieved_utilization(
    samples_path: Path,
    sampling_interval_seconds: float,
) -> dict[str, float]:
    """Mean measured utilization per controller, skipping the warm-up."""
    skip = int(
        ACHIEVED_SKIP_SECONDS
        / sampling_interval_seconds
    )

    by_controller: dict[str, list[float]] = {}

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            row = json.loads(line)

            by_controller.setdefault(
                str(row["controller_id"]),
                [],
            ).append(float(row["utilization"]))

    return {
        controller_id: statistics.fmean(values[skip:])
        for controller_id, values in by_controller.items()
        if len(values) > skip
    }


def check_achieved_utilization(
    achieved: dict[str, float],
) -> list[str]:
    """Problems that make the run unusable as an 80%-load hot-switch run."""
    problems: list[str] = []

    for controller_id in CONTROLLER_IDS:
        value = achieved.get(controller_id)

        if value is None:
            problems.append(
                f"{controller_id}: no utilization samples"
            )
        elif (
            abs(value - TARGET_UTILIZATION)
            > UTILIZATION_TOLERANCE
        ):
            problems.append(
                f"{controller_id}: achieved utilization "
                f"{value:.3f}, target "
                f"{TARGET_UTILIZATION:.2f} "
                f"+/- {UTILIZATION_TOLERANCE:.2f}"
            )

    return problems


def validate_local_connectivity(
    net: Any,
) -> None:
    """Verify each source can reach its local sink."""

    failures: list[str] = []

    for index in range(1, 21):
        source = net.get(f"h{index}")
        sink_ip = f"10.0.0.{100 + index}"

        result = source.cmd(
            f"ping -c 1 -W 1 {sink_ip}"
        )

        if "1 received" not in result:
            failures.append(
                f"h{index} -> hs{index} "
                f"({sink_ip})"
            )

    if failures:
        raise RuntimeError(
            "local hot-switch connectivity "
            "preflight failed: "
            + ", ".join(failures)
        )

    print(
        "HOT_SWITCH_LOCAL_CONNECTIVITY: PASS"
    )


def run_forecast_hot_switch(
    net: Any,
    *,
    capacity_runs_path: Path,
    controller_capacity_path: Path,
    output_dir: Path,
    duration_seconds: float = 60.0,
    sampling_interval_seconds: float = 1.0,
    orchestrator_url: str = (
        DEFAULT_ORCHESTRATOR_URL
    ),
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
            "hot-switch workload requires "
            "duration_seconds >= 60"
        )

    # capacity_runs_path is kept in the signature for the shared runtime CLI,
    # but the chain benchmark does not describe the local path (see above).
    capacity_model = CapacityModel.from_file(
        controller_capacity_path
    )

    packet_in_per_flow = local_packet_in_per_flow()

    distributions = {
        controller_id: build_distribution(
            capacity_model.safe_capacity(
                controller_id
            ),
            controller_id,
            packet_in_per_flow,
        )
        for controller_id in CONTROLLER_IDS
    }

    wait_for_twin_state(
        orchestrator_url
    )

    validate_local_connectivity(net)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_id = run_id or (
        "hot-switch-"
        f"{uuid.uuid4().hex[:8]}"
    )

    run_dir = (
        output_dir / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    logs_dir = (
        run_dir / "logs"
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence: dict[str, Any] = {
        "target_utilization": (
            TARGET_UTILIZATION
        ),
        "target_hot_switch_share": (
            HOT_SHARE
        ),
        "traffic_model": (
            "local-source-to-local-sink"
        ),
        "packet_in_per_flow": (
            packet_in_per_flow
        ),
        "controllers": {},
    }

    for controller_id in CONTROLLER_IDS:
        hot_host = (
            HOT_HOST[controller_id]
        )

        evidence["controllers"][
            controller_id
        ] = {
            "hot_host": f"h{hot_host}",
            "hot_switch": f"s{hot_host}",
            "local_sink": f"hs{hot_host}",
            "local_sink_ip": (
                f"10.0.0.{100 + hot_host}"
            ),
            "host_rates_fps": {
                f"h{host}": rate
                for host, rate
                in distributions[
                    controller_id
                ].items()
            },
            "total_offered_rate_fps": sum(
                distributions[
                    controller_id
                ].values()
            ),
        }

    (
        run_dir
        / "hot_switch_distribution.json"
    ).write_text(
        json.dumps(
            evidence,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    processes: list[
        subprocess.Popen[Any]
    ] = []

    try:
        # Start one sink directly attached to
        # every switch.
        for index in range(1, 21):
            sink_name = f"hs{index}"
            sink_ip = f"10.0.0.{100 + index}"

            process = start_host_process(
                net.get(sink_name),
                [
                    "python3",
                    "-m",
                    (
                        "src.experiments."
                        "workloads.udp_sink"
                    ),
                    "--bind-ip",
                    sink_ip,
                    "--port",
                    "9000",
                ],
                output_path=(
                    logs_dir
                    / f"sink_{sink_name}.log"
                ),
            )

            processes.append(
                process
            )

        time.sleep(0.5)

        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(
                    "UDP sink exited early "
                    f"with code "
                    f"{process.returncode}"
                )

        generator_processes: list[
            subprocess.Popen[Any]
        ] = []

        for controller_id in CONTROLLER_IDS:
            for (
                host_number,
                rate,
            ) in distributions[
                controller_id
            ].items():
                host_name = (
                    f"h{host_number}"
                )

                source_ip = (
                    f"10.0.0.{host_number}"
                )

                target_host = (
                    f"hs{host_number}"
                )

                target_ip = (
                    f"10.0.0.{100 + host_number}"
                )

                output_path = (
                    run_dir
                    / (
                        "workload_"
                        f"{controller_id}_"
                        f"{host_name}.jsonl"
                    )
                )

                process = (
                    start_host_process(
                        net.get(host_name),
                        [
                            "python3",
                            "-m",
                            (
                                "src.experiments."
                                "workloads."
                                "udp_new_flow"
                            ),
                            "--run-id",
                            (
                                f"{run_id}-"
                                f"{controller_id}-"
                                f"{host_name}"
                            ),
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
                            str(rate),
                            "--duration",
                            str(
                                duration_seconds
                            ),
                            "--sample-interval",
                            str(
                                sampling_interval_seconds
                            ),
                            "--output",
                            str(output_path),
                        ],
                        output_path=(
                            logs_dir
                            / (
                                "generator_"
                                f"{controller_id}_"
                                f"{host_name}.log"
                            )
                        ),
                    )
                )

                processes.append(
                    process
                )

                generator_processes.append(
                    process
                )

        time.sleep(0.5)

        for process in generator_processes:
            if process.poll() is not None:
                raise RuntimeError(
                    "hot-switch generator "
                    "exited early with code "
                    f"{process.returncode}"
                )

        profile = HotSwitchWorkload(
            target_utilization=(
                TARGET_UTILIZATION
            ),
            hot_switch_id="s1",
            hot_switch_share=HOT_SHARE,
        )

        runner = ForecastDataRunner(
            config=ForecastRunConfig(
                workload_type="hot-switch",
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
            workload_step=(
                profile.target_at
            ),
        )

        result = runner.run()

        for process in generator_processes:
            try:
                process.wait(
                    timeout=2.0
                )
            except subprocess.TimeoutExpired:
                pass

        achieved = achieved_utilization(
            result / "forecast_samples.jsonl",
            sampling_interval_seconds,
        )

        problems = check_achieved_utilization(
            achieved
        )

        evidence["achieved_mean_utilization"] = achieved
        evidence["achieved_load_valid"] = not problems

        (
            run_dir
            / "hot_switch_distribution.json"
        ).write_text(
            json.dumps(
                evidence,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        if problems:
            raise RuntimeError(
                "hot-switch run did not reach its target load "
                "(adjust NDT_HOT_SWITCH_PIN_PER_FLOW): "
                + "; ".join(problems)
            )

        return result

    finally:
        for process in reversed(
            processes
        ):
            terminate_process(
                process
            )


if __name__ == "__main__":
    raise SystemExit(
        "Use runtime_2c20s.py "
        "--forecast-hot-switch"
    )
