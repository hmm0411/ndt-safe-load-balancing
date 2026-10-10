from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.request import urlopen

from src.experiments.capacity.aggregator import aggregate_run
from src.experiments.capacity.collector import RunCollector
from src.experiments.capacity.validator import validate_summary
from src.experiments.common.run_io import (
    create_run_dir,
    git_commit,
    make_run_id,
    restore_sudo_owner,
    utc_now,
    write_json,
)
from src.schemas.experiment import ExperimentMetadata

COOKIE_REACTIVE = "0x10"
COOKIE_VERIFICATION = "0x20"
COOKIE_BENCHMARK = "0x30"
COOKIE_MASK = "0xffffffffffffffff"
EXPERIMENT_COOKIES = (
    COOKIE_REACTIVE,
    COOKIE_VERIFICATION,
    COOKIE_BENCHMARK,
)


def get_json(url: str, timeout_seconds: float = 3.0) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout_seconds) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return payload


def validate_runtime_state(config: Dict[str, Any], controller_id: str) -> Dict[str, Any]:
    runtime = config["runtime"]
    scenario = config["scenarios"][controller_id]
    for current_controller, base_url in runtime["controller_urls"].items():
        payload = get_json(f"{str(base_url).rstrip('/')}/api/v1/telemetry")
        if not isinstance(payload.get("controller"), dict):
            raise RuntimeError(f"controller telemetry missing for {current_controller}")

    state = get_json(f"{str(runtime['orchestrator_url']).rstrip('/')}/api/v1/state")
    ownership = state.get("ownership")
    if not isinstance(ownership, dict):
        raise RuntimeError("orchestrator state does not contain ownership mapping")
    for switch_name in scenario["path_switches"]:
        if ownership.get(switch_name) != controller_id:
            raise RuntimeError(
                f"{switch_name} owner is {ownership.get(switch_name)}, expected {controller_id}"
            )
    return state


def wait_for_valid_snapshot(orchestrator_url: str, timeout_seconds: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_reason = "no valid snapshot available"
    while time.monotonic() < deadline:
        try:
            snapshot = get_json(
                f"{orchestrator_url.rstrip('/')}/api/v1/state"
            ).get("latest_snapshot")
            if isinstance(snapshot, dict) and snapshot.get("quality", {}).get("valid") is True:
                return snapshot
            last_reason = "latest snapshot is not valid"
        except Exception as exc:
            last_reason = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for valid snapshot: {last_reason}")


def clear_experiment_flows(switches: Iterable[str]) -> None:
    for switch_name in switches:
        for cookie in EXPERIMENT_COOKIES:
            subprocess.run(
                [
                    "ovs-ofctl",
                    "-O",
                    "OpenFlow13",
                    "del-flows",
                    switch_name,
                    f"cookie={cookie}/{COOKIE_MASK}",
                ],
                check=True,
                text=True,
                capture_output=True,
            )


def count_cookie_flows(switch_name: str, cookie: str) -> int:
    result = subprocess.run(
        [
            "ovs-ofctl",
            "-O",
            "OpenFlow13",
            "dump-flows",
            switch_name,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    # Match the requested cookie exactly at the cookie field.
    needle = f"cookie={cookie}"
    return sum(needle in line for line in result.stdout.splitlines())


def wait_for_cookie_absent(
    switches: Iterable[str],
    cookie: str,
    timeout_seconds: float,
) -> None:
    switch_list = list(switches)

    deadline = (
        time.monotonic()
        + timeout_seconds
    )

    remaining: dict[str, int] = {}

    while time.monotonic() < deadline:
        remaining = {
            switch_name:
                count_cookie_flows(
                    switch_name,
                    cookie,
                )
            for switch_name
            in switch_list
        }

        if all(
            count == 0
            for count in remaining.values()
        ):
            return

        time.sleep(0.25)

    raise RuntimeError(
        f"cookie {cookie} flows still present "
        f"after cleanup: {remaining}"
    )

def wait_for_experiment_flows_absent(
    switches: Iterable[str],
    timeout_seconds: float,
) -> None:
    switch_list = list(switches)

    for cookie in EXPERIMENT_COOKIES:
        wait_for_cookie_absent(
            switches=switch_list,
            cookie=cookie,
            timeout_seconds=timeout_seconds,
        )

def warmup_hosts(source: Any, target: Any) -> None:
    source_result = source.cmd(f"ping -c 2 {target.IP()}")
    target_result = target.cmd(f"ping -c 2 {source.IP()}")
    if "0% packet loss" not in source_result or "0% packet loss" not in target_result:
        raise RuntimeError("bidirectional host warmup ping failed")


def terminate_process(process: Optional[Any], timeout_seconds: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)


def ensure_process_running(process: Any, name: str) -> None:
    if process.poll() is not None:
        raise RuntimeError(f"{name} exited early with return code {process.returncode}")


def process_command(module: str, *arguments: str) -> list[str]:
    return ["python3", "-m", module, *arguments]

def experiment_flow_counts(
    switches: Iterable[str],
) -> dict[str, dict[str, int]]:
    switch_list = list(
        switches
    )

    return {
        cookie: {
            switch_name:
                count_cookie_flows(
                    switch_name,
                    cookie,
                )
            for switch_name
            in switch_list
        }
        for cookie
        in EXPERIMENT_COOKIES
    }

def run_capacity_experiment(
    net: Any,
    config: Dict[str, Any],
    controller_id: str,
    rate: float,
    repeat_index: int,
    repo_root: Path,
) -> Path:
    if controller_id not in config["scenarios"]:
        raise ValueError(f"unknown controller scenario: {controller_id}")
    if rate <= 0 or repeat_index <= 0:
        raise ValueError("rate must be > 0 and repeat_index must be > 0")
    if config["migration"].get("enabled", True):
        raise ValueError("migration must be disabled during capacity benchmark")

    scenario = config["scenarios"][controller_id]
    workload_config = config["workload"]
    timing = config["timing"]
    qos_config = config["qos_probe"]
    telemetry_config = config["telemetry"]
    experiment = config["experiment"]
    runtime = config["runtime"]
    warmup = float(timing["warmup_seconds"])
    measurement = float(timing["measurement_seconds"])
    cooldown = float(timing["cooldown_seconds"])
    workload_duration = warmup + measurement
    run_id = make_run_id(str(experiment["type"]), controller_id, rate, repeat_index)
    run_dir = create_run_dir(repo_root / str(experiment["output_root"]), run_id)

    source = net.get(scenario["source_host"])
    target = net.get(scenario["target_host"])
    collector = None
    processes: list[Any] = []
    streams: list[Any] = []
    try:
        runtime_state = validate_runtime_state(config, controller_id)
        ownership = runtime_state["ownership"]
        controller_count = len(runtime["controller_urls"])
        switch_count = len(ownership)
        metadata = ExperimentMetadata(
            schema_version=str(config["schema_version"]),
            run_id=run_id,
            experiment_type=str(experiment["type"]),
            target_controller=controller_id,
            topology=str(experiment["topology"]),
            controller_count=controller_count,
            switch_count=switch_count,
            workload_pattern=str(workload_config["pattern"]),
            target_new_flow_rate=float(rate),
            source_host=str(scenario["source_host"]),
            source_ip=str(scenario["source_ip"]),
            target_host=str(scenario["target_host"]),
            target_ip=str(scenario["target_ip"]),
            protocol=str(workload_config["protocol"]),
            target_port=int(workload_config["target_port"]),
            workload_sample_interval_seconds=float(workload_config["sample_interval_seconds"]),
            benchmark_idle_timeout_seconds=float(workload_config["flow_idle_timeout_seconds"]),
            telemetry_interval_seconds=float(telemetry_config["interval_seconds"]),
            warmup_seconds=warmup,
            measurement_seconds=measurement,
            cooldown_seconds=cooldown,
            repeat_index=repeat_index,
            seed=repeat_index,
            git_commit=git_commit(repo_root),
            started_at=utc_now(),
        )
        metadata_path = run_dir / "metadata.json"
        write_json(metadata_path, metadata.to_dict())

        wait_for_valid_snapshot(str(runtime["orchestrator_url"]))
        warmup_hosts(source, target)
        clear_experiment_flows(scenario["path_switches"])

        wait_for_experiment_flows_absent(
                switches=scenario[
                    "path_switches"
                ],
                timeout_seconds=5.0,
            )

        logs = run_dir / "logs"
        sink_log = (logs / "workload_sink.log").open("w", encoding="utf-8")
        workload_log = (logs / "workload.log").open("w", encoding="utf-8")
        streams.extend([sink_log, workload_log])
        sink_process = target.popen(
            process_command(
                "src.experiments.workloads.udp_sink",
                "--bind-ip", str(scenario["target_ip"]),
                "--port", str(workload_config["target_port"]),
            ),
            cwd=str(repo_root), stdout=sink_log, stderr=subprocess.STDOUT,
        )
        processes.append(sink_process)
        time.sleep(0.25)
        ensure_process_running(sink_process, "workload UDP sink")

        qos_echo_process = None
        qos_echo_log = None
        qos_log = None
        if qos_config.get("enabled", False):
            qos_echo_log = (logs / "qos_echo.log").open("w", encoding="utf-8")
            qos_log = (logs / "qos.log").open("w", encoding="utf-8")
            streams.extend([qos_echo_log, qos_log])
            qos_echo_process = target.popen(
                process_command(
                    "src.experiments.workloads.qos_echo",
                    "--bind-ip", str(scenario["target_ip"]),
                    "--port", str(qos_config["target_port"]),
                ),
                cwd=str(repo_root), stdout=qos_echo_log, stderr=subprocess.STDOUT,
            )
            processes.append(qos_echo_process)
            time.sleep(0.25)
            ensure_process_running(qos_echo_process, "QoS echo server")

        collector = RunCollector(
            controller_urls=runtime["controller_urls"],
            orchestrator_url=str(runtime["orchestrator_url"]),
            run_dir=run_dir,
            poll_interval_seconds=float(telemetry_config["interval_seconds"]),
        )
        collector.start()
        workload_process = source.popen(
            process_command(
                "src.experiments.workloads.udp_new_flow",
                "--run-id", run_id,
                "--source-host", str(scenario["source_host"]),
                "--source-ip", str(scenario["source_ip"]),
                "--target-host", str(scenario["target_host"]),
                "--target-ip", str(scenario["target_ip"]),
                "--target-port", str(workload_config["target_port"]),
                "--pattern", str(workload_config["pattern"]),
                "--rate", str(rate),
                "--duration", str(workload_duration),
                "--sample-interval", str(workload_config["sample_interval_seconds"]),
                "--source-port-start", str(workload_config["source_port_start"]),
                "--source-port-end", str(workload_config["source_port_end"]),
                "--flow-idle-timeout", str(workload_config["flow_idle_timeout_seconds"]),
                "--port-reuse-safety-factor", str(workload_config["port_reuse_safety_factor"]),
                "--output", str(run_dir / "workload.jsonl"),
            ),
            cwd=str(repo_root), stdout=workload_log, stderr=subprocess.STDOUT,
        )
        processes.append(workload_process)

        qos_process = None
        if qos_config.get("enabled", False):
            qos_process = source.popen(
                process_command(
                    "src.experiments.workloads.qos_probe",
                    "--run-id", run_id,
                    "--source-host", str(scenario["source_host"]),
                    "--source-ip", str(scenario["source_ip"]),
                    "--target-host", str(scenario["target_host"]),
                    "--target-ip", str(scenario["target_ip"]),
                    "--target-port", str(qos_config["target_port"]),
                    "--rate", str(qos_config["rate_per_second"]),
                    "--duration", str(workload_duration),
                    "--timeout", str(qos_config["response_timeout_seconds"]),
                    "--source-port-start", str(qos_config["source_port_start"]),
                    "--source-port-end", str(qos_config["source_port_end"]),
                    "--flow-idle-timeout", str(qos_config["flow_idle_timeout_seconds"]),
                    "--output", str(run_dir / "qos.jsonl"),
                ),
                cwd=str(repo_root), stdout=qos_log, stderr=subprocess.STDOUT,
            )
            processes.append(qos_process)

        time.sleep(warmup)
        ensure_process_running(workload_process, "workload generator")
        if qos_process is not None:
            ensure_process_running(qos_process, "QoS probe")
        metadata.measurement_started_at = utc_now()
        write_json(metadata_path, metadata.to_dict())
        time.sleep(measurement)
        metadata.measurement_ended_at = utc_now()
        write_json(metadata_path, metadata.to_dict())

        workload_process.wait(timeout=10.0)

        if workload_process.returncode != 0:
            raise RuntimeError(
                "workload generator failed with "
                f"return code {workload_process.returncode}"
            )

        if qos_process is not None:
            qos_wait_timeout = max(
                10.0,
                float(
                    qos_config[
                        "response_timeout_seconds"
                    ]
                )
                + 5.0,
            )

            qos_process.wait(
                timeout=qos_wait_timeout
            )

            if qos_process.returncode != 0:
                raise RuntimeError(
                    "QoS probe failed with "
                    f"return code {qos_process.returncode}"
                )

        time.sleep(cooldown)

        residual_flows = (
            experiment_flow_counts(
                scenario[
                    "path_switches"
                ]
            )
        )

        write_json(
            run_dir
            / "post_run_flows.json",
            {
                "run_id":
                    run_id,

                "observed_at":
                    utc_now().isoformat(),

                "flows":
                    residual_flows,
            },
        )

        metadata.ended_at = utc_now()
        write_json(
            metadata_path,
            metadata.to_dict(),
        )

        # Stop polling before aggregation so the JSONL streams are final and
        # no collector thread survives into the next run of the matrix.
        collector.stop()
        collector = None

        post_state = validate_runtime_state(config, controller_id)
        if any(
            post_state["ownership"].get(switch) != controller_id
            for switch in scenario["path_switches"]
        ):
            raise RuntimeError("ownership changed during capacity experiment")

        summary = aggregate_run(run_dir)
        write_json(run_dir / "summary.json", summary)
        validation = validate_summary(summary, config)
        write_json(run_dir / "validation.json", validation)
        if not validation["valid"]:
            raise RuntimeError("capacity run validation failed: " + ", ".join(validation["errors"]))
        return run_dir
    except Exception as exc:
        write_json(run_dir / "failure.json", {
            "run_id": run_id,
            "observed_at": utc_now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    finally:
        if collector is not None:
            collector.stop()

        for process in processes:
            terminate_process(
                process
            )

        for stream in streams:
            stream.close()

        # Cleanup is best-effort only.
        # Measurement validity must not depend
        # on post-run flow expiry because an
        # overloaded controller may still be
        # draining queued Packet-In events.
        try:
            clear_experiment_flows(
                scenario["path_switches"]
            )
        except Exception:
            pass

        restore_sudo_owner(
            run_dir
        )
