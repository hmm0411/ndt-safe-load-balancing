"""Start the flow-setup probe on chosen switches.

p<i> sends to h<i>, both on s<i>, so each probe flow crosses exactly one switch.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def parse_switches(text: str, total: int) -> list[int]:
    if text == "all":
        return list(range(1, total + 1))
    indices = sorted({int(part) for part in text.split(",")})
    if not indices or indices[0] < 1 or indices[-1] > total:
        raise ValueError(f"switch indices must be within 1..{total}")
    return indices


def echo_command(bind_ip: str, port: int) -> list[str]:
    return [
        "python3", "-m", "src.experiments.workloads.qos_echo",
        "--bind-ip", bind_ip, "--port", str(port),
    ]


def probe_command(
    index: int, source_ip: str, target_ip: str, port: int, rate: float,
    duration: float, timeout: float, run_id: str, output: Path,
) -> list[str]:
    return [
        "python3", "-m", "src.experiments.workloads.qos_probe",
        "--run-id", run_id,
        "--source-host", f"p{index}", "--source-ip", source_ip,
        "--target-host", f"h{index}", "--target-ip", target_ip,
        "--target-port", str(port), "--rate", str(rate),
        "--duration", str(duration), "--timeout", str(timeout),
        "--output", str(output),
    ]


@dataclass
class ProbeSet:
    echoes: list[Any] = field(default_factory=list)
    probes: list[Any] = field(default_factory=list)
    logs: list[Any] = field(default_factory=list)

    def stop(self) -> None:
        for process in self.probes + self.echoes:
            if process.poll() is None:
                process.terminate()
        for process in self.probes + self.echoes:
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        for handle in self.logs:
            handle.close()


def launch_setup_probes(
    net: Any, indices: list[int], out_dir: Path, run_id: str, port: int,
    rate: float, duration: float, timeout: float, repo_root: Path,
) -> ProbeSet:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = ProbeSet()
    try:
        for index in indices:
            target = net.get(f"h{index}")
            log = (out_dir / f"echo_s{index}.log").open("w", encoding="utf-8")
            started.logs.append(log)
            started.echoes.append(target.popen(
                echo_command(target.IP(), port),
                cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT,
            ))
        time.sleep(0.25)
        for index, process in zip(indices, started.echoes):
            if process.poll() is not None:
                raise RuntimeError(f"qos_echo on h{index} exited early")
        for index in indices:
            source, target = net.get(f"p{index}"), net.get(f"h{index}")
            log = (out_dir / f"probe_s{index}.log").open("w", encoding="utf-8")
            started.logs.append(log)
            started.probes.append(source.popen(
                probe_command(
                    index, source.IP(), target.IP(), port, rate, duration,
                    timeout, run_id, out_dir / f"qos_s{index}.jsonl",
                ),
                cwd=str(repo_root), stdout=log, stderr=subprocess.STDOUT,
            ))
    except Exception:
        started.stop()
        raise
    return started
