"""Probe-ON settings of the 4C outcome set (Week 15, Day 6).

The probe runs on all 40 switches for the whole attempt and a push loop posts the
per-controller flow-setup P95 to the orchestrator, so GET /api/v1/twin/state (the
snapshots the runner saves) carries setup_p95_ms. The probe is the capacity-v5 ON
sweep's (configs/experiments/probe_sweep_4c40s.yaml, setup_probe:), so U means the
same in both sets. The push loop has no experiment value in the repo (day2_setup_probe.md
calls its numbers a path check): window = the 10 s of the probe design, period = the
1 s snapshot interval, timeout = the probe's. They are saved next to the attempt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

RATE_PER_SECOND = 1.0
TIMEOUT_S = 1.0
PORT = 9101
SWITCHES = "all"
WINDOW_S = 10.0
PERIOD_S = 1.0
HOLD_MARGIN_S = 30.0    # the probe outlives the workload; the workload runner stops it
PUSH_MARGIN_S = 120.0   # the push loop outlives the post window; the runner stops it


def probe_dir(workload_out: Path) -> Path:
    return workload_out / "setup_probe"


def workload_args(enabled: bool, workload_out: Path, run_id: str, duration: float) -> list[str]:
    """Extra runtime_4c40s.py arguments: empty when the probe is off."""
    if not enabled:
        return []
    return [
        "--setup-probe-dir", str(probe_dir(workload_out)),
        "--setup-probe-run-id", run_id,
        "--setup-probe-rate", str(RATE_PER_SECOND),
        "--setup-probe-timeout", str(TIMEOUT_S),
        "--setup-probe-port", str(PORT),
        "--setup-probe-switches", SWITCHES,
        "--hold-seconds", str(duration + HOLD_MARGIN_S),
    ]


def push_command(python: str, directory: Path, url: str, duration: float,
                 post_timeout_s: float) -> list[str]:
    return [
        python, "scripts/experiments/push_setup_probe.py",
        "--probe-dir", str(directory), "--orchestrator-url", url,
        "--window-seconds", str(WINDOW_S), "--period-seconds", str(PERIOD_S),
        "--timeout-seconds", str(TIMEOUT_S),
        "--duration-seconds", str(duration + post_timeout_s + PUSH_MARGIN_S),
    ]


def written_files(directory: Path) -> int:
    return len(list(directory.glob("qos_s*.jsonl")))


def parameters() -> dict[str, Any]:
    return {
        "rate_per_second": RATE_PER_SECOND, "timeout_s": TIMEOUT_S, "port": PORT,
        "switches": SWITCHES, "push_window_s": WINDOW_S, "push_period_s": PERIOD_S,
    }
