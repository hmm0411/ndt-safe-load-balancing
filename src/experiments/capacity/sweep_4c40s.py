"""4C40S sweeps of Week 15 (Day 3 probe sweep, Day 4 capacity-v5): the pure part.

The Mininet driver (src/experiments/topologies/capacity_4c40s.py) and the VM script
(scripts/experiments/run_week15_vm_4c40s.sh) use these helpers; nothing here needs
Mininet, sudo or a controller, so it is unit tested.

Two sections are added to the capacity config of capacity-v4:

    background:   {rate_fps: 30}   every non-target controller carries this new-flow rate
                                   on its own 10-switch path (null or absent: idle, as v4)
    setup_probe:  {enabled, rate_per_second, timeout_seconds, port, switches}
                                   the Day 2 flow-setup probe p<i> -> h<i> on every switch

A run is never retried automatically: a failed or incomplete run directory stays as
evidence and is reported; to run it again, move it to data/experiment_runs_invalid/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiments.common.config import load_yaml
from src.experiments.common.run_io import make_run_id
from src.experiments.workloads.setup_probe import OwnershipTimeline, summarize_window
from src.experiments.workloads.setup_probe_io import read_samples

STATUSES = ("missing", "valid", "failed", "incomplete")


@dataclass(frozen=True)
class PlannedRun:
    controller: str
    rate: float
    repeat: int
    run_id: str


def _switch_index(name: str) -> int:
    if not name.startswith("s") or not name[1:].isdigit():
        raise ValueError(f"bad switch name: {name}")
    return int(name[1:])


def background_rate(config: Mapping[str, Any]) -> float | None:
    section = config.get("background") or {}
    value = section.get("rate_fps")
    if value is None:
        return None
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("background.rate_fps must be > 0 or null")
    return rate


def setup_probe_spec(config: Mapping[str, Any]) -> dict[str, Any] | None:
    section = config.get("setup_probe") or {}
    if not section.get("enabled", False):
        return None
    rate = float(section["rate_per_second"])
    timeout = float(section["timeout_seconds"])
    if rate <= 0 or timeout <= 0:
        raise ValueError("setup_probe rate_per_second and timeout_seconds must be > 0")
    return {
        "rate_per_second": rate,
        "timeout_seconds": timeout,
        "port": int(section["port"]),
        "switches": str(section.get("switches", "all")),
    }


def validate_sweep_config(
    config: Mapping[str, Any], controllers_config: Mapping[str, Any]
) -> None:
    """Checks the 2C validator does not make: all four controllers, ownership, ports."""
    controllers = sorted(controllers_config["controllers"])
    ownership = {str(k): str(v) for k, v in controllers_config["initial_ownership"].items()}
    urls = config["runtime"]["controller_urls"]
    scenarios = config["scenarios"]
    for cid in controllers:
        if cid not in urls or cid not in scenarios:
            raise ValueError(f"{cid}: missing controller URL or scenario")
        expected_url = str(controllers_config["controllers"][cid]["rest_url"])
        if str(urls[cid]).rstrip("/") != expected_url.rstrip("/"):
            raise ValueError(f"{cid}: controller URL differs from the controllers config")
        scenario = scenarios[cid]
        path = [str(s) for s in scenario["path_switches"]]
        wrong = [s for s in path if ownership.get(s) != cid]
        if wrong:
            raise ValueError(f"{cid}: path switches not owned by {cid}: {wrong}")
        first, last = _switch_index(path[0]), _switch_index(path[-1])
        if (scenario["source_host"], scenario["target_host"]) != (f"h{first}", f"h{last}"):
            raise ValueError(f"{cid}: hosts must be h{first} -> h{last}")
        if (scenario["source_ip"], scenario["target_ip"]) != (f"10.0.0.{first}",
                                                              f"10.0.0.{last}"):
            raise ValueError(f"{cid}: IPs must be 10.0.0.{first} -> 10.0.0.{last}")
    rates = [float(r) for r in config["workload"]["rates_fps"]]
    if not rates or any(r <= 0 for r in rates) or rates != sorted(set(rates)):
        raise ValueError("workload.rates_fps must be positive, unique and increasing")
    if int(config["repeats"]) < 1:
        raise ValueError("repeats must be >= 1")
    background_rate(config)
    sweep_controllers(config, controllers_config)
    probe = setup_probe_spec(config)
    if probe is not None:
        used = {int(config["workload"]["target_port"])}
        if config["qos_probe"].get("enabled", False):
            used.add(int(config["qos_probe"]["target_port"]))
        if probe["port"] in used:
            raise ValueError("setup_probe.port collides with the workload or QoS port")
    if config["migration"].get("enabled", True):
        raise ValueError("migration must be disabled in a sweep")


def sweep_controllers(
    config: Mapping[str, Any], controllers_config: Mapping[str, Any]
) -> list[str]:
    """Target controllers of the sweep: sweep_controllers if set, else all of them."""
    every = sorted(str(c) for c in controllers_config["controllers"])
    chosen = [str(c) for c in config.get("sweep_controllers") or every]
    unknown = sorted(set(chosen) - set(every))
    if unknown or len(set(chosen)) != len(chosen):
        raise ValueError(f"sweep_controllers must be distinct known controllers: {chosen}")
    return chosen


def plan_runs(
    config: Mapping[str, Any], controllers: Iterable[str], rates: Iterable[float] | None = None
) -> list[PlannedRun]:
    kind = str(config["experiment"]["type"])
    chosen = [float(r) for r in (rates if rates is not None else config["workload"]["rates_fps"])]
    return [
        PlannedRun(cid, rate, repeat, make_run_id(kind, cid, rate, repeat))
        for cid in controllers
        for rate in chosen
        for repeat in range(1, int(config["repeats"]) + 1)
    ]


def run_status(run_dir: Path) -> str:
    if not run_dir.exists():
        return "missing"
    validation = run_dir / "validation.json"
    if validation.exists():
        valid = json.loads(validation.read_text(encoding="utf-8")).get("valid") is True
        return "valid" if valid else "failed"
    if (run_dir / "failure.json").exists():
        return "failed"
    return "incomplete"


def probe_summary(
    probe_dir: Path,
    ownership: Mapping[str, str],
    start: datetime,
    end: datetime,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Per-controller P95 over [start, end); ownership is static (no migration in a sweep)."""
    timeline = OwnershipTimeline()
    for switch, owner in sorted(ownership.items()):
        timeline.record(0.0, switch, owner)
    samples = []
    for path in sorted(probe_dir.glob("qos_s*.jsonl")):
        samples.extend(read_samples(path, timeline, path.stem.removeprefix("qos_")))
    out: dict[str, Any] = {}
    for cid in sorted(set(ownership.values())):
        stats = summarize_window(
            samples, cid, start.timestamp(), end.timestamp(), timeout_seconds * 1000.0
        )
        out[cid] = None if stats is None else {
            "setup_p95_ms": stats.p95_ms, "samples": stats.samples, "timeouts": stats.timeouts,
        }
    return out


def status_table(
    config: Mapping[str, Any], controllers: Iterable[str], repo_root: Path
) -> dict[str, list[str]]:
    root = repo_root / str(config["experiment"]["output_root"])
    table: dict[str, list[str]] = {s: [] for s in STATUSES}
    for run in plan_runs(config, controllers):
        table[run_status(root / run.run_id)].append(run.run_id)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Status of a 4C40S sweep config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--controllers-config", type=Path,
                        default=Path("configs/controllers_4c40s.yaml"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--list", action="store_true", help="print the run ids per status")
    args = parser.parse_args(argv)
    config = load_yaml(args.repo_root / args.config)
    controllers_config = load_yaml(args.repo_root / args.controllers_config)
    validate_sweep_config(config, controllers_config)
    table = status_table(config, sweep_controllers(config, controllers_config), args.repo_root)
    total = sum(len(v) for v in table.values())
    print(f"{config['experiment']['type']}: " + ", ".join(
        f"{status} {len(table[status])}" for status in STATUSES) + f" / {total}")
    if args.list:
        for status in ("failed", "incomplete"):
            for run_id in table[status]:
                print(f"  {status.upper()}: {run_id}")
    return 0 if not table["failed"] and not table["incomplete"] else 1


if __name__ == "__main__":
    sys.exit(main())
