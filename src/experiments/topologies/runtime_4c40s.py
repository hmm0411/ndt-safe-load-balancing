#!/usr/bin/env python3
"""4C40S runtime: four Ryu controllers, 40 OVS switches, and on every
switch one workload host (h<i>) and one probe host (p<i>)."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo

from src.experiments.common.config import load_yaml
from src.experiments.topologies.capacity_smoke_2c20s import (
    post_json,
    wait_for_controller_connections,
)
from src.experiments.workloads import forecast_4c
from src.experiments.workloads.profile_spec import parse_profile_specs
from src.experiments.topologies.setup_probe_launcher import (
    launch_setup_probes,
    parse_switches,
)

SWITCH_COUNT = 40
DEFAULT_CONFIG = Path("configs/controllers_4c40s.yaml")
REPO_ROOT = Path(__file__).resolve().parents[3]


class Runtime4C40STopo(Topo):
    """Chain s1--s2--...--s40; h<i> and p<i> are attached to s<i>."""

    def build(self) -> None:
        switches = []
        for index in range(1, SWITCH_COUNT + 1):
            switch = self.addSwitch(
                f"s{index}",
                dpid=f"{index:016x}",
                protocols="OpenFlow13",
            )
            host = self.addHost(
                f"h{index}",
                ip=f"10.0.0.{index}/24",
                mac=f"00:00:00:00:00:{index:02x}",
            )
            probe = self.addHost(
                f"p{index}",
                ip=f"10.0.0.{100 + index}/24",
                mac=f"00:00:00:00:01:{index:02x}",
            )
            self.addLink(host, switch)
            self.addLink(probe, switch)
            switches.append(switch)
        for left, right in zip(switches, switches[1:]):
            self.addLink(left, right)


def controller_specs(config: dict[str, Any]) -> dict[str, tuple[int, str]]:
    """Map controller id -> (OpenFlow port, REST url) from the config."""
    return {
        str(cid): (int(spec["openflow_port"]), str(spec["rest_url"]))
        for cid, spec in config["controllers"].items()
    }


def expected_ownership(config: dict[str, Any]) -> dict[str, str]:
    return {
        str(switch): str(owner)
        for switch, owner in config["initial_ownership"].items()
    }


def validate_ownership(ownership: object, expected: dict[str, str]) -> None:
    if not isinstance(ownership, dict):
        raise RuntimeError("init-roles response missing ownership")
    differing = sorted(
        key
        for key in set(ownership) | set(expected)
        if ownership.get(key) != expected.get(key)
    )
    if differing:
        raise RuntimeError(f"ownership differs from config at: {differing}")


def initialize_runtime(config: dict[str, Any], orchestrator_url: str) -> None:
    urls = {cid: rest for cid, (_, rest) in controller_specs(config).items()}
    expected = expected_ownership(config)
    print(f"*** Waiting for all {len(expected)} switch-controller connections")
    wait_for_controller_connections(
        controller_urls=urls,
        expected_switch_count=len(expected),
        timeout_seconds=120.0,
    )
    print("*** Initializing OpenFlow roles")
    roles = post_json(f"{orchestrator_url.rstrip('/')}/api/v1/init-roles")
    validate_ownership(roles.get("ownership"), expected)
    print("*** INIT_ROLES:", roles.get("ownership"), flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4C40S runtime topology")
    parser.add_argument("--controllers-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--orchestrator-url", default="http://127.0.0.1:9000")
    parser.add_argument(
        "--init-roles",
        action="store_true",
        help="wait for every connection, then POST /api/v1/init-roles",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="keep the network up this long instead of opening the CLI",
    )
    parser.add_argument("--setup-probe-dir", type=Path, default=None)
    parser.add_argument("--setup-probe-run-id", default=None)
    parser.add_argument("--setup-probe-rate", type=float, default=None)
    parser.add_argument("--setup-probe-timeout", type=float, default=None)
    parser.add_argument("--setup-probe-switches", default="all")
    parser.add_argument("--setup-probe-port", type=int, default=9001)
    # Week 15 Day 5: run one forecast workload (the 2C20S runners, host tables from the config)
    parser.add_argument("--forecast", choices=sorted(forecast_4c.RUNNERS), default=None)
    parser.add_argument("--capacity-runs", type=Path,
                        default=Path("data/benchmarks/capacity_runs_4c40s.csv"))
    parser.add_argument("--capacity-experiment", default=forecast_4c.CAPACITY_EXPERIMENT)
    parser.add_argument("--controller-capacity", type=Path,
                        default=Path("data/benchmarks/controller_capacity_v5.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/experiment_runs/transfer_4c"))
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment-type", default="transfer-4c-forecast")
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--target-utilization", type=float, default=0.60)
    parser.add_argument("--profile", action="append", default=None,
                        metavar="CONTROLLER=PROFILE")
    return parser


def setup_probe_args_error(args: argparse.Namespace) -> str | None:
    """With --setup-probe-dir the rate, timeout and run id have no default;
    the probe runs for --hold-seconds."""
    if args.setup_probe_dir is None:
        return None
    needed = ("setup_probe_run_id", "setup_probe_rate", "setup_probe_timeout")
    missing = [n.replace("_", "-") for n in needed if getattr(args, n) is None]
    if missing:
        return "--setup-probe-dir also needs: --" + ", --".join(missing)
    if args.hold_seconds <= 0:
        return "--setup-probe-dir needs --hold-seconds > 0 (the probe duration)"
    return None


def main() -> None:
    args = build_arg_parser().parse_args()
    probe_error = setup_probe_args_error(args)
    if probe_error:
        raise SystemExit(probe_error)
    if args.forecast and args.setup_probe_dir is not None:
        raise SystemExit("--forecast runs without the setup probe (the 2C models never saw it)")
    runner = None
    if args.forecast:
        # the runners build their host tables at import time: environment first
        os.environ.update(forecast_4c.runner_env(args.controllers_config))
        runner = forecast_4c.load_runner(args.forecast)
        used = args.output_dir / "capacity_runs_used.csv"
        n = forecast_4c.filter_capacity_rows(args.capacity_runs, used, args.capacity_experiment)
        print(f"*** capacity rows: {n} valid {args.capacity_experiment} rows -> {used}")
        args.capacity_runs = used
    config = load_yaml(args.controllers_config)
    specs = controller_specs(config)
    setLogLevel("info")
    net = Mininet(
        topo=Runtime4C40STopo(),
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        build=False,
    )
    for cid, (port, _) in specs.items():
        net.addController(cid, controller=RemoteController, ip="127.0.0.1", port=port)
    probes = None
    try:
        print("*** Building 4C40S topology")
        net.build()
        print("*** Starting network")
        net.start()
        print(f"*** 4C40S runtime topology READY: {len(specs)} controllers")
        if args.init_roles or runner is not None:
            initialize_runtime(config, args.orchestrator_url)
        if runner is not None:
            print(f"*** Starting {args.forecast} forecast workload on 4C40S", flush=True)
            run_dir = runner(net, **forecast_4c.runner_kwargs(
                args.forecast, capacity_runs=args.capacity_runs,
                controller_capacity=args.controller_capacity, output_dir=args.output_dir,
                duration=args.duration, sample_interval=args.sample_interval,
                orchestrator_url=args.orchestrator_url, run_id=args.run_id, seed=args.seed,
                repeat_index=args.repeat_index, experiment_type=args.experiment_type,
                git_commit=args.git_commit, profiles=parse_profile_specs(args.profile),
                target_utilization=args.target_utilization))
            print(f"*** FORECAST RUN DONE: {run_dir}", flush=True)
            return
        if args.setup_probe_dir is not None:
            probes = launch_setup_probes(
                net,
                parse_switches(args.setup_probe_switches, SWITCH_COUNT),
                args.setup_probe_dir,
                args.setup_probe_run_id,
                args.setup_probe_port,
                args.setup_probe_rate,
                args.hold_seconds,
                args.setup_probe_timeout,
                REPO_ROOT,
            )
            print(f"*** SETUP PROBE started: {len(probes.probes)} probes", flush=True)
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        else:
            CLI(net)
    finally:
        if probes is not None:
            probes.stop()
        print("*** Stopping network")
        net.stop()


if __name__ == "__main__":
    main()
