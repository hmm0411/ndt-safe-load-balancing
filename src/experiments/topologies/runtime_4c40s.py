#!/usr/bin/env python3
"""4C40S runtime: four Ryu controllers, 40 OVS switches, and on every
switch one workload host (h<i>) and one probe host (p<i>)."""
from __future__ import annotations

import argparse
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

SWITCH_COUNT = 40
DEFAULT_CONFIG = Path("configs/controllers_4c40s.yaml")


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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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
    try:
        print("*** Building 4C40S topology")
        net.build()
        print("*** Starting network")
        net.start()
        print(f"*** 4C40S runtime topology READY: {len(specs)} controllers")
        if args.init_roles:
            initialize_runtime(config, args.orchestrator_url)
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        else:
            CLI(net)
    finally:
        print("*** Stopping network")
        net.stop()


if __name__ == "__main__":
    main()
