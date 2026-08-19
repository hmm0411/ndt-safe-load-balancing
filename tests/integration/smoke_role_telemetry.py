#!/usr/bin/env python3
"""Headless 2C-4S smoke test for a pre-provisioned self-hosted runner.

C1, C2 and the orchestrator must already be running. This script owns Mininet.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

from src.experiments.topologies.smoke_2c4s import Smoke2C4STopo

ORCH = "http://127.0.0.1:9000"


def request_json(method, url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode())


def wait_until(predicate, timeout=12, interval=0.25):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"condition timed out; last={last}")


def force_packet_in(net):
    s1 = net.get("s1")
    h1 = net.get("h1")
    h4 = net.get("h4")
    s1.cmd("ovs-ofctl -O OpenFlow13 del-flows s1 'priority=10'")
    h1.cmd(f"ping -c 1 -W 1 {h4.IP()}")


def migrate_with_traffic(net, target, failure="none"):
    result = {}
    error = {}

    def invoke():
        try:
            result.update(
                request_json(
                    "POST",
                    f"{ORCH}/api/v1/migrations",
                    {
                        "switch_id": "s1",
                        "target_controller": target,
                        "simulate_failure": failure,
                    },
                    timeout=20,
                )
            )
        except Exception as exc:  # surfaced below
            error["value"] = exc

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    for _ in range(8):
        time.sleep(0.35)
        force_packet_in(net)
        if not thread.is_alive():
            break
    thread.join(timeout=20)
    if thread.is_alive():
        raise AssertionError("migration request did not finish")
    if error:
        raise error["value"]
    return result


def main():
    setLogLevel("warning")
    net = Mininet(
        topo=Smoke2C4STopo(),
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        build=False,
    )
    net.addController("c1", controller=RemoteController, ip="127.0.0.1", port=6653)
    net.addController("c2", controller=RemoteController, ip="127.0.0.1", port=6654)
    net.build()
    net.start()

    try:
        time.sleep(2)
        init = request_json("POST", f"{ORCH}/api/v1/init-roles")
        assert init["status"] == "INITIALIZED", init

        force_packet_in(net)

        def valid_pre_snapshot():
            state = request_json("GET", f"{ORCH}/api/v1/state")
            snap = state.get("latest_snapshot")
            return state if snap and snap.get("quality", {}).get("valid") is True else None

        pre = wait_until(valid_pre_snapshot)
        pre_version = int(pre["ownership_version"])
        assert pre["ownership"]["s1"] == "c1", pre

        committed = migrate_with_traffic(net, "c2")
        assert committed["status"] == "COMMITTED", committed

        def post_snapshot():
            state = request_json("GET", f"{ORCH}/api/v1/state")
            snap = state.get("latest_snapshot")
            if (
                state["ownership"].get("s1") == "c2"
                and int(state["ownership_version"]) > pre_version
                and snap
                and int(snap["ownership_version"]) == int(state["ownership_version"])
                and snap.get("quality", {}).get("valid") is True
            ):
                return state
            return None

        post = wait_until(post_snapshot)
        post_version = int(post["ownership_version"])

        # Prove the reverse committed direction as well, returning s1 to c1.
        committed_back = migrate_with_traffic(net, "c1")
        assert committed_back["status"] == "COMMITTED", committed_back

        def back_snapshot():
            state = request_json("GET", f"{ORCH}/api/v1/state")
            snap = state.get("latest_snapshot")
            if (
                state["ownership"].get("s1") == "c1"
                and int(state["ownership_version"]) > post_version
                and snap
                and int(snap["ownership_version"]) == int(state["ownership_version"])
                and snap.get("quality", {}).get("valid") is True
            ):
                return state
            return None

        back = wait_until(back_snapshot)
        back_version = int(back["ownership_version"])

        # Now inject failure on c1 -> c2. Rollback must restore c1 exactly as
        # required by the telemetry milestone document.
        restored = migrate_with_traffic(net, "c2", failure="flow_mod")
        assert restored["status"] == "RESTORED", restored

        def rollback_snapshot():
            state = request_json("GET", f"{ORCH}/api/v1/state")
            snap = state.get("latest_snapshot")
            if (
                state["ownership"].get("s1") == "c1"
                and int(state["ownership_version"]) > back_version
                and snap
                and int(snap["ownership_version"]) == int(state["ownership_version"])
                and snap.get("quality", {}).get("valid") is True
            ):
                return state
            return None

        wait_until(rollback_snapshot)
        print("SMOKE_OK role+telemetry+snapshot+bidirectional-migration+rollback")
    finally:
        net.stop()


if __name__ == "__main__":
    main()
