#!/usr/bin/env python3
"""Push per-controller flow-setup P95 from the probe files to the orchestrator.

The window, period, timeout and duration have no default: pass them.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from src.experiments.workloads.setup_probe_loop import PushLoop

ORCH = "http://127.0.0.1:9000"


def call(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        url,
        data=data,
        method="GET" if body is None else "POST",
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10.0) as response:
        result: dict[str, Any] = json.load(response)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--orchestrator-url", default=ORCH)
    parser.add_argument("--window-seconds", type=float, required=True)
    parser.add_argument("--period-seconds", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = args.orchestrator_url.rstrip("/")
    ownership = call(f"{base}/api/v1/state")["ownership"]
    if not isinstance(ownership, dict):
        raise SystemExit("/api/v1/state has no ownership mapping")
    since = time.time()
    loop = PushLoop(
        args.probe_dir, ownership, since, args.window_seconds, args.timeout_seconds
    )
    deadline = since + args.duration_seconds
    while time.time() < deadline:
        began = time.time()
        transactions = call(f"{base}/api/v1/state")["transactions"]
        summary = loop.cycle(
            transactions,
            lambda body: call(f"{base}/api/v1/twin/setup-probe", body),
            began,
        )
        summary["at"] = datetime.fromtimestamp(began, timezone.utc).isoformat()
        print(json.dumps(summary), flush=True)
        time.sleep(max(0.0, args.period_seconds - (time.time() - began)))


if __name__ == "__main__":
    main()
