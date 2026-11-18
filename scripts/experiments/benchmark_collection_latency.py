#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ORCH = "http://127.0.0.1:9000"


def request_json(
    url: str,
    timeout: float = 5.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            value = json.loads(
                response.read().decode()
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"HTTP {exc.code} GET {url}: "
            f"{body}"
        ) from exc

    if not isinstance(value, dict):
        raise RuntimeError(
            "expected JSON object"
        )

    return value


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    if not values:
        raise ValueError(
            "cannot calculate percentile "
            "of empty values"
        )

    ordered = sorted(values)

    index = min(
        len(ordered) - 1,
        math.ceil(
            fraction * len(ordered)
        ) - 1,
    )

    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/benchmarks/runtime/"
            "collection_latency.jsonl"
        ),
    )

    parser.add_argument(
        "--controllers",
        default="c1,c2",
        help="comma-separated controller ids the twin state must contain",
    )

    args = parser.parse_args()

    expected_ids = {item for item in args.controllers.split(",") if item}

    if args.count <= 0:
        raise ValueError(
            "--count must be positive"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.unlink(
        missing_ok=True,
    )

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []

    started = time.monotonic()

    while len(rows) < args.count:
        if (
            time.monotonic() - started
            > args.timeout
        ):
            raise RuntimeError(
                "collection benchmark timed out: "
                f"collected={len(rows)}/"
                f"{args.count}"
            )

        state = request_json(
            f"{ORCH}/api/v1/twin/state"
        )

        snapshot_id = str(
            state.get(
                "snapshot_id",
                "",
            )
        )

        if not snapshot_id:
            raise RuntimeError(
                "TwinState missing snapshot_id"
            )

        if snapshot_id in seen:
            time.sleep(
                args.poll_interval
            )
            continue

        quality = state.get(
            "quality",
            {},
        )

        if not isinstance(
            quality,
            dict,
        ):
            raise RuntimeError(
                "TwinState quality is invalid"
            )

        if quality.get("valid") is not True:
            print(
                "SKIP INVALID:",
                snapshot_id,
                "age_ms=",
                quality.get("age_of_twin_ms"),
                "twinning_rate=",
                quality.get("twinning_rate"),
                "consistent=",
                quality.get("consistent"),
            )
            time.sleep(
                args.poll_interval
            )
            continue

        controllers = state.get(
            "controllers",
            [],
        )

        if not isinstance(
            controllers,
            list,
        ):
            raise RuntimeError(
                "TwinState controllers "
                "must be a list"
            )

        controller_ids = {
            str(
                item.get(
                    "controller_id",
                    "",
                )
            )
            for item in controllers
            if isinstance(item, dict)
        }

        if controller_ids != expected_ids:
            raise RuntimeError(
                "expected controllers "
                f"{sorted(expected_ids)}, got "
                f"{sorted(controller_ids)}"
            )

        for controller in controllers:
            latency = controller.get(
                "collection_latency_ms"
            )

            if latency is None:
                raise RuntimeError(
                    "controller missing "
                    "collection_latency_ms"
                )

            if float(latency) < 0:
                raise RuntimeError(
                    "negative collection latency"
                )

        seen.add(snapshot_id)
        rows.append(state)

        with args.output.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(state)
                + "\n"
            )

        print(
            f"SNAPSHOT "
            f"{len(rows):02d}/"
            f"{args.count}: "
            f"{snapshot_id}"
        )

        time.sleep(
            args.poll_interval
        )

    elapsed = (
        time.monotonic()
        - started
    )

    values: list[float] = []

    for row in rows:
        for controller in row[
            "controllers"
        ]:
            values.append(
                float(
                    controller[
                        "collection_latency_ms"
                    ]
                )
            )

    print()
    print(
        "COLLECTION_BENCHMARK_PASS:"
    )
    print(
        "snapshots:",
        len(rows),
    )
    print(
        "unique_snapshot_ids:",
        len(seen),
    )
    print(
        "controller_observations:",
        len(values),
    )
    print(
        "elapsed_seconds:",
        round(elapsed, 3),
    )
    print(
        "mean_ms:",
        round(
            sum(values)
            / len(values),
            3,
        ),
    )
    print(
        "p50_ms:",
        round(
            percentile(
                values,
                0.50,
            ),
            3,
        ),
    )
    print(
        "p95_ms:",
        round(
            percentile(
                values,
                0.95,
            ),
            3,
        ),
    )
    print(
        "max_ms:",
        round(
            max(values),
            3,
        ),
    )
    print(
        "OUTPUT:",
        args.output,
    )


if __name__ == "__main__":
    main()
