from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


INPUT_PATH = Path(
    "data/benchmarks/capacity_runs.csv"
)

OUTPUT_PATH = Path(
    "data/benchmarks/controller_capacity.json"
)

FINE_PREFIX = "capacity-2c20s-fine-"

EXPECTED_REPEATS = 5

# Constants of the capacity-v4 artifact (data/benchmarks/controller_capacity.json,
# "method" block): safe reference 90 fps, setup knee 140 fps, throughput knee
# 160 fps. The earlier values (140, 200, 240) did not match the artifact: with
# the committed capacity_runs.csv the script stopped with "missing required
# fine group". tests/unit/test_derive_controller_capacity.py regenerates the
# artifact from the CSV and compares it. The earlier comment block, which
# described the 140/200/240 sweep, is in git history.
SAFE_REFERENCE_RATE = 90.0
SETUP_KNEE_RATE = {
    "c1": 140.0,
    "c2": 140.0,
}
THROUGHPUT_KNEE_RATE = 160.0

# One-sided 95% Student-t critical value,
# df = n - 1 = 4.
T_95_ONE_SIDED_DF4 = 2.132


def load_rows() -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"missing input: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    fine = [
        row
        for row in rows
        if row.get(
            "run_id",
            "",
        ).startswith(FINE_PREFIX)
        and row.get(
            "valid",
            ""
        ).lower() == "true"
    ]

    if not fine:
        raise SystemExit(
            "no valid fine runs found"
        )

    return fine


def processed_values(
    rows: list[dict[str, str]],
) -> list[float]:
    return [
        float(
            row[
                "processed_packet_in_rate_mean"
            ]
        )
        for row in rows
    ]


def one_sided_lcb_95(
    data: list[float],
) -> float:
    if len(data) != EXPECTED_REPEATS:
        raise ValueError(
            "expected exactly "
            f"{EXPECTED_REPEATS} repeats"
        )

    mean_value = statistics.mean(data)
    std_value = statistics.stdev(data)

    return (
        mean_value
        - T_95_ONE_SIDED_DF4
        * std_value
        / math.sqrt(len(data))
    )


def main() -> None:
    rows = load_rows()

    groups: dict[
        tuple[str, float],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        key = (
            row["controller"],
            float(
                row[
                    "target_new_flow_rate"
                ]
            ),
        )

        groups[key].append(row)

    controllers = sorted(
        {
            row["controller"]
            for row in rows
        }
    )

    result: dict[str, Any] = {
        "schema_version": "1.0",
        "method": {
            "experiment":
                "capacity-2c20s-fine",

            "repeats_per_rate":
                EXPECTED_REPEATS,

            "critical_definition":
                (
                    "minimum observed limiting "
                    "capacity among setup, "
                    "throughput, and response"
                ),

            "setup_knee_rate_fps":
                SETUP_KNEE_RATE,

            "throughput_knee_rate_fps":
                THROUGHPUT_KNEE_RATE,

            "safe_reference_rate_fps":
                SAFE_REFERENCE_RATE,

            "safe_definition":
                (
                    "one-sided 95% lower "
                    "confidence bound at the "
                    "highest healthy pre-knee "
                    "operating point"
                ),

            "student_t":
                T_95_ONE_SIDED_DF4,
        },

        "controllers": {},
    }

    for controller in controllers:
        safe_group = groups.get(
            (
                controller,
                SAFE_REFERENCE_RATE,
            )
        )

        if controller not in SETUP_KNEE_RATE:
            raise RuntimeError(
                f"no setup knee configured for {controller}"
            )

        setup_group = groups.get(
            (
                controller,
                SETUP_KNEE_RATE[controller],
            )
        )

        throughput_group = groups.get(
            (
                controller,
                THROUGHPUT_KNEE_RATE,
            )
        )

        if (
            safe_group is None
            or setup_group is None
            or throughput_group is None
        ):
            raise RuntimeError(
                f"missing required fine group "
                f"for {controller}"
            )

        for group in (
            safe_group,
            setup_group,
            throughput_group,
        ):
            if len(group) != EXPECTED_REPEATS:
                raise RuntimeError(
                    f"{controller}: expected "
                    f"{EXPECTED_REPEATS} repeats"
                )

        safe_values = processed_values(
            safe_group
        )

        setup_values = processed_values(
            setup_group
        )

        throughput_values = processed_values(
            throughput_group
        )

        safe_mean = statistics.mean(
            safe_values
        )

        safe_std = statistics.stdev(
            safe_values
        )

        c_safe = one_sided_lcb_95(
            safe_values
        )

        c_setup = statistics.mean(
            setup_values
        )

        c_throughput = statistics.mean(
            throughput_values
        )

        # No response-time knee was observed
        # before setup/throughput degradation.
        c_response = None

        observed_limits = [
            c_setup,
            c_throughput,
        ]

        if c_response is not None:
            observed_limits.append(
                c_response
            )

        c_critical = min(
            observed_limits
        )

        result["controllers"][
            controller
        ] = {
            "c_throughput_pps":
                c_throughput,

            "c_setup_pps":
                c_setup,

            "c_response_pps":
                c_response,

            "c_critical_pps":
                c_critical,

            "c_safe_pps":
                c_safe,

            "setup_knee_rate_fps":
                SETUP_KNEE_RATE[controller],

            "safe_reference": {
                "target_rate_fps":
                    SAFE_REFERENCE_RATE,

                "processed_mean_pps":
                    safe_mean,

                "processed_std_pps":
                    safe_std,

                "sample_count":
                    len(safe_values),
            },
        }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print(
        f"wrote capacity model to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
