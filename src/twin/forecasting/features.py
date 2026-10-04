from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Sequence


CURRENT_FEATURES = (
    "utilization",
    "processed_packet_in_rate",
    "flow_mod_rate",
    "process_cpu_percent",
    "process_memory_rss_mb",
    "response_p95_ms",
    "managed_switch_count",
    "max_switch_control_load_share",
)

LAG_STEPS = tuple(range(1, 11))

ROLLING_WINDOWS = (3, 5, 10)


def _numeric(
    row: dict[str, Any],
    field: str,
) -> float:
    if field not in row:
        raise ValueError(
            f"missing required numeric field: {field}"
        )

    value = row[field]

    if value is None:
        raise ValueError(
            f"{field} must not be null"
        )

    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be numeric, got bool"
        )

    if not isinstance(value, (int, float, str)):
        raise ValueError(
            f"{field} must be numeric"
        )

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{field} must be numeric"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"{field} must be finite"
        )

    return number


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _population_std(
    values: Sequence[float],
) -> float:
    mean = _mean(values)

    return math.sqrt(
        sum(
            (value - mean) ** 2
            for value in values
        )
        / len(values)
    )


def _slope(values: Sequence[float]) -> float:
    """
    Ordinary least-squares slope for equally spaced observations.

    x = 0, 1, ..., n-1
    """

    n = len(values)

    if n < 2:
        raise ValueError(
            "at least two values required for slope"
        )

    x_mean = (n - 1) / 2.0
    y_mean = _mean(values)

    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    )

    denominator = sum(
        (index - x_mean) ** 2
        for index in range(n)
    )

    return numerator / denominator


def build_features(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach leakage-safe historical features without dropping rows.

    Rows without enough history are retained with unavailable
    historical features set to None. Final filtering happens only
    after both features and future labels have been attached.
    """
    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        sequence_id = str(
            record.get("sequence_id", "")
        )

        if not sequence_id:
            raise ValueError(
                "sequence_id is required before feature engineering"
            )

        grouped[sequence_id].append(
            dict(record)
        )

    output: list[dict[str, Any]] = []

    for sequence_id in sorted(grouped):
        group = grouped[sequence_id]

        group.sort(
            key=lambda row: str(row["observed_at"])
        )

        utilization = [
            _numeric(row, "utilization")
            for row in group
        ]

        packet_in = [
            _numeric(
                row,
                "processed_packet_in_rate",
            )
            for row in group
        ]

        cpu = [
            _numeric(
                row,
                "process_cpu_percent",
            )
            for row in group
        ]

        response = [
            _numeric(
                row,
                "response_p95_ms",
            )
            for row in group
        ]

        for index, source in enumerate(group):
            row = dict(source)

            for field in CURRENT_FEATURES:
                row[field] = _numeric(
                    source,
                    field,
                )

            for lag in LAG_STEPS:
                name = f"utilization_lag_{lag}"

                if index >= lag:
                    row[name] = (
                        utilization[index - lag]
                    )
                else:
                    row[name] = None

            for window in ROLLING_WINDOWS:
                name = f"util_mean_{window}"

                if index + 1 >= window:
                    start = index - window + 1
                    row[name] = _mean(
                        utilization[
                            start:index + 1
                        ]
                    )
                else:
                    row[name] = None

            for window in (5, 10):
                name = f"util_std_{window}"

                if index + 1 >= window:
                    start = index - window + 1
                    row[name] = (
                        _population_std(
                            utilization[
                                start:index + 1
                            ]
                        )
                    )
                else:
                    row[name] = None

            if index + 1 >= 5:
                pin_window = packet_in[
                    index - 4:index + 1
                ]

                row["pin_mean_5"] = _mean(
                    pin_window
                )
                row["pin_std_5"] = (
                    _population_std(
                        pin_window
                    )
                )

                cpu_window = cpu[
                    index - 4:index + 1
                ]

                row["cpu_mean_5"] = _mean(
                    cpu_window
                )

                response_window = response[
                    index - 4:index + 1
                ]

                row["response_mean_5"] = (
                    _mean(response_window)
                )

                row["util_slope_5"] = _slope(
                    utilization[
                        index - 4:index + 1
                    ]
                )
            else:
                row["pin_mean_5"] = None
                row["pin_std_5"] = None
                row["cpu_mean_5"] = None
                row["response_mean_5"] = None
                row["util_slope_5"] = None

            if index >= 1:
                row["util_delta_1"] = (
                    utilization[index]
                    - utilization[index - 1]
                )
            else:
                row["util_delta_1"] = None

            output.append(row)

    return output

def feature_names() -> tuple[str, ...]:
    names: list[str] = list(CURRENT_FEATURES)

    names.extend(
        f"utilization_lag_{lag}"
        for lag in LAG_STEPS
    )

    names.extend(
        f"util_mean_{window}"
        for window in ROLLING_WINDOWS
    )

    names.extend(
        (
            "util_std_5",
            "util_std_10",
            "pin_mean_5",
            "pin_std_5",
            "cpu_mean_5",
            "response_mean_5",
            "util_delta_1",
            "util_slope_5",
        )
    )

    return tuple(names)
