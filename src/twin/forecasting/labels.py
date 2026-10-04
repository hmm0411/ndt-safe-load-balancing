from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def attach_future_utilization(
    records: Iterable[dict[str, Any]],
    *,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    """
    Attach U(t+h) within the same continuity sequence.

    All rows are preserved. Rows without a future observation
    at the requested horizon receive future_utilization=None.
    Labels never cross run/controller/sequence boundaries.
    """
    if horizon_steps <= 0:
        raise ValueError(
            "horizon_steps must be positive"
        )

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
                "sequence_id is required before labeling"
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

        for index, source in enumerate(group):
            current = dict(source)

            future_index = (
                index + horizon_steps
            )

            current[
                "forecast_horizon_steps"
            ] = horizon_steps

            if future_index >= len(group):
                current[
                    "future_utilization"
                ] = None

                output.append(current)
                continue

            future_utilization = float(
                group[future_index][
                    "utilization"
                ]
            )

            if not math.isfinite(
                future_utilization
            ):
                raise ValueError(
                    "future utilization must be finite"
                )

            current[
                "future_utilization"
            ] = future_utilization

            output.append(current)

    return output
