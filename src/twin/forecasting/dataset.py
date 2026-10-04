from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

from src.twin.forecasting.continuity import assign_segments


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 forecast observation timestamp."""

    if not isinstance(value, str) or not value:
        raise ValueError("observed_at must be a non-empty string")

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid observed_at timestamp: {value!r}"
        ) from exc


def assign_sequence_ids(
    records: Iterable[dict[str, Any]],
    *,
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    """
    Sort and segment records within each run/controller stream.

    A sequence never crosses:
    - run_id
    - controller_id
    - a continuity gap greater than max_gap_seconds
    """

    grouped: dict[
        tuple[str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)

    for record in records:
        run_id = str(record.get("run_id", ""))
        controller_id = str(
            record.get("controller_id", "")
        )

        if not run_id:
            raise ValueError("missing run_id")

        if not controller_id:
            raise ValueError("missing controller_id")

        grouped[(run_id, controller_id)].append(
            dict(record)
        )

    output: list[dict[str, Any]] = []

    for key in sorted(grouped):
        run_id, controller_id = key
        group = grouped[key]

        group.sort(
            key=lambda row: parse_timestamp(
                str(row["observed_at"])
            )
        )

        timestamps = [
            row["observed_at"]
            for row in group
        ]

        local_segments = assign_segments(
            timestamps,
            max_gap_seconds=max_gap_seconds,
        )

        for row, segment_id in zip(
            group,
            local_segments,
            strict=True,
        ):
            item = dict(row)

            item["sequence_id"] = (
                f"{run_id}:"
                f"{controller_id}:"
                f"{segment_id}"
            )

            item["sequence_index"] = segment_id

            output.append(item)

    output.sort(
        key=lambda row: (
            str(row["run_id"]),
            str(row["controller_id"]),
            parse_timestamp(str(row["observed_at"])),
        )
    )

    return output
