from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class CapacityModel:
    def __init__(
        self,
        capacities: dict[str, float],
    ) -> None:
        if not capacities:
            raise ValueError(
                "capacities must not be empty"
            )

        for controller_id, capacity in (
            capacities.items()
        ):
            if not controller_id:
                raise ValueError(
                    "controller id must not be empty"
                )

            if capacity <= 0:
                raise ValueError(
                    "safe capacity must be > 0"
                )

        self._capacities = dict(
            capacities
        )

    @classmethod
    def from_file(
        cls,
        path: Path,
    ) -> "CapacityModel":
        payload: dict[str, Any] = (
            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        )

        controllers = payload.get(
            "controllers"
        )

        if not isinstance(
            controllers,
            dict,
        ):
            raise ValueError(
                "controllers must be a mapping"
            )

        capacities = {
            controller_id:
                float(
                    item["c_safe_pps"]
                )
            for controller_id, item
            in controllers.items()
        }

        return cls(
            capacities
        )

    def safe_capacity(
        self,
        controller_id: str,
    ) -> float:
        try:
            return self._capacities[
                controller_id
            ]
        except KeyError as exc:
            raise KeyError(
                "unknown controller: "
                f"{controller_id}"
            ) from exc

    def utilization(
        self,
        controller_id: str,
        processed_packet_in_rate: float,
    ) -> float:
        if processed_packet_in_rate < 0:
            raise ValueError(
                "processed_packet_in_rate "
                "must be >= 0"
            )

        capacity = self.safe_capacity(
            controller_id
        )

        return (
            processed_packet_in_rate
            / capacity
        )


CAPACITY_ENV = "NDT_CAPACITY_FILE"
DEFAULT_CAPACITY_PATH = Path("data/benchmarks/controller_capacity.json")


def resolve_capacity_path(
    root: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Capacity file of the orchestrator; NDT_CAPACITY_FILE overrides it."""
    source = os.environ if env is None else env
    value = source.get(CAPACITY_ENV)
    if not value:
        return root / DEFAULT_CAPACITY_PATH
    path = Path(value)
    return path if path.is_absolute() else root / path
