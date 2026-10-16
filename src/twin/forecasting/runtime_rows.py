"""TwinState -> one forecasting row per controller (runtime side).

The training data was recorded by ForecastDataRunner._samples_from_twin_state,
which adds max_switch_control_load_share from the switch rows and the
ownership list. TwinState controller rows do not carry that column, and
build_features() requires it, so the runtime predictor must add it the same
way. tests/unit/test_forecast_runtime_rows.py keeps the two in step.
"""
from __future__ import annotations

from typing import Any, Mapping


def max_owned_load_share(twin_state: Mapping[str, Any]) -> dict[str, float]:
    """Largest control_load_share among the switches each controller OWNS.

    A switch row seen by a non-owner (SLAVE role) does not count; a controller
    that owns no switch gets 0.0, as in the data collection.
    """
    owner = {
        str(item["switch_id"]): str(item["owner_controller_id"])
        for item in twin_state.get("ownership", []) or []
    }
    shares: dict[str, float] = {}
    for switch in twin_state.get("switches", []) or []:
        controller_id = str(switch.get("controller_id", ""))
        if owner.get(str(switch.get("switch_id", ""))) != controller_id:
            continue
        share = float(switch.get("control_load_share", 0.0))
        shares[controller_id] = max(shares.get(controller_id, 0.0), share)
    return shares


def forecast_rows(twin_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Controller rows of a TwinState dict, each with max_switch_control_load_share."""
    shares = max_owned_load_share(twin_state)
    rows: list[dict[str, Any]] = []
    for controller in twin_state.get("controllers", []) or []:
        controller_id = str(controller["controller_id"])
        rows.append(
            {**controller, "max_switch_control_load_share": shares.get(controller_id, 0.0)}
        )
    return rows
