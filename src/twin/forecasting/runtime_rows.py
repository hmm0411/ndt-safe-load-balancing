from __future__ import annotations

from typing import Any, Mapping


def forecast_rows(twin_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build per-controller runtime rows from one TwinState snapshot.

    This mirrors ForecastDataRunner._samples_from_twin_state for the fields
    consumed by feature engineering.
    """

    controllers = twin_state.get("controllers")
    if not isinstance(controllers, list):
        return []

    switches = twin_state.get("switches")
    if not isinstance(switches, list):
        switches = []

    ownership = twin_state.get("ownership")
    if not isinstance(ownership, list):
        ownership = []

    owner_by_switch = {
        str(item.get("switch_id", "")): str(item.get("owner_controller_id", ""))
        for item in ownership
        if isinstance(item, Mapping)
    }

    observed_at = str(twin_state.get("created_at", ""))
    snapshot_id = str(twin_state.get("snapshot_id", ""))

    rows: list[dict[str, Any]] = []

    for controller in controllers:
        if not isinstance(controller, Mapping):
            continue

        controller_id = str(controller.get("controller_id", ""))
        if not controller_id:
            continue

        owned_switch_load_shares = [
            float(switch.get("control_load_share", 0.0))
            for switch in switches
            if isinstance(switch, Mapping)
            and str(switch.get("controller_id", "")) == controller_id
            and owner_by_switch.get(str(switch.get("switch_id", ""))) == controller_id
        ]

        rows.append(
            {
                "controller_id": controller_id,
                "observed_at": observed_at,
                "snapshot_id": snapshot_id,
                "processed_packet_in_rate": float(controller["processed_packet_in_rate"]),
                "flow_mod_rate": float(controller.get("flow_mod_rate", 0.0)),
                "process_cpu_percent": float(controller.get("process_cpu_percent", 0.0)),
                "process_memory_rss_mb": float(controller.get("process_memory_rss_mb", 0.0)),
                "response_p95_ms": float(controller.get("response_p95_ms", 0.0)),
                "managed_switch_count": int(controller.get("managed_switch_count", 0)),
                "safe_capacity_pps": float(controller["safe_capacity_pps"]),
                "utilization": float(controller["utilization"]),
                "max_switch_control_load_share": float(max(owned_switch_load_shares, default=0.0)),
            }
        )

    return rows
