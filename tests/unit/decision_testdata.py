from __future__ import annotations

from datetime import datetime, timezone

from src.schemas.snapshot import OwnershipState
from src.schemas.telemetry import SwitchTelemetry
from src.schemas.twin import ControllerTwinState, TwinQuality, TwinState

NOW = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)


def make_twin(
    owners: dict[str, str] | None = None,
    utilization: dict[str, float] | None = None,
    valid: bool = True,
    capacity: dict[str, float] | None = None,
) -> TwinState:
    owners = owners or {"s1": "c1", "s2": "c1", "s3": "c1", "s4": "c1", "s5": "c2"}
    utilization = utilization or {"c1": 0.95, "c2": 0.40}
    capacity = capacity or {"c1": 933.0, "c2": 933.0}
    controllers = [
        ControllerTwinState(
            controller_id=cid,
            processed_packet_in_rate=u * capacity[cid],
            safe_capacity_pps=capacity[cid],
            utilization=u,
            collection_latency_ms=1.0,
            flow_mod_rate=0.0,
            process_cpu_percent=10.0,
            process_memory_rss_mb=100.0,
            response_p95_ms=1.0,
            managed_switch_count=sum(1 for o in owners.values() if o == cid),
        )
        for cid, u in utilization.items()
    ]
    switches = [
        SwitchTelemetry(
            switch_id=sid,
            controller_id=cid,
            observed_at=NOW,
            packet_in_total=0,
            packet_in_rate=0.0,
            processed_packet_in_total=0,
            processed_packet_in_rate=0.0,
            flow_mod_total=0,
            flow_mod_rate=0.0,
            control_load_share=0.1,
        )
        for sid, cid in owners.items()
    ]
    ownership = [
        OwnershipState(sid, cid, "MASTER", "SLAVE", 0, 1) for sid, cid in owners.items()
    ]
    quality = TwinQuality(
        age_of_twin_ms=10.0,
        twinning_rate=1.0,
        completeness_ratio=1.0,
        synchronization_jitter_ms=1.0,
        consistent=True,
        valid=valid,
    )
    return TwinState("snap-1", NOW, 1, 1, controllers, switches, ownership, quality)


def make_roles(twin: TwinState) -> dict[tuple[str, str], dict[str, object]]:
    ids = [c.controller_id for c in twin.controllers]
    roles: dict[tuple[str, str], dict[str, object]] = {}
    for o in twin.ownership:
        for cid in ids:
            role = "MASTER" if cid == o.owner_controller_id else "SLAVE"
            roles[(o.switch_id, cid)] = {"role": role, "connected": True}
    return roles
