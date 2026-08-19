from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

from src.schemas.telemetry import ControllerTelemetry, SwitchTelemetry


def parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ControllerCollection:
    controller: ControllerTelemetry
    switches: list[SwitchTelemetry]
    runtime_state: dict[str, Any]


class TelemetryCollector:
    """Poll one controller at a time so one failed controller does not block the rest."""

    def __init__(self, controller_urls: dict[str, str], timeout_seconds: float = 1.0):
        self.controller_urls = {
            controller_id: url.rstrip("/")
            for controller_id, url in controller_urls.items()
        }
        self.timeout_seconds = float(timeout_seconds)

    def _get_json(self, controller_id: str, path: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.controller_urls[controller_id]}{path}",
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def collect_controller(self, controller_id: str) -> ControllerCollection:
        telemetry_payload = self._get_json(controller_id, "/api/v1/telemetry")
        ingested_at = datetime.now(timezone.utc)  # immediately after telemetry arrives
        source = telemetry_payload.get("controller")
        if not source:
            raise ValueError(f"controller:{controller_id}:telemetry_missing")

        controller = ControllerTelemetry(
            controller_id=controller_id,
            observed_at=parse_dt(source["observed_at"]),
            ingested_at=ingested_at,
            packet_in_total=int(source["packet_in_total"]),
            packet_in_rate=float(source["packet_in_rate"]),
            processed_packet_in_total=int(source["processed_packet_in_total"]),
            processed_packet_in_rate=float(source["processed_packet_in_rate"]),
            flow_mod_total=int(source["flow_mod_total"]),
            flow_mod_rate=float(source["flow_mod_rate"]),
            process_cpu_percent=float(source["process_cpu_percent"]),
            process_memory_rss_mb=float(source["process_memory_rss_mb"]),
            response_mean_ms=float(source["response_mean_ms"]),
            response_p95_ms=float(source["response_p95_ms"]),
            managed_switch_count=int(source["managed_switch_count"]),
        )

        switches = [
            SwitchTelemetry(
                switch_id=str(item["switch_id"]),
                controller_id=controller_id,
                observed_at=parse_dt(item["observed_at"]),
                packet_in_total=int(item["packet_in_total"]),
                packet_in_rate=float(item["packet_in_rate"]),
                processed_packet_in_total=int(item["processed_packet_in_total"]),
                processed_packet_in_rate=float(item["processed_packet_in_rate"]),
                flow_mod_total=int(item["flow_mod_total"]),
                flow_mod_rate=float(item["flow_mod_rate"]),
                control_load_share=float(item["control_load_share"]),
            )
            for item in telemetry_payload.get("switches", [])
        ]

        # Runtime role/connectivity is intentionally collected separately from telemetry.
        # Snapshot consistency is based on observed controller state, not inferred ownership.
        runtime_state = self._get_json(controller_id, "/api/v1/state")
        return ControllerCollection(controller, switches, runtime_state)
