from __future__ import annotations

from threading import RLock
from typing import Any

from src.schemas.telemetry import ControllerTelemetry, SwitchTelemetry


class CurrentStateStore:
    def __init__(self):
        self._controllers: dict[str, ControllerTelemetry] = {}
        self._switches: dict[tuple[str, str], SwitchTelemetry] = {}
        self._runtime_states: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, str] = {}
        self._latest_snapshot = None
        self._lock = RLock()

    def update_collection(self, controller_id, controller, switches, runtime_state):
        with self._lock:
            self._controllers[controller_id] = controller
            for item in switches:
                self._switches[(item.switch_id, item.controller_id)] = item
            self._runtime_states[controller_id] = dict(runtime_state)
            self._errors.pop(controller_id, None)

    def mark_error(self, controller_id: str, error: Exception | str):
        with self._lock:
            self._errors[controller_id] = str(error)

    def values(self):
        with self._lock:
            return list(self._controllers.values()), list(self._switches.values())

    def role_matrix(self) -> dict[tuple[str, str], dict[str, Any]]:
        with self._lock:
            result: dict[tuple[str, str], dict[str, Any]] = {}
            for controller_id, state in self._runtime_states.items():
                for item in state.get("switches", []):
                    switch_id = str(item.get("switch_id") or f"s{item.get('dpid')}")
                    result[(switch_id, controller_id)] = {
                        "role": str(item.get("role", "UNKNOWN")),
                        "connected": bool(item.get("connected") is True),
                    }
            return result

    def set_snapshot(self, snapshot):
        with self._lock:
            self._latest_snapshot = snapshot

    def snapshot_dict(self):
        with self._lock:
            return self._latest_snapshot.to_dict() if self._latest_snapshot else None

    def errors(self) -> dict[str, str]:
        with self._lock:
            return dict(self._errors)
