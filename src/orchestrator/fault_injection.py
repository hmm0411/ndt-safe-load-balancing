"""Telemetry faults of the fault scenarios D1 and D2 (docs/evaluation-protocol-v1.md, 4.D).

Off unless the orchestrator starts with NDT_FAULT_INJECTION=1 (only the D episodes set it);
when off, every call is refused and the telemetry loop never consults the injector, so the
B and C episodes run the code as before.

  freeze_telemetry (D1, stale snapshot): the controller is not polled for ``seconds``; its
      last sample stays in the store and ages past freshness_max_age_ms.
  drop_telemetry (D2, missing data): the store forgets the controller once, then it is not
      polled for ``seconds``; the snapshot lacks its telemetry.

Controller disconnection, timeouts, conflicts and a dead destination (D3-D6) are real
faults made by the episode runner (signals to the controller process, concurrent requests),
not simulated here.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Lock
from typing import Any

FAULT_KINDS = ("freeze_telemetry", "drop_telemetry")
MAX_FAULT_SECONDS = 600.0


@dataclass
class _Fault:
    kind: str
    controller_id: str
    until: float
    dropped: bool = False


class FaultInjector:
    def __init__(self, enabled: bool, controllers: Iterable[str],
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.enabled = bool(enabled)
        self.controllers = tuple(controllers)
        self._clock = clock
        self._faults: list[_Fault] = []
        self._lock = Lock()

    def inject(self, kind: str, controller_id: str, seconds: float) -> dict[str, Any]:
        if not self.enabled:
            raise PermissionError("fault injection is off (NDT_FAULT_INJECTION=1 to enable)")
        if kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault {kind!r}; expected one of {FAULT_KINDS}")
        if controller_id not in self.controllers:
            raise ValueError(f"unknown controller {controller_id!r}")
        if not math.isfinite(seconds) or not 0.0 < seconds <= MAX_FAULT_SECONDS:
            raise ValueError(f"seconds must be in (0, {MAX_FAULT_SECONDS:.0f}], got {seconds}")
        with self._lock:
            fault = _Fault(kind, controller_id, self._clock() + float(seconds))
            self._faults.append(fault)
            return {"kind": kind, "controller_id": controller_id, "seconds": float(seconds)}

    def _live(self) -> list[_Fault]:
        now = self._clock()
        self._faults = [f for f in self._faults if f.until > now]
        return self._faults

    def skip_poll(self, controller_id: str) -> bool:
        """True while a fault keeps the controller out of the telemetry poll."""
        with self._lock:
            return any(f.controller_id == controller_id for f in self._live())

    def take_drop(self, controller_id: str) -> bool:
        """True once per drop_telemetry fault: the caller then forgets the controller."""
        with self._lock:
            for fault in self._live():
                if (fault.kind == "drop_telemetry" and fault.controller_id == controller_id
                        and not fault.dropped):
                    fault.dropped = True
                    return True
            return False

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            now = self._clock()
            return [{"kind": f.kind, "controller_id": f.controller_id,
                     "remaining_s": round(f.until - now, 3)} for f in self._live()]
