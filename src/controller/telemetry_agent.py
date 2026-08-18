from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from math import ceil
from threading import RLock
from time import monotonic
from typing import Any
import psutil

def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, ceil(q * len(ordered)))
    return float(ordered[min(rank - 1, len(ordered) - 1)])

class TelemetryAgent:
    def __init__(self, controller_id: str, logger):
        self.controller_id = controller_id
        self.logger = logger
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)

        self._packet_in_total = 0
        self._flow_mod_total = 0
        self._packet_in_by_switch = defaultdict(int)
        self._flow_mod_by_switch = defaultdict(int)
        self._processed_packet_in_total = 0
        self._processed_packet_in_by_switch = defaultdict(int)
        self._response_times_ms: list[float] = []

        self._prev_packet_in_total = 0
        self._prev_flow_mod_total = 0
        self._prev_packet_in_by_switch = defaultdict(int)
        self._prev_flow_mod_by_switch = defaultdict(int)

        self._last_sample_mono = monotonic()
        self._latest = {"controller": None, "switches": []}
        self._lock = RLock()

    def record_packet_in(self, dpid: int) -> None:
        with self._lock:
            self._packet_in_total += 1
            self._packet_in_by_switch[dpid] += 1

    def record_processed_packet_in(self, dpid: int) -> None:
        with self._lock:
            self._processed_packet_in_total += 1
            self._processed_packet_in_by_switch[dpid] += 1

    def record_flow_mod(self, dpid: int) -> None:
        with self._lock:
            self._flow_mod_total += 1
            self._flow_mod_by_switch[dpid] += 1

    def record_response_time(self, value_ms: float) -> None:
        with self._lock:
            self._response_times_ms.append(float(value_ms))

    def sample(self, managed_switch_count: int) -> dict[str, Any]:
        now_mono = monotonic()
        observed_at = datetime.now(timezone.utc)
        interval_s = max(now_mono - self._last_sample_mono, 1e-9)

        with self._lock:
            pin_rate = (self._packet_in_total - self._prev_packet_in_total) / interval_s
            fm_rate = (self._flow_mod_total - self._prev_flow_mod_total) / interval_s

            values = self._response_times_ms
            mean_ms = sum(values) / len(values) if values else 0.0
            p95_ms = percentile(values, 0.95)

            controller = {
                "controller_id": self.controller_id,
                "observed_at": observed_at.isoformat(),
                "packet_in_total": self._packet_in_total,
                "packet_in_rate": pin_rate,
                "processed_packet_in_total": self._processed_packet_in_total,
                "flow_mod_total": self._flow_mod_total,
                "flow_mod_rate": fm_rate,
                "process_cpu_percent": float(self._process.cpu_percent(interval=None)),
                "process_memory_rss_mb": float(self._process.memory_info().rss / (1024 * 1024)),
                "response_mean_ms": mean_ms,
                "response_p95_ms": p95_ms,
                "managed_switch_count": int(managed_switch_count),
                "sample_interval_ms": interval_s * 1000.0,
            }

            switch_samples = []
            switch_ids = sorted(set(self._packet_in_by_switch) | set(self._processed_packet_in_by_switch) | set(self._flow_mod_by_switch))
            for dpid in switch_ids:
                pin_total = self._packet_in_by_switch[dpid]
                fm_total = self._flow_mod_by_switch[dpid]
                s_pin_rate = (pin_total - self._prev_packet_in_by_switch[dpid]) / interval_s
                s_fm_rate = (fm_total - self._prev_flow_mod_by_switch[dpid]) / interval_s
                share = s_pin_rate / pin_rate if pin_rate > 0 else 0.0
                processed_pin_total = self._processed_packet_in_by_switch[dpid]

                switch_samples.append({
                    "switch_id": f"s{dpid}",
                    "controller_id": self.controller_id,
                    "observed_at": observed_at.isoformat(),
                    "packet_in_total": pin_total,
                    "packet_in_rate": s_pin_rate,
                    "processed_packet_in_total": processed_pin_total,
                    "flow_mod_total": fm_total,
                    "flow_mod_rate": s_fm_rate,
                    "control_load_share": share,
                })
                self._prev_packet_in_by_switch[dpid] = pin_total
                self._prev_flow_mod_by_switch[dpid] = fm_total

            self._prev_packet_in_total = self._packet_in_total
            self._prev_flow_mod_total = self._flow_mod_total
            self._response_times_ms = []
            self._last_sample_mono = now_mono
            self._latest = {"controller": controller, "switches": switch_samples}
            return self.latest()

    def latest(self) -> dict[str, Any]:
        with self._lock:
            controller = self._latest["controller"]
            return {
                "controller": dict(controller) if controller else None,
                "switches": [dict(x) for x in self._latest["switches"]],
            }
