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


def _safe_rate(current: int, previous: int, interval_s: float) -> float:
    if interval_s <= 0:
        return 0.0
    delta = current - previous
    return max(0.0, delta / interval_s)


class TelemetryAgent:
    """In-process Ryu telemetry counters.

    Total counters are monotonic during one controller process lifetime. Only the
    response-time window is cleared after each sample. This object is the single
    source of truth for verification and the telemetry REST API.
    """

    def __init__(self, controller_id: str, logger):
        self.controller_id = controller_id
        self.logger = logger
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)  # prime psutil's first sample

        self._packet_in_total = 0
        self._processed_packet_in_total = 0
        self._flow_mod_total = 0

        self._packet_in_by_switch: defaultdict[int, int] = defaultdict(int)
        self._processed_packet_in_by_switch: defaultdict[int, int] = defaultdict(int)
        self._flow_mod_by_switch: defaultdict[int, int] = defaultdict(int)

        self._response_times_ms: list[float] = []

        self._prev_packet_in_total = 0
        self._prev_processed_packet_in_total = 0
        self._prev_flow_mod_total = 0
        self._prev_packet_in_by_switch: defaultdict[int, int] = defaultdict(int)
        self._prev_processed_packet_in_by_switch: defaultdict[int, int] = defaultdict(int)
        self._prev_flow_mod_by_switch: defaultdict[int, int] = defaultdict(int)

        self._last_sample_mono = monotonic()
        self._latest: dict[str, Any] = {"controller": None, "switches": []}
        self._lock = RLock()

    def register_switch(self, dpid: int) -> None:
        """Ensure a quiet connected switch is still represented with zero rates."""
        with self._lock:
            self._packet_in_by_switch[int(dpid)] += 0
            self._processed_packet_in_by_switch[int(dpid)] += 0
            self._flow_mod_by_switch[int(dpid)] += 0

    def record_packet_in(self, dpid: int) -> None:
        """Record every Packet-In received by this controller."""
        with self._lock:
            self._packet_in_total += 1
            self._packet_in_by_switch[int(dpid)] += 1

    def record_processed_packet_in(self, dpid: int) -> None:
        """Record a Packet-In that passed the role check and was processed."""
        with self._lock:
            self._processed_packet_in_total += 1
            self._processed_packet_in_by_switch[int(dpid)] += 1

    def record_flow_mod(self, dpid: int) -> None:
        """Record a Flow-Mod sent by the reactive controller."""
        with self._lock:
            self._flow_mod_total += 1
            self._flow_mod_by_switch[int(dpid)] += 1

    def record_response_time(self, value_ms: float) -> None:
        with self._lock:
            self._response_times_ms.append(float(value_ms))

    def sample(self, managed_switch_count: int) -> dict[str, Any]:
        now_mono = monotonic()
        observed_at = datetime.now(timezone.utc)
        interval_s = max(now_mono - self._last_sample_mono, 1e-9)

        with self._lock:
            packet_in_rate = _safe_rate(
                self._packet_in_total, self._prev_packet_in_total, interval_s
            )
            processed_packet_in_rate = _safe_rate(
                self._processed_packet_in_total,
                self._prev_processed_packet_in_total,
                interval_s,
            )
            flow_mod_rate = _safe_rate(
                self._flow_mod_total, self._prev_flow_mod_total, interval_s
            )

            latency_values = list(self._response_times_ms)
            response_mean_ms = (
                sum(latency_values) / len(latency_values) if latency_values else 0.0
            )
            response_p95_ms = percentile(latency_values, 0.95)

            controller = {
                "controller_id": self.controller_id,
                "observed_at": observed_at.isoformat(),
                "packet_in_total": self._packet_in_total,
                "packet_in_rate": packet_in_rate,
                "processed_packet_in_total": self._processed_packet_in_total,
                "processed_packet_in_rate": processed_packet_in_rate,
                "flow_mod_total": self._flow_mod_total,
                "flow_mod_rate": flow_mod_rate,
                "process_cpu_percent": float(self._process.cpu_percent(interval=None)),
                "process_memory_rss_mb": float(
                    self._process.memory_info().rss / (1024 * 1024)
                ),
                "response_mean_ms": response_mean_ms,
                "response_p95_ms": response_p95_ms,
                "managed_switch_count": int(managed_switch_count),
                "sample_interval_ms": interval_s * 1000.0,
            }

            switch_samples: list[dict[str, Any]] = []
            switch_ids = sorted(
                set(self._packet_in_by_switch)
                | set(self._processed_packet_in_by_switch)
                | set(self._flow_mod_by_switch)
            )

            for dpid in switch_ids:
                packet_in_total = self._packet_in_by_switch[dpid]
                processed_packet_in_total = self._processed_packet_in_by_switch[dpid]
                flow_mod_total = self._flow_mod_by_switch[dpid]

                switch_packet_in_rate = _safe_rate(
                    packet_in_total, self._prev_packet_in_by_switch[dpid], interval_s
                )
                switch_processed_packet_in_rate = _safe_rate(
                    processed_packet_in_total,
                    self._prev_processed_packet_in_by_switch[dpid],
                    interval_s,
                )
                switch_flow_mod_rate = _safe_rate(
                    flow_mod_total, self._prev_flow_mod_by_switch[dpid], interval_s
                )
                control_load_share = (
                    switch_processed_packet_in_rate / processed_packet_in_rate
                    if processed_packet_in_rate > 0
                    else 0.0
                )

                switch_samples.append(
                    {
                        "switch_id": f"s{dpid}",
                        "controller_id": self.controller_id,
                        "observed_at": observed_at.isoformat(),
                        "packet_in_total": packet_in_total,
                        "packet_in_rate": switch_packet_in_rate,
                        "processed_packet_in_total": processed_packet_in_total,
                        "processed_packet_in_rate": switch_processed_packet_in_rate,
                        "flow_mod_total": flow_mod_total,
                        "flow_mod_rate": switch_flow_mod_rate,
                        "control_load_share": control_load_share,
                    }
                )

                self._prev_packet_in_by_switch[dpid] = packet_in_total
                self._prev_processed_packet_in_by_switch[dpid] = processed_packet_in_total
                self._prev_flow_mod_by_switch[dpid] = flow_mod_total

            self._prev_packet_in_total = self._packet_in_total
            self._prev_processed_packet_in_total = self._processed_packet_in_total
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
                "switches": [dict(item) for item in self._latest["switches"]],
            }