"""Analytical (load-conservation) outcome model, the domain baseline of Week 12.

The moved switch takes its measured Packet-In load from the source to the
destination (plan sections 13-19):

    lambda_src_post = max(0, lambda_src - lambda_s)      U_src_post = lambda_src_post / C_src
    lambda_dst_post = lambda_dst + lambda_s               U_dst_post = lambda_dst_post / C_dst

lambda are processed Packet-In rates (pps), averaged over the pre window, lambda_s
is the switch's rate reported by the source, C the safe capacities. U is not
clamped. Response has no measured response-vs-load curve on this testbed
(response_p95_ms stays near 0.77 ms and drops when saturated), so the response
baseline is persistence: R_post = mean R of the pre window.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Sequence

from src.experiments.migration.schema import StateSnapshotView
from src.twin.outcome.features import ActionSpec, check_window
from src.twin.outcome.targets import OutcomeValues


@dataclass(frozen=True)
class AnalyticalInputs:
    src_rate_pps: float
    dst_rate_pps: float
    switch_rate_pps: float
    src_capacity_pps: float
    dst_capacity_pps: float
    src_response_p95_ms: float
    dst_response_p95_ms: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.src_capacity_pps <= 0 or self.dst_capacity_pps <= 0:
            raise ValueError("safe capacities must be > 0")
        if min(self.src_rate_pps, self.dst_rate_pps, self.switch_rate_pps) < 0:
            raise ValueError("Packet-In rates cannot be negative")


def predict(inputs: AnalyticalInputs) -> OutcomeValues:
    src_post = max(0.0, inputs.src_rate_pps - inputs.switch_rate_pps)
    dst_post = inputs.dst_rate_pps + inputs.switch_rate_pps
    return OutcomeValues(
        u_src_post=src_post / inputs.src_capacity_pps,
        u_dst_post=dst_post / inputs.dst_capacity_pps,
        r_src95_post=inputs.src_response_p95_ms,
        r_dst95_post=inputs.dst_response_p95_ms,
    )


def inputs_from_window(
    pre_window: Sequence[StateSnapshotView],
    action: ActionSpec | None,
    source: str,
    target: str,
    required_window: int = 5,
) -> AnalyticalInputs:
    """Inputs from the pre window. ``action=None`` is the no-action case (lambda_s = 0)."""
    if action is not None:
        check_window(pre_window, action, required_window)
        if (action.source_controller, action.target_controller) != (source, target):
            raise ValueError("action and source/target disagree")
    elif len(pre_window) < required_window:
        raise ValueError(f"pre window has {len(pre_window)} snapshots, needs {required_window}")
    rates: list[float] = []
    if action is not None:
        rates = [e.processed_packet_in_rate for e in
                 (s.switch_load(action.switch_id, source) for s in pre_window) if e is not None]
        if not rates:
            raise ValueError(f"{action.switch_id}: no switch load in the pre window")
    last = pre_window[-1]
    return AnalyticalInputs(
        src_rate_pps=statistics.fmean(s.controller(source).processed_packet_in_rate
                                      for s in pre_window),
        dst_rate_pps=statistics.fmean(s.controller(target).processed_packet_in_rate
                                      for s in pre_window),
        switch_rate_pps=statistics.fmean(rates) if rates else 0.0,
        src_capacity_pps=last.controller(source).safe_capacity_pps,
        dst_capacity_pps=last.controller(target).safe_capacity_pps,
        src_response_p95_ms=statistics.fmean(s.controller(source).response_p95_ms
                                             for s in pre_window),
        dst_response_p95_ms=statistics.fmean(s.controller(target).response_p95_ms
                                             for s in pre_window),
    )
