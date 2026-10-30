"""Targets of the Post-Migration Outcome Model (``outcome-targets-v1``).

Four base targets are learned, two are derived so the outputs cannot contradict
each other (plan sections 5-8):

* ``u_src_post`` / ``u_dst_post``: mean utilization of the source / destination over
  the post window (the Hard-Gate-valid snapshots after the action, not the first
  sample, which can still be transient);
* ``r_src95_post`` / ``r_dst95_post``: mean ``response_p95_ms`` over the post window;
* ``r95_post = max(r_src95_post, r_dst95_post)``: worst controller response;
* ``i_post = imbalance(u_src_post, u_dst_post)``: population std of the controller
  utilizations (2C: |U1 - U2| / 2), from ``src.experiments.migration.metrics``.

U is never clamped: U > 1 is the overload the Safety Gate must see.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.experiments.migration.metrics import imbalance
from src.experiments.migration.schema import StateSnapshotView

TARGET_SCHEMA_VERSION = "outcome-targets-v1"
BASE_TARGETS = ("u_src_post", "u_dst_post", "r_src95_post", "r_dst95_post")
DERIVED_TARGETS = ("r95_post", "i_post")
ALL_TARGETS = BASE_TARGETS + DERIVED_TARGETS


@dataclass(frozen=True)
class OutcomeValues:
    """Base targets (observed or predicted) with the two derived ones."""

    u_src_post: float
    u_dst_post: float
    r_src95_post: float
    r_dst95_post: float

    def __post_init__(self) -> None:
        for name in BASE_TARGETS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.u_src_post < 0 or self.u_dst_post < 0:
            raise ValueError("utilization cannot be negative")

    @property
    def r95_post(self) -> float:
        return max(self.r_src95_post, self.r_dst95_post)

    @property
    def i_post(self) -> float:
        return imbalance({"src": self.u_src_post, "dst": self.u_dst_post})

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {f"{prefix}{name}": float(getattr(self, name)) for name in ALL_TARGETS}


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("post window is empty")
    return sum(values) / len(values)


def targets_from_window(
    after: Sequence[StateSnapshotView], source: str, target: str
) -> OutcomeValues:
    """Observed targets from the post window of one attempt."""
    return OutcomeValues(
        u_src_post=_mean([s.controller(source).utilization for s in after]),
        u_dst_post=_mean([s.controller(target).utilization for s in after]),
        r_src95_post=_mean([s.controller(source).response_p95_ms for s in after]),
        r_dst95_post=_mean([s.controller(target).response_p95_ms for s in after]),
    )


def derive(values: Mapping[str, float]) -> dict[str, float]:
    """r95_post and i_post from a mapping that holds the four base targets."""
    base = OutcomeValues(**{name: float(values[name]) for name in BASE_TARGETS})
    return {"r95_post": base.r95_post, "i_post": base.i_post}


def target_schema() -> dict[str, Any]:
    return {
        "target_schema": TARGET_SCHEMA_VERSION,
        "base_targets": list(BASE_TARGETS),
        "derived_targets": {
            "r95_post": "max(r_src95_post, r_dst95_post)",
            "i_post": "population std of (u_src_post, u_dst_post) = |u_src_post - u_dst_post| / 2",
        },
        "aggregation": "mean over the post window (Hard-Gate-valid snapshots after the action)",
        "units": {"u_*": "processed Packet-In rate / c_safe_pps (not clamped)",
                  "r_*": "controller response_p95_ms"},
        "note": "response_p95_ms is controller processing time; it does not rise with overload "
                "on this testbed (about 0.77 ms healthy, 0.59 ms saturated)",
    }
