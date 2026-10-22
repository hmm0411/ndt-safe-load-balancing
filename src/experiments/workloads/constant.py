from __future__ import annotations

import math

from .profile import WorkloadPoint, WorkloadProfile


class ConstantProfile(WorkloadProfile):
    """Hold a fixed utilization (used for the destination controller)."""

    def __init__(self, target: float) -> None:
        if not math.isfinite(target):
            raise ValueError("target must be finite")
        if target < 0:
            raise ValueError("target must be non-negative")
        self.target = target

    def target_at(self, elapsed_seconds: float) -> WorkloadPoint:
        return WorkloadPoint(
            elapsed_seconds=elapsed_seconds,
            target_utilization=self.target,
            phase="hold",
        )
