from __future__ import annotations

from .profile import WorkloadPoint, WorkloadProfile


class StableWorkload(WorkloadProfile):
    def __init__(
        self,
        target_utilization: float = 0.60,
        duration_seconds: float | None = None,
    ) -> None:
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self.target_utilization = target_utilization
        self.duration_seconds = duration_seconds

    def _phase(self, elapsed_seconds: float) -> str:
        # Old behaviour (no duration given): single phase "stable".
        if self.duration_seconds is None:
            return "stable"
        third = self.duration_seconds / 3.0
        if elapsed_seconds < third:
            return "early"
        if elapsed_seconds < 2.0 * third:
            return "middle"
        return "late"

    def target_at(self, elapsed_seconds: float) -> WorkloadPoint:
        return WorkloadPoint(
            elapsed_seconds=elapsed_seconds,
            target_utilization=self.target_utilization,
            phase=self._phase(elapsed_seconds),
        )
