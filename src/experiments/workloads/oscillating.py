from __future__ import annotations

import math

from .profile import WorkloadPoint, WorkloadProfile


class OscillatingWorkload(WorkloadProfile):
    def __init__(
        self,
        baseline: float = 0.70,
        amplitude: float = 0.20,
        period_seconds: float = 20.0,
        phased: bool = False,
        peak_fraction: float = 0.8,
    ) -> None:
        if period_seconds <= 0:
            raise ValueError(
                "period_seconds must be positive"
            )

        self.baseline = baseline
        self.amplitude = amplitude
        self.period_seconds = period_seconds
        self.phased = phased
        self.peak_fraction = peak_fraction

    def _phase(self, elapsed_seconds: float) -> str:
        # Old behaviour (phased=False): single phase "oscillating".
        if not self.phased:
            return "oscillating"
        angle = 2.0 * math.pi * elapsed_seconds / self.period_seconds
        if math.sin(angle) >= self.peak_fraction:
            return "peak"
        return "rising" if math.cos(angle) > 0 else "falling"

    def target_at(
        self,
        elapsed_seconds: float,
    ) -> WorkloadPoint:

        utilization = (
            self.baseline
            + self.amplitude
            * math.sin(
                2.0
                * math.pi
                * elapsed_seconds
                / self.period_seconds
            )
        )

        return WorkloadPoint(
            elapsed_seconds=elapsed_seconds,
            target_utilization=utilization,
            phase=self._phase(elapsed_seconds),
        )
