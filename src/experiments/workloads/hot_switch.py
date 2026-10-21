from __future__ import annotations

from .profile import WorkloadPoint, WorkloadProfile

# Three agreed hot_switch_share levels (Week 11 Day 1).
# 0.55 was the only level measured in Week 10 (measured ~0.54).
HOT_SWITCH_LEVELS: dict[str, float] = {
    "moderate": 0.25,
    "high": 0.40,
    "peak": 0.55,
}


class HotSwitchWorkload(WorkloadProfile):
    def __init__(
        self,
        target_utilization: float = 0.80,
        hot_switch_id: str = "s1",
        hot_switch_share: float = 0.55,  # generation target, not the measured share
        level: str | None = None,
    ) -> None:
        if level is not None:
            if level not in HOT_SWITCH_LEVELS:
                raise ValueError(
                    f"unknown hot-switch level: {level!r}; "
                    f"expected one of {sorted(HOT_SWITCH_LEVELS)}"
                )
            hot_switch_share = HOT_SWITCH_LEVELS[level]

        self.target_utilization = target_utilization
        self.hot_switch_id = hot_switch_id
        self.hot_switch_share = hot_switch_share
        self.level = level

    def target_at(
        self,
        elapsed_seconds: float,
    ) -> WorkloadPoint:
        return WorkloadPoint(
            elapsed_seconds=elapsed_seconds,
            target_utilization=self.target_utilization,
            phase=self.level or "hot_switch",
            hot_switch_id=self.hot_switch_id,
            hot_switch_share=self.hot_switch_share,
        )
