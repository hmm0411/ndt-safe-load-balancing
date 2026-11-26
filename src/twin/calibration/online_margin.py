"""Feedback & Calibration: margins updated from observed errors (thesis loop step 6).

Online quantile tracking (the margin form of adaptive conformal inference): after every
observed error e = actual - predicted (signed, upper side: what the Safety Gate needs),

    covered = e <= m
    m      <- max(floor, m + gamma * ((0 if covered else 1) - alpha))

so m grows by gamma * (1 - alpha) after a miss and shrinks by gamma * alpha after a hit; in
the long run the miss rate tends to alpha (Angelopoulos, Candes, Tibshirani 2023, "Conformal
PID control", quantile tracking; Gibbs & Candes 2021, ACI). The start value is the margin
calibrated on validation (error_margin.json / uncertainty_margin.json), so with no
feedback the tracker returns exactly that margin: "NDT without feedback" = the same code
with updates switched off.

State lives in memory for one episode (the protocol resets the twin between episodes).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class TrackerConfig:
    alpha: float = 0.05
    gamma: float = 0.01
    floor: float = 0.0
    cap: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not (math.isfinite(self.gamma) and self.gamma > 0.0):
            raise ValueError("gamma must be > 0")
        if not (math.isfinite(self.floor) and self.floor >= 0.0):
            raise ValueError("floor must be >= 0")
        if self.cap is not None and not (math.isfinite(self.cap) and self.cap >= self.floor):
            raise ValueError("cap must be >= floor")


@dataclass
class OnlineMargin:
    """One margin (one controller's m_f, or m_U)."""

    initial: float
    config: TrackerConfig = field(default_factory=TrackerConfig)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.initial = _finite(self.initial, "initial margin")
        if self.initial < 0:
            raise ValueError("initial margin must be >= 0")
        self.value = self.initial
        self.updates = 0
        self.misses = 0

    def update(self, error: float) -> bool:
        """Feed one signed error (actual - predicted); returns whether it was covered."""
        e = _finite(error, "error")
        covered = e <= self.value
        self.updates += 1
        self.misses += 0 if covered else 1
        if self.enabled:
            step = self.config.gamma * ((0.0 if covered else 1.0) - self.config.alpha)
            value = max(self.config.floor, self.value + step)
            self.value = min(value, self.config.cap) if self.config.cap is not None else value
        return covered

    def to_dict(self) -> dict[str, Any]:
        return {"initial": self.initial, "value": self.value, "updates": self.updates,
                "misses": self.misses, "enabled": self.enabled,
                "alpha": self.config.alpha, "gamma": self.config.gamma}


class MarginTracker:
    """Margins per key (controller id for m_f); unknown keys start from ``default``."""

    def __init__(self, initial: Mapping[str, float], default: float,
                 config: TrackerConfig | None = None, enabled: bool = True) -> None:
        self.config = config or TrackerConfig()
        self.enabled = enabled
        self.default = _finite(default, "default margin")
        self._margins = {str(k): OnlineMargin(v, self.config, enabled)
                         for k, v in initial.items()}

    def _get(self, key: str) -> OnlineMargin:
        if key not in self._margins:
            self._margins[key] = OnlineMargin(self.default, self.config, self.enabled)
        return self._margins[key]

    def value(self, key: str) -> float:
        return self._get(key).value

    def update(self, key: str, error: float) -> bool:
        return self._get(key).update(error)

    def to_dict(self) -> dict[str, Any]:
        return {k: m.to_dict() for k, m in sorted(self._margins.items())}
