"""Out-of-distribution gate for the runtime forecaster.

Ranges come from the TRAIN split: per feature [min, max], widened by
tolerance_fraction * (max - min) on each side. A sample is out of range when
any feature leaves its widened range. The gate turns on at the
``consecutive_steps``-th out-of-range sample in a row and turns off as soon as
a sample is back in range. While it is on, the runtime falls back to
persistence (U_hat = current U, ood_fallback = true).

The tree models extrapolate badly on hot-switch (RMSE 0.065 vs persistence
0.029), which is what this gate is for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

METHOD = "train_min_max"


@dataclass(frozen=True)
class FeatureRanges:
    feature_names: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    tolerance_fraction: float = 0.10
    consecutive_steps: int = 3

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        if len(self.minimum) != n or len(self.maximum) != n:
            raise ValueError("one min and one max per feature")
        if not math.isfinite(self.tolerance_fraction) or self.tolerance_fraction < 0:
            raise ValueError("tolerance_fraction must be finite and >= 0")
        if self.consecutive_steps < 1:
            raise ValueError("consecutive_steps must be >= 1")
        for name, lo, hi in zip(self.feature_names, self.minimum, self.maximum, strict=True):
            if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
                raise ValueError(f"invalid range for {name}: [{lo}, {hi}]")

    @classmethod
    def fit(
        cls,
        matrix: Any,
        feature_names: Sequence[str],
        tolerance_fraction: float = 0.10,
        consecutive_steps: int = 3,
    ) -> FeatureRanges:
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(feature_names) or x.shape[0] == 0:
            raise ValueError("matrix must be (n_samples > 0, n_features)")
        if not np.isfinite(x).all():
            raise ValueError("matrix must be finite")
        return cls(
            tuple(feature_names),
            tuple(float(v) for v in x.min(axis=0)),
            tuple(float(v) for v in x.max(axis=0)),
            tolerance_fraction,
            consecutive_steps,
        )

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray(self.minimum)
        hi = np.asarray(self.maximum)
        pad = self.tolerance_fraction * (hi - lo)
        return lo - pad, hi + pad

    def violations(self, row: Any) -> tuple[str, ...]:
        """Names of the features outside their widened range (empty = in range)."""
        x = np.asarray(row, dtype=float).reshape(-1)
        if x.size != len(self.feature_names):
            raise ValueError(f"row must have {len(self.feature_names)} features")
        lo, hi = self._bounds()
        # NaN counts as out of range.
        outside = ~((x >= lo) & (x <= hi))
        return tuple(name for name, bad in zip(self.feature_names, outside) if bad)

    def out_of_range(self, matrix: Any) -> np.ndarray:
        """Boolean per row of a (n, n_features) matrix."""
        x = np.asarray(matrix, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("matrix must be (n_samples, n_features)")
        lo, hi = self._bounds()
        return np.asarray(~((x >= lo) & (x <= hi)).all(axis=1))

    def to_dict(self) -> dict[str, Any]:
        """feature_ranges.json content (contract C1)."""
        return {
            "method": METHOD,
            "tolerance_fraction": self.tolerance_fraction,
            "consecutive_steps": self.consecutive_steps,
            "features": {
                name: {"min": lo, "max": hi}
                for name, lo, hi in zip(
                    self.feature_names, self.minimum, self.maximum, strict=True
                )
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        feature_names: Sequence[str],
    ) -> FeatureRanges:
        """Load feature_ranges.json in the order of ``feature_names``
        (feature_schema.json). JSON key order carries no meaning: the file is
        written with sorted keys."""
        if data.get("method") != METHOD:
            raise ValueError(f"unsupported feature_ranges method: {data.get('method')!r}")
        features = data["features"]
        names = tuple(feature_names)
        if set(features) != set(names):
            raise ValueError("feature_ranges.json features differ from feature_schema.json")
        return cls(
            names,
            tuple(float(features[n]["min"]) for n in names),
            tuple(float(features[n]["max"]) for n in names),
            float(data["tolerance_fraction"]),
            int(data["consecutive_steps"]),
        )


@dataclass(frozen=True)
class OODDecision:
    out_of_range: bool
    consecutive: int
    active: bool
    features: tuple[str, ...]


class OODGate:
    """Stateful gate, one consecutive counter per key (controller id)."""

    def __init__(self, ranges: FeatureRanges) -> None:
        self.ranges = ranges
        self._consecutive: dict[str, int] = {}

    def update(self, key: str, row: Any) -> OODDecision:
        features = self.ranges.violations(row)
        count = self._consecutive.get(key, 0) + 1 if features else 0
        self._consecutive[key] = count
        return OODDecision(
            out_of_range=bool(features),
            consecutive=count,
            active=count >= self.ranges.consecutive_steps,
            features=features,
        )

    def reset(self, key: str | None = None) -> None:
        if key is None:
            self._consecutive.clear()
        else:
            self._consecutive.pop(key, None)


def gate_mask(
    out_of_range: Any,
    sequence_ids: Sequence[str],
    consecutive_steps: int = 3,
) -> np.ndarray:
    """Offline gate: True where the gate is on. Rows must be time-ordered
    within each sequence; the counter restarts at every new sequence."""
    flags = np.asarray(out_of_range, dtype=bool).reshape(-1)
    if flags.size != len(sequence_ids):
        raise ValueError("one sequence id per row")
    active = np.zeros(flags.size, dtype=bool)
    count = 0
    previous: str | None = None
    for i, (flag, sequence) in enumerate(zip(flags, sequence_ids)):
        if sequence != previous:
            count, previous = 0, sequence
        count = count + 1 if flag else 0
        active[i] = count >= consecutive_steps
    return active
