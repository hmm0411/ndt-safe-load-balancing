"""Out-of-distribution gate for the runtime forecaster.

Ranges come from the TRAIN split: per feature [min, max], widened by
tolerance_fraction * (max - min) on each side. The gate watches the model
features AND the monitor-only columns (features.AUDIT_ONLY_FEATURES:
managed_switch_count, max_switch_control_load_share). Those two are not model
inputs because they never vary in training, which is exactly why a change in
them means the model is outside the conditions it was trained on (hot-switch:
load share ~0.54 vs 0.100-0.107 in train; any migration changes the switch
count). A sample is out of range when
any feature leaves its widened range. The gate turns on at the
``consecutive_steps``-th out-of-range sample in a row and turns off as soon as
a sample is back in range. While it is on, the runtime falls back to
persistence (U_hat = current U, ood_fallback = true).

The tree models extrapolate badly on hot-switch (forecast-v1 XGBoost RMSE
0.0775 vs persistence 0.0293), which is what this gate is for. Watching only
the 27 model features, the gate never fired on hot-switch (D_OOD = 0): every
model feature of hot-switch stays inside the widened train range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.twin.forecasting.features import AUDIT_ONLY_FEATURES

METHOD = "train_min_max"

# Watched by the gate, never fed to the model.
MONITOR_ONLY_FEATURES: tuple[str, ...] = AUDIT_ONLY_FEATURES


def monitored_names(
    feature_names: Sequence[str],
    monitor_only: Sequence[str] = MONITOR_ONLY_FEATURES,
) -> tuple[str, ...]:
    """Model features (feature_schema order) followed by the monitor-only columns."""
    names = tuple(feature_names)
    return names + tuple(n for n in monitor_only if n not in names)


def monitored_row(
    model_vector: Sequence[float],
    source: Mapping[str, Any],
    monitor_only: Sequence[str],
) -> list[float]:
    """Runtime row for the gate: the model vector plus the monitor-only values
    read from the TwinState row (forecast_rows adds max_switch_control_load_share)."""
    return [float(v) for v in model_vector] + [float(source[n]) for n in monitor_only]


@dataclass(frozen=True)
class FeatureRanges:
    feature_names: tuple[str, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    tolerance_fraction: float = 0.10
    consecutive_steps: int = 3
    # Trailing entries of feature_names that are watched but are not model inputs.
    monitor_only: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        k = len(self.monitor_only)
        if k and tuple(self.feature_names[n - k:]) != tuple(self.monitor_only):
            raise ValueError("monitor_only must be the trailing feature_names")
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
        monitor_only: Sequence[str] = (),
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
            tuple(monitor_only),
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
            "monitor_only_features": list(self.monitor_only),
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
        (feature_schema.json) followed by ``monitor_only_features``. JSON key
        order carries no meaning: the file is written with sorted keys. A file
        without ``monitor_only_features`` watches the model features only."""
        if data.get("method") != METHOD:
            raise ValueError(f"unsupported feature_ranges method: {data.get('method')!r}")
        features = data["features"]
        monitor_only = tuple(str(n) for n in data.get("monitor_only_features", ()))
        names = monitored_names(feature_names, monitor_only)
        if set(features) != set(names):
            raise ValueError(
                "feature_ranges.json features differ from feature_schema.json "
                "+ monitor_only_features"
            )
        return cls(
            names,
            tuple(float(features[n]["min"]) for n in names),
            tuple(float(features[n]["max"]) for n in names),
            float(data["tolerance_fraction"]),
            int(data["consecutive_steps"]),
            monitor_only,
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
