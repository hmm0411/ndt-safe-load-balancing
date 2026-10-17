from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ForecastRawSample:
    """
    Raw forecasting sample collected from TwinState.

    Utilization is intentionally not capped at 1.0 because values
    above 1.0 represent controller load exceeding safe capacity.
    """

    run_id: str
    controller_id: str
    observed_at: datetime
    workload_type: str
    workload_phase: str

    processed_packet_in_rate: float
    flow_mod_rate: float
    process_cpu_percent: float
    process_memory_rss_mb: float
    response_p95_ms: float

    managed_switch_count: int
    safe_capacity_pps: float
    utilization: float
    max_switch_control_load_share: float

    age_of_twin_ms: float
    twinning_rate: float
    completeness_ratio: float
    synchronization_jitter_ms: float
    snapshot_valid: bool
    snapshot_id: str

    def __post_init__(self) -> None:
        # Identifiers
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")

        if not self.controller_id.strip():
            raise ValueError("controller_id must not be empty")

        if not self.snapshot_id.strip():
            raise ValueError("snapshot_id must not be empty")

        # Workload metadata
        if not self.workload_type.strip():
            raise ValueError("workload_type must not be empty")

        if not self.workload_phase.strip():
            raise ValueError("workload_phase must not be empty")

        # Structural fields
        if self.managed_switch_count < 0:
            raise ValueError(
                "managed_switch_count must be non-negative"
            )

        # Safe capacity must always be finite and positive.
        if (
            not math.isfinite(self.safe_capacity_pps)
            or self.safe_capacity_pps <= 0
        ):
            raise ValueError(
                "safe_capacity_pps must be finite and positive"
            )

        # Raw telemetry / NDT metrics.
        #
        # Do NOT cap utilization at 1.0:
        # utilization > 1.0 represents overload.
        non_negative_fields = {
            "processed_packet_in_rate":
                self.processed_packet_in_rate,
            "flow_mod_rate":
                self.flow_mod_rate,
            "process_cpu_percent":
                self.process_cpu_percent,
            "process_memory_rss_mb":
                self.process_memory_rss_mb,
            "response_p95_ms":
                self.response_p95_ms,
            "utilization":
                self.utilization,
            "max_switch_control_load_share":
                self.max_switch_control_load_share,
            "age_of_twin_ms":
                self.age_of_twin_ms,
            "twinning_rate":
                self.twinning_rate,
            "completeness_ratio":
                self.completeness_ratio,
            "synchronization_jitter_ms":
                self.synchronization_jitter_ms,
        }

        for name, value in non_negative_fields.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative"
                )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["observed_at"] = self.observed_at.isoformat()
        return data

# Reason codes of a NOT_READY forecast (ForecastPrediction.reason).
INVALID_SNAPSHOT = "INVALID_SNAPSHOT"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
HISTORY_GAP = "HISTORY_GAP"
DUPLICATE_SNAPSHOT = "DUPLICATE_SNAPSHOT"
MISSING_CAPACITY = "MISSING_CAPACITY"
MISSING_FEATURE = "MISSING_FEATURE"


@dataclass(frozen=True)
class ForecastPrediction:
    """Runtime forecast for one controller and one TwinState snapshot.

    NOT_READY predictions carry None in every forecast field, never 0: a 0
    would read as "idle controller" to the planner.
    """

    snapshot_id: str
    controller_id: str
    status: str  # "READY" | "NOT_READY"
    reason: str | None
    model_name: str
    model_version: str
    current_utilization: float | None = None
    predicted_utilization: float | None = None
    predicted_packet_in_rate: float | None = None
    error_margin_utilization: float | None = None
    upper_utilization_bound: float | None = None
    overload_risk: bool | None = None
    ood_fallback: bool = False
    ood_features: tuple[str, ...] = ()
    inference_time_ms: float | None = None
    quality_score: float | None = None
    consistency_score: float | None = None
    horizon_seconds: float | None = None
    predicted_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ood_features"] = list(self.ood_features)
        return data
