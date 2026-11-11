"""Safety metrics of the Gate (Week 13, Day 3): UCRR, UAR, FRR, SAP, DOAR, RVR.

Ground truth is the OBSERVED outcome of an executed migration: it is unsafe when the
destination utilization after it is above the limit (boundary inclusive: U = 1 is safe) or,
when R95_max is configured, when the destination P95 response is above it. The verdict
(accepted) comes from the Safety Gate and is never recomputed here. Every rate keeps its
counts and a Wilson 95 % interval; with an empty denominator the rate is None, never 0.
Without R95_max the response label does not exist: RVR is None and says NOT_CONFIGURED.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from src.schemas.decision import NOT_CONFIGURED
from src.twin.decision.safety_policy import within_limit

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    numerator: int
    denominator: int
    value: float | None
    low: float | None
    high: float | None


def wilson(successes: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a proportion; needs 0 <= successes <= total, total > 0."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("need 0 <= successes <= total and total > 0")
    p = successes / total
    z2 = z * z
    scale = 1.0 + z2 / total
    centre = (p + z2 / (2 * total)) / scale
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def make_rate(numerator: int, denominator: int) -> Rate:
    if denominator == 0:
        return Rate(numerator, 0, None, None, None)
    low, high = wilson(numerator, denominator)
    return Rate(numerator, denominator, numerator / denominator, low, high)


@dataclass(frozen=True)
class ReplayRecord:
    """One executed migration: the Gate verdict and what was observed afterwards."""

    candidate_id: str
    accepted: bool
    actual_dst_utilization: float
    actual_dst_response_p95_ms: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank")
        values = [self.actual_dst_utilization]
        if self.actual_dst_response_p95_ms is not None:
            values.append(self.actual_dst_response_p95_ms)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("observed utilization and response must be finite and >= 0")


def utilization_violation(record: ReplayRecord, limit: float) -> bool:
    return not within_limit(record.actual_dst_utilization, 0.0, limit)


def response_violation(record: ReplayRecord, limit: float | None) -> bool:
    if limit is None:
        return False
    if record.actual_dst_response_p95_ms is None:
        raise ValueError(f"{record.candidate_id}: R95_max is set but no observed response")
    return not within_limit(record.actual_dst_response_p95_ms, 0.0, limit)


@dataclass(frozen=True)
class SafetyMetrics:
    records: int
    accepted: int
    unsafe: int
    unsafe_rejected: int
    unsafe_accepted: int
    safe_rejected: int
    safe_accepted: int
    ucrr: Rate  # unsafe candidates rejected / unsafe candidates
    uar: Rate  # unsafe candidates accepted / unsafe candidates
    frr: Rate  # safe candidates rejected / safe candidates
    sap: Rate  # accepted candidates that were safe / accepted candidates
    doar: Rate  # accepted with U_dst_post above the limit / accepted
    rvr: Rate | None  # accepted with R95 above R95_max / accepted; None = not configured
    rvr_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    records: Sequence[ReplayRecord], u_limit: float, r_limit: float | None = None
) -> SafetyMetrics:
    """Counts and rates of one Gate configuration over executed migrations."""
    cells = {"ur": 0, "ua": 0, "sr": 0, "sa": 0}
    util_bad = resp_bad = 0
    for rec in records:
        over_u = utilization_violation(rec, u_limit)
        over_r = response_violation(rec, r_limit)
        key = ("u" if over_u or over_r else "s") + ("a" if rec.accepted else "r")
        cells[key] += 1
        if rec.accepted:
            util_bad += over_u
            resp_bad += over_r
    unsafe = cells["ur"] + cells["ua"]
    accepted = cells["ua"] + cells["sa"]
    return SafetyMetrics(
        records=len(records), accepted=accepted, unsafe=unsafe,
        unsafe_rejected=cells["ur"], unsafe_accepted=cells["ua"],
        safe_rejected=cells["sr"], safe_accepted=cells["sa"],
        ucrr=make_rate(cells["ur"], unsafe), uar=make_rate(cells["ua"], unsafe),
        frr=make_rate(cells["sr"], cells["sr"] + cells["sa"]),
        sap=make_rate(cells["sa"], accepted), doar=make_rate(util_bad, accepted),
        rvr=None if r_limit is None else make_rate(resp_bad, accepted),
        rvr_status="OK" if r_limit is not None else NOT_CONFIGURED,
    )
