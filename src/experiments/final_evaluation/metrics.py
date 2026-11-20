"""Episode metrics of the closed-loop evaluation (plan 16-22/11, Day 2; thesis proposal sec. 4).

One episode = one run of one method on one workload and seed; every metric here returns ONE
value per episode (the unit of analysis). Inputs are plain data so the same code scores every
method, including the baselines that never call the twin:

* ``samples``: time-ordered (t_seconds, {controller: U}) at the sampling period ``dt``,
  measurement window only (warm-up already removed);
* ``migrations``: one dict per migration that reached execution, with ``switch_id``,
  ``source``, ``target``, ``status`` (COMMITTED / RESTORED / FAILED / ...), ``started_at``
  (seconds), and, when known, ``u_dst_post`` and ``r95_dst_post_ms`` (observed after it).

U is never clamped. A metric that cannot be computed returns None, never 0.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.experiments.migration.metrics import imbalance

Sample = tuple[float, Mapping[str, float]]
COMMITTED = "COMMITTED"
RESTORED = "RESTORED"


def _check(samples: Sequence[Sample], dt: float) -> None:
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("dt must be a positive number of seconds")
    if not samples:
        raise ValueError("no sample in the measurement window")
    times = [t for t, _ in samples]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("samples must be strictly increasing in time")
    for _, u in samples:
        if not u or not all(math.isfinite(float(v)) for v in u.values()):
            raise ValueError(f"every sample needs finite U for each controller, got {dict(u)}")


def overload_exposure(samples: Sequence[Sample], dt: float, limit: float = 1.0) -> float:
    """OE = sum over controllers and samples of max(0, U - limit) * dt  (U·s).

    Plan check: U = 1.2 for 10 samples of 1 s on one controller -> 0.2 x 10 = 2.0.
    """
    _check(samples, dt)
    return sum(max(0.0, float(v) - limit) * dt for _, u in samples for v in u.values())


def normalized_overload_exposure(samples: Sequence[Sample], dt: float,
                                 limit: float = 1.0) -> float:
    """OE / (C * T): mean excess utilization per controller-second."""
    _check(samples, dt)
    controllers = len(samples[0][1])
    return overload_exposure(samples, dt, limit) / (controllers * len(samples) * dt)


def overload_time(samples: Sequence[Sample], dt: float, limit: float = 1.0) -> float:
    """Seconds during which at least one controller has U >= limit (proposal: thoi gian qua tai)."""
    _check(samples, dt)
    return sum(dt for _, u in samples if max(float(v) for v in u.values()) >= limit)


def max_u(samples: Sequence[Sample]) -> float:
    if not samples:
        raise ValueError("no sample in the measurement window")
    return max(float(v) for _, u in samples for v in u.values())


def mean_imbalance(samples: Sequence[Sample]) -> float:
    """Time mean of the population std of U across controllers (proposal: muc mat can bang)."""
    if not samples:
        raise ValueError("no sample in the measurement window")
    return statistics.fmean(imbalance(u) for _, u in samples)


def ping_pongs(migrations: Sequence[Mapping[str, Any]], window_seconds: float) -> int:
    """Committed migrations that move a switch back to the controller it left less than
    window_seconds earlier (A -> B, then B -> A)."""
    last: dict[str, tuple[float, str, str]] = {}
    count = 0
    for m in sorted((m for m in migrations if m["status"] == COMMITTED),
                    key=lambda m: float(m["started_at"])):
        switch, at = str(m["switch_id"]), float(m["started_at"])
        prev = last.get(switch)
        if prev and prev[2] == m["source"] and prev[1] == m["target"] \
                and at - prev[0] < window_seconds:
            count += 1
        last[switch] = (at, str(m["source"]), str(m["target"]))
    return count


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass(frozen=True)
class EpisodeMetrics:
    overload_time_s: float
    overload_exposure_us: float
    overload_exposure_normalized: float
    max_u: float
    mean_imbalance: float
    migrations: int               # migrations that reached execution
    committed: int
    rollbacks: int                # RESTORED
    success_rate: float | None    # committed / migrations
    unsafe_migrations: int        # committed with U_dst_post > limit or R95 > R95_max
    unsafe_rate: float | None     # DOR: unsafe / committed
    unsafe_unknown: int           # committed without an observed post state
    ping_pongs: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_metrics(
    samples: Sequence[Sample],
    migrations: Sequence[Mapping[str, Any]],
    dt: float,
    limit: float = 1.0,
    r95_max_ms: float | None = None,
    ping_pong_window_s: float = 60.0,
) -> EpisodeMetrics:
    committed = [m for m in migrations if m["status"] == COMMITTED]
    unsafe = unknown = 0
    for m in committed:
        u_post = m.get("u_dst_post")
        r_post = m.get("r95_dst_post_ms")
        if u_post is None and (r95_max_ms is None or r_post is None):
            unknown += 1
            continue
        bad_u = u_post is not None and float(u_post) > limit
        bad_r = r95_max_ms is not None and r_post is not None and float(r_post) > r95_max_ms
        unsafe += int(bad_u or bad_r)
    known = len(committed) - unknown
    return EpisodeMetrics(
        overload_time_s=overload_time(samples, dt, limit),
        overload_exposure_us=overload_exposure(samples, dt, limit),
        overload_exposure_normalized=normalized_overload_exposure(samples, dt, limit),
        max_u=max_u(samples),
        mean_imbalance=mean_imbalance(samples),
        migrations=len(migrations),
        committed=len(committed),
        rollbacks=sum(1 for m in migrations if m["status"] == RESTORED),
        success_rate=_rate(len(committed), len(migrations)),
        unsafe_migrations=unsafe,
        unsafe_rate=_rate(unsafe, known),
        unsafe_unknown=unknown,
        ping_pongs=ping_pongs(migrations, ping_pong_window_s),
    )
