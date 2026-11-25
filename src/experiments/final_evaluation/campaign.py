"""Closed-loop campaign: the episode plan and one row of metrics per episode (protocol v1).

Plan: every (method, workload, seed) once, in an order shuffled by a declared seed, so no
method always runs first or last on the VM (protocol 5: "xao tron thu tu").

Episode row: read back from what the orchestrator already writes, nothing new to log:
  * snapshots  <snapshot_data_dir>/<RUN_ID>.jsonl, one per second: U = processed Packet-In
    rate / c_safe of the capacity file the episode ran with;
  * decision records <record_dir>/<decision_id>/record.json of the episode's run (method,
    status, transaction, actual_post, timings).
The first ``warmup_s`` seconds after the workload start are dropped (protocol 5: 30 s).
Metrics come from metrics.episode_metrics (one value per episode); M1 (response time) is
None while R95 is off in the Safety Gate (Huong 2), never 0.
"""
from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.experiments.final_evaluation.metrics import Sample, episode_metrics

EXECUTED = ("COMMITTED", "RESTORED", "FAILED")


@dataclass(frozen=True)
class PlannedEpisode:
    order: int
    episode_id: str
    method: str
    workload: str
    seed: int


def plan_episodes(methods: Sequence[str], workloads: Sequence[str], seeds: Sequence[int],
                  order_seed: int, prefix: str = "ep") -> list[PlannedEpisode]:
    for name, values in (("methods", methods), ("workloads", workloads), ("seeds", seeds)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be non-empty and unique")
    cells = [(m, w, int(s)) for m in methods for w in workloads for s in seeds]
    random.Random(order_seed).shuffle(cells)
    return [PlannedEpisode(i + 1, f"{prefix}-{m}-{w.replace('_', '-')}-s{s}", m, w, s)
            for i, (m, w, s) in enumerate(cells)]


def _time(value: Any) -> float:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def utilization_samples(snapshots: Iterable[Mapping[str, Any]], capacity: Mapping[str, float],
                        start: float, warmup_s: float) -> tuple[list[Sample], int]:
    """(t since start, {controller: U}) after the warm-up; snapshots that lack a controller
    or a finite rate are skipped and counted (a gap, never a 0)."""
    samples: list[Sample] = []
    skipped = 0
    for snap in snapshots:
        t = _time(snap["created_at"]) - start
        if t < warmup_s:
            continue
        rates = {str(c["controller_id"]): c.get("processed_packet_in_rate")
                 for c in snap.get("controllers") or []}
        if set(rates) != set(capacity) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in rates.values()):
            skipped += 1
            continue
        samples.append((t, {c: float(v) / capacity[c] for c, v in rates.items()
                            if isinstance(v, (int, float))}))
    samples.sort(key=lambda s: s[0])
    deduped = [s for i, s in enumerate(samples) if i == 0 or s[0] > samples[i - 1][0]]
    return deduped, skipped + len(samples) - len(deduped)


def migrations_from_records(records: Iterable[Mapping[str, Any]],
                            start: float) -> list[dict[str, Any]]:
    """One entry per decision that reached the executor, in time order."""
    out = []
    for r in records:
        tx = r.get("transaction")
        if r.get("status") not in EXECUTED or not isinstance(tx, Mapping):
            continue
        sel = r.get("selected") or {}
        post = r.get("actual_post") or {}
        out.append({"switch_id": sel.get("switch_id"), "source": sel.get("source_controller"),
                    "target": sel.get("target_controller"), "status": r["status"],
                    "started_at": _time(r["decided_at"]) - start,
                    "u_dst_post": post.get("u_dst"), "r95_dst_post_ms": None})
    return sorted(out, key=lambda m: m["started_at"])


def decision_latency_p95_ms(records: Iterable[Mapping[str, Any]]) -> float | None:
    """P95 over decisions of the summed stage timings (M8); None without decisions."""
    totals = sorted(sum(float(v) for v in (r.get("timings_ms") or {}).values()
                        if isinstance(v, (int, float))) for r in records)
    if not totals:
        return None
    k = max(0, math.ceil(0.95 * len(totals)) - 1)
    return totals[k]


def episode_row(episode: PlannedEpisode, snapshots: Sequence[Mapping[str, Any]],
                records: Sequence[Mapping[str, Any]], capacity: Mapping[str, float],
                start: float, warmup_s: float = 30.0, dt: float = 1.0,
                limit: float = 1.0) -> dict[str, Any]:
    samples, skipped = utilization_samples(snapshots, capacity, start, warmup_s)
    if not samples:
        raise ValueError(f"{episode.episode_id}: no sample after the warm-up")
    methods = {str(r.get("method")) for r in records if r.get("method") is not None}
    if methods - {episode.method}:
        raise ValueError(f"{episode.episode_id}: records of another method {sorted(methods)}")
    kept = [r for r in records if _time(r["decided_at"]) - start >= warmup_s]
    m = episode_metrics(samples, migrations_from_records(kept, start), dt, limit)
    statuses: dict[str, int] = {}
    for r in kept:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    return {**asdict(episode), **m.to_dict(), "response_p95_ms": None,
            "decision_p95_ms": decision_latency_p95_ms(kept), "decisions": len(kept),
            "samples": len(samples), "skipped_samples": skipped,
            "statuses": json.dumps(statuses, sort_keys=True)}
