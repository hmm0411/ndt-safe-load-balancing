"""Enumerate every (switch, destination) migration option for at-risk sources.

Generation only answers "what could be done". It never ranks and never looks at
safety: a candidate is made for each switch owned by an at-risk source and each other
controller, in a fixed natural order, so the output is deterministic.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from src.schemas.decision import MigrationCandidate, make_candidate_id
from src.schemas.twin import TwinState


@dataclass(frozen=True)
class CandidateGeneration:
    candidates: tuple[MigrationCandidate, ...]
    # (switch_id, source) that cannot form a candidate: no valid control_load_share.
    dropped: tuple[tuple[str, str], ...]


def natural_key(name: str) -> tuple[str, int, str]:
    """Sort s2 before s10 (and c2 before c10)."""
    match = re.fullmatch(r"(\D*)(\d+)(.*)", name)
    if not match:
        return (name, -1, "")
    return (match.group(1), int(match.group(2)), match.group(3))


def _valid_share(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0


def generate_candidates(
    twin: TwinState, at_risk: Iterable[str], now: datetime
) -> CandidateGeneration:
    """Raw candidates for every at-risk source; empty when nothing is at risk."""
    utilization = {c.controller_id: c.utilization for c in twin.controllers}
    share = {(s.switch_id, s.controller_id): s.control_load_share for s in twin.switches}
    controllers = sorted(utilization, key=natural_key)
    candidates: list[MigrationCandidate] = []
    dropped: list[tuple[str, str]] = []
    for source in sorted(set(at_risk), key=natural_key):
        if source not in utilization:
            continue
        owned = sorted(
            (o.switch_id for o in twin.ownership if o.owner_controller_id == source),
            key=natural_key,
        )
        for switch_id in owned:
            load_share = share.get((switch_id, source))
            if not _valid_share(load_share) or load_share is None:
                dropped.append((switch_id, source))
                continue
            for target in controllers:
                if target == source:
                    continue
                candidates.append(
                    MigrationCandidate(
                        snapshot_id=twin.snapshot_id,
                        switch_id=switch_id,
                        source_controller=source,
                        target_controller=target,
                        source_utilization=utilization[source],
                        target_utilization=utilization[target],
                        switch_control_load_share=load_share,
                        generated_at=now,
                        candidate_id=make_candidate_id(
                            twin.snapshot_id, switch_id, source, target
                        ),
                    )
                )
    return CandidateGeneration(tuple(candidates), tuple(dropped))
