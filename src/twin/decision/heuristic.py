"""Action choice of the baselines Threshold and Forecast-only (protocol 3).

The heaviest switch of the most loaded at-risk source goes to the least loaded destination,
if a simple room check passes: the destination's utilization after the move, by load
conservation, stays at or below ``room_limit``

    U_dst_after = U_dst + share_s * U_src * C_src / C_dst

No forecast of the outcome, no margin, no Safety Gate, no ranking: that is what the NDT
pipeline adds. Candidates come from the same generator and pre-check as the pipeline
(ownership, connection, locks, cooldown), so the baselines never try an action the executor
would refuse.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.schemas.decision import MigrationCandidate
from src.twin.decision.candidate_generator import natural_key

NO_CANDIDATE = "NO_CANDIDATE"
NO_ROOM = "NO_ROOM"


@dataclass(frozen=True)
class HeuristicChoice:
    candidate: MigrationCandidate | None
    predicted_dst_utilization: float | None
    predicted_src_utilization: float | None
    reason: str | None


def choose(candidates: Sequence[MigrationCandidate], capacity_pps: Mapping[str, float],
           room_limit: float) -> HeuristicChoice:
    if not candidates:
        return HeuristicChoice(None, None, None, NO_CANDIDATE)
    source = min({c.source_controller for c in candidates},
                 key=lambda s: (-max(c.source_utilization for c in candidates
                                     if c.source_controller == s), natural_key(s)))
    own = [c for c in candidates if c.source_controller == source]
    switch = min({c.switch_id for c in own},
                 key=lambda w: (-max(c.switch_control_load_share for c in own
                                     if c.switch_id == w), natural_key(w)))
    options = [c for c in own if c.switch_id == switch]
    best = min(options, key=lambda c: (c.target_utilization, natural_key(c.target_controller)))
    c_src = capacity_pps.get(best.source_controller)
    c_dst = capacity_pps.get(best.target_controller)
    if not c_src or not c_dst or c_src <= 0 or c_dst <= 0:
        raise ValueError(f"no safe capacity for {best.source_controller} or "
                         f"{best.target_controller}")
    moved = best.switch_control_load_share * best.source_utilization * c_src
    dst_after = best.target_utilization + moved / c_dst
    src_after = max(0.0, best.source_utilization - moved / c_src)
    if dst_after > room_limit:
        return HeuristicChoice(None, dst_after, src_after, NO_ROOM)
    return HeuristicChoice(best, dst_after, src_after, None)
