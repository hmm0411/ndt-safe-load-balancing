"""Structural pre-check: can this candidate be executed at all right now?

Runs before any outcome prediction. Every failing reason is collected (not only the
first) so each rejection is traceable. Reason codes are the stable strings of
src/schemas/decision.py. Predicted safety is NOT checked here: that is the Safety Gate.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.schemas.decision import (
    ACTIVE_TRANSACTION_CONFLICT,
    COOLDOWN_ACTIVE,
    DESTINATION_DISCONNECTED,
    MISSING_CAPACITY,
    NOT_CONFIGURED,
    OWNER_MISMATCH,
    REASON_CODES,
    ROLE_STATE_INVALID,
    SOURCE_NOT_AT_RISK,
    SWITCH_DISCONNECTED,
    MigrationCandidate,
)
from src.schemas.twin import TwinState


@dataclass(frozen=True)
class PrecheckContext:
    at_risk: frozenset[str]
    # (switch_id, controller_id) -> {"role": "MASTER"|"SLAVE"|..., "connected": bool}
    role_matrix: Mapping[tuple[str, str], Mapping[str, Any]]
    locked_switches: frozenset[str]
    # Cooldown scope is global (safety_policy_v1): finish time of the last migration.
    last_migration_finished_at: datetime | None
    # None = not configured in the policy: the check is skipped and reported.
    cooldown_seconds: float | None
    now: datetime


@dataclass(frozen=True)
class PrecheckResult:
    candidate_id: str
    eligible: bool
    reasons: tuple[str, ...]
    skipped_checks: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.reasons) - set(REASON_CODES))
        if unknown:
            raise ValueError(f"unknown reason codes {unknown}")
        if self.eligible != (not self.reasons):
            raise ValueError("eligible must be True exactly when there is no reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "skipped_checks": [list(item) for item in self.skipped_checks],
        }


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _connected(entry: Mapping[str, Any] | None) -> bool:
    return entry is not None and entry.get("connected") is True


def precheck_candidate(
    candidate: MigrationCandidate, twin: TwinState, ctx: PrecheckContext
) -> PrecheckResult:
    reasons: list[str] = []
    skipped: list[tuple[str, str]] = []
    src, dst, switch = (
        candidate.source_controller,
        candidate.target_controller,
        candidate.switch_id,
    )

    if src not in ctx.at_risk:
        reasons.append(SOURCE_NOT_AT_RISK)

    owners = {o.switch_id: o.owner_controller_id for o in twin.ownership}
    if owners.get(switch) != src or src == dst:
        reasons.append(OWNER_MISMATCH)

    src_role = ctx.role_matrix.get((switch, src))
    dst_role = ctx.role_matrix.get((switch, dst))
    if not _connected(src_role):
        reasons.append(SWITCH_DISCONNECTED)
    if not _connected(dst_role):
        reasons.append(DESTINATION_DISCONNECTED)
    if src_role is not None and dst_role is not None and (
        _connected(src_role) and _connected(dst_role)
    ) and (src_role.get("role") != "MASTER" or dst_role.get("role") != "SLAVE"):
        reasons.append(ROLE_STATE_INVALID)

    capacity = {c.controller_id: c.safe_capacity_pps for c in twin.controllers}
    if not (_positive(capacity.get(src)) and _positive(capacity.get(dst))):
        reasons.append(MISSING_CAPACITY)

    if switch in ctx.locked_switches:
        reasons.append(ACTIVE_TRANSACTION_CONFLICT)

    if ctx.cooldown_seconds is None:
        skipped.append(("cooldown_clear", NOT_CONFIGURED))
    else:
        last = ctx.last_migration_finished_at
        if last is not None and (ctx.now - last).total_seconds() < ctx.cooldown_seconds:
            reasons.append(COOLDOWN_ACTIVE)

    return PrecheckResult(
        candidate_id=candidate.candidate_id,
        eligible=not reasons,
        reasons=tuple(reasons),
        skipped_checks=tuple(skipped),
    )


def precheck_all(
    candidates: Sequence[MigrationCandidate], twin: TwinState, ctx: PrecheckContext
) -> list[PrecheckResult]:
    return [precheck_candidate(c, twin, ctx) for c in candidates]
