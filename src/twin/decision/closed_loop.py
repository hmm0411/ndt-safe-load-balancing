"""Closed-loop rules that need no orchestrator (Week 13, Day 5): states, the check
before execution, the observed post-migration window and the decision record.

State machine of one decision (terminal states in capitals, one per record):
  pinned snapshot -> KILL_SWITCH | INVALID_SNAPSHOT | NOT_READY | NO_RISK | DEFERRED
  -> candidates, outcome, Safety Gate, ranking -> NO_SAFE_ACTION | NO_BENEFICIAL_ACTION
  -> SELECTED (dry run stops here) -> revalidation -> ABORT_STALE_DECISION
  -> executor -> COMMITTED | RESTORED | FAILED.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.schemas.decision import (
    ACTIVE_TRANSACTION_CONFLICT,
    COOLDOWN_ACTIVE,
    DESTINATION_ALREADY_AT_RISK,
    DESTINATION_DISCONNECTED,
    INVALID_SNAPSHOT,
    OWNER_MISMATCH,
    MigrationCandidate,
)
from src.schemas.twin import TwinState

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "decision" / "closed_loop.yaml"

KILL_SWITCH = "KILL_SWITCH"
NOT_READY = "NOT_READY"
NO_RISK = "NO_RISK"
DEFERRED = "DEFERRED"
SELECTED = "SELECTED"
ABORT_STALE_DECISION = "ABORT_STALE_DECISION"
COMMITTED = "COMMITTED"
RESTORED = "RESTORED"
FAILED = "FAILED"


@dataclass(frozen=True)
class ClosedLoopConfig:
    version: str
    enabled: bool
    record_dir: str
    post_settle_seconds: float
    post_window_valid_snapshots: int
    max_cached_decisions: int


def parse_config(data: Mapping[str, Any]) -> ClosedLoopConfig:
    version = str(data.get("version") or "").strip()
    record_dir = str(data.get("record_dir") or "").strip()
    if not version or not record_dir:
        raise ValueError("closed loop config needs 'version' and 'record_dir'")
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be true or false")
    settle = data.get("post_settle_seconds")
    if isinstance(settle, bool) or not isinstance(settle, (int, float)) or settle < 0:
        raise ValueError("post_settle_seconds must be a number >= 0")
    window = data.get("post_window_valid_snapshots")
    cache = data.get("max_cached_decisions")
    for name, value in (("post_window_valid_snapshots", window), ("max_cached_decisions", cache)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    assert isinstance(window, int) and isinstance(cache, int)
    return ClosedLoopConfig(version, enabled, record_dir, float(settle), window, cache)


def load_config(path: Path = CONFIG_PATH) -> ClosedLoopConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return parse_config(data)


def revalidate(
    candidate: MigrationCandidate,
    twin: TwinState,
    role_matrix: Mapping[tuple[str, str], Mapping[str, Any]],
    destination_forecast_risk: bool | None,
    active_transaction: bool,
    seconds_since_last_migration: float | None,
    cooldown_seconds: float | None,
) -> tuple[str, ...]:
    """Reasons the selected action must NOT be executed on the state as it is NOW.

    Runs after ranking and right before the executor, on the latest snapshot (which may
    be newer than the pinned one). Empty = execute. An unknown destination forecast
    (None) is not a reason: the Gate already recorded it as NOT_AVAILABLE.
    """
    reasons: list[str] = []
    if not twin.quality.valid:
        reasons.append(INVALID_SNAPSHOT)
    owners = {o.switch_id: o.owner_controller_id for o in twin.ownership}
    if owners.get(candidate.switch_id) != candidate.source_controller:
        reasons.append(OWNER_MISMATCH)
    dst_role = role_matrix.get((candidate.switch_id, candidate.target_controller)) or {}
    if dst_role.get("connected") is not True:
        reasons.append(DESTINATION_DISCONNECTED)
    if active_transaction:
        reasons.append(ACTIVE_TRANSACTION_CONFLICT)
    if (
        cooldown_seconds is not None
        and seconds_since_last_migration is not None
        and seconds_since_last_migration < cooldown_seconds
    ):
        reasons.append(COOLDOWN_ACTIVE)
    if destination_forecast_risk is True:
        reasons.append(DESTINATION_ALREADY_AT_RISK)
    return tuple(reasons)


def _time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class PostWindow:
    """Observed outcome of one COMMITTED migration, from the snapshots that follow it."""

    def __init__(self, decision_id: str, source: str, target: str, finished_at: datetime,
                 settle_seconds: float, window: int) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.decision_id = decision_id
        self.source = source
        self.target = target
        self.not_before = finished_at + timedelta(seconds=settle_seconds)
        self.window = window
        self.samples: list[tuple[str, float, float]] = []

    def observe(self, twin_state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Feed one TwinState dict; returns actual_post once the window is complete."""
        if len(self.samples) >= self.window:
            return None
        quality = twin_state.get("quality") or {}
        if quality.get("valid") is not True:
            return None
        if _time(twin_state["created_at"]) < self.not_before:
            return None
        utilization = {
            str(c["controller_id"]): float(c["utilization"])
            for c in twin_state.get("controllers") or []
        }
        if self.source not in utilization or self.target not in utilization:
            return None
        snapshot_id = str(twin_state.get("snapshot_id"))
        if any(snapshot_id == s[0] for s in self.samples):
            return None
        self.samples.append((snapshot_id, utilization[self.source], utilization[self.target]))
        if len(self.samples) < self.window:
            return None
        u_src = sum(s[1] for s in self.samples) / self.window
        u_dst = sum(s[2] for s in self.samples) / self.window
        return {
            "u_src": u_src,
            "u_dst": u_dst,
            "i": abs(u_src - u_dst) / 2.0,  # same definition as the i_post target (2C)
            "snapshot_ids": [s[0] for s in self.samples],
            "not_before": self.not_before.isoformat(),
        }


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_clean(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_record(root: Path, record: Mapping[str, Any]) -> Path:
    """Write <root>/<decision_id>/record.json atomically (rewrite = same path)."""
    decision_id = str(record["decision_id"])
    directory = Path(root) / decision_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "record.json"
    tmp = directory / f".record.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(_clean(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path
