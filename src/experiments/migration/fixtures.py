"""Synthetic orchestrator payloads for the migration-outcome tests.

Shaped exactly like ``GET /api/v1/twin/state`` (TwinState.to_dict) and the
200 / 500 responses of ``POST /api/v1/migrations``. Used by unit tests of the
schema, metrics and (Day 2) validator; not used by the runner.
"""
from __future__ import annotations

import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

T0 = datetime(2026, 10, 20, tzinfo=timezone.utc)
SAFE_CAPACITY = 1433.0


def default_ownership(controllers: tuple[str, ...] = ("c1", "c2"), per_controller: int = 10) -> dict[str, str]:
    """2C20S: c1 owns s1-s10, c2 owns s11-s20 (4C40S with 4 ids)."""
    mapping: dict[str, str] = {}
    for index, cid in enumerate(controllers):
        for k in range(per_controller):
            mapping[f"s{index * per_controller + k + 1}"] = cid
    return mapping


def make_twin_state(
    utilization: Mapping[str, float],
    *,
    ownership: Mapping[str, str] | None = None,
    hot_switch: tuple[str, float] | None = None,
    valid: bool = True,
    consistency_score: float = 1.0,
    quality_score: float | None = 0.98,
    index: int = 0,
    ownership_version: int = 1,
    response_p95_ms: float = 0.7,
) -> dict[str, Any]:
    """One TwinState dict. Each controller's processed rate is U * C_safe and is
    split equally over the switches it owns, unless ``hot_switch = (sid, share)``
    gives one switch that share of its owner's load."""
    owners = dict(ownership) if ownership is not None else default_ownership(tuple(utilization))
    created = T0 + timedelta(seconds=index)
    controllers = []
    switches = []
    for cid, u in utilization.items():
        rate = u * SAFE_CAPACITY
        owned = sorted((s for s, o in owners.items() if o == cid), key=lambda s: int(s[1:]))
        controllers.append(
            {
                "controller_id": cid,
                "processed_packet_in_rate": rate,
                "safe_capacity_pps": SAFE_CAPACITY,
                "utilization": u,
                "collection_latency_ms": 12.0,
                "flow_mod_rate": rate,
                "process_cpu_percent": 60.0,
                "process_memory_rss_mb": 56.8,
                "response_p95_ms": response_p95_ms,
                "managed_switch_count": len(owned),
            }
        )
        shares = {s: 1.0 / len(owned) for s in owned} if owned else {}
        if hot_switch is not None and hot_switch[0] in shares:
            hot, share = hot_switch
            rest = (1.0 - share) / max(len(owned) - 1, 1)
            shares = {s: (share if s == hot else rest) for s in owned}
        for sid in owned:
            switches.append(
                {
                    "switch_id": sid,
                    "controller_id": cid,
                    "observed_at": created.isoformat(),
                    "packet_in_total": 1000,
                    "packet_in_rate": rate * shares[sid],
                    "processed_packet_in_total": 1000,
                    "processed_packet_in_rate": rate * shares[sid],
                    "flow_mod_total": 1000,
                    "flow_mod_rate": rate * shares[sid],
                    "control_load_share": shares[sid],
                }
            )
    return {
        "snapshot_id": f"snap-{index:04d}",
        "created_at": created.isoformat(),
        "topology_version": 1,
        "ownership_version": ownership_version,
        "controllers": controllers,
        "switches": switches,
        "ownership": [
            {
                "switch_id": sid,
                "owner_controller_id": owner,
                "source_role": "MASTER",
                "target_role": "SLAVE",
                "generation_id": 1,
                "ownership_version": ownership_version,
            }
            for sid, owner in owners.items()
        ],
        "quality": {
            "age_of_twin_ms": 700.0,
            "twinning_rate": 1.0,
            "completeness_ratio": 1.0,
            "synchronization_jitter_ms": 400.0,
            "consistent": consistency_score == 1.0,
            "valid": valid,
            "consistency_score": consistency_score,
            "quality_score": quality_score,
            "quality_components": {},
        },
    }


def make_window(
    utilizations: list[Mapping[str, float]], start_index: int = 0, **kwargs: Any
) -> list[dict[str, Any]]:
    return [make_twin_state(u, index=start_index + i, **kwargs) for i, u in enumerate(utilizations)]


def _history(states: list[str]) -> list[dict[str, Any]]:
    return [
        {"state": state, "at": (T0 + timedelta(milliseconds=300 * i)).isoformat(), "detail": None}
        for i, state in enumerate(states)
    ]


def _transaction(switch_id: str, source: str, target: str, states: list[str], reason: str | None) -> dict[str, Any]:
    return {
        "transaction_id": "tx-0123456789ab",
        "switch_id": switch_id,
        "source_controller": source,
        "target_controller": target,
        "generation_id": 7,
        "state": states[-1],
        "started_at": T0.isoformat(),
        "finished_at": (T0 + timedelta(seconds=2)).isoformat(),
        "failure_reason": reason,
        "history": _history(states),
    }


def committed_response(switch_id: str = "s1", source: str = "c1", target: str = "c2") -> dict[str, Any]:
    states = ["PREPARING", "ROLE_SWITCHING", "VERIFYING", "COMMITTED"]
    return {
        "status": "COMMITTED",
        "verification": {"ok": True, "reasons": []},
        "transaction": _transaction(switch_id, source, target, states, None),
        "ownership_version": 2,
    }


def restored_response(switch_id: str = "s1", source: str = "c1", target: str = "c2") -> dict[str, Any]:
    states = ["PREPARING", "ROLE_SWITCHING", "VERIFYING", "FAILED", "ROLLING_BACK", "RESTORED"]
    reason = "Verification failed: flow_mod"
    return {
        "status": "RESTORED",
        "failure_reason": reason,
        "rollback_verification": {"ok": True, "reasons": []},
        "transaction": _transaction(switch_id, source, target, states, reason),
        "ownership_version": 2,
    }


def failed_detail(switch_id: str = "s1", source: str = "c1", target: str = "c2") -> dict[str, Any]:
    states = ["PREPARING", "ROLE_SWITCHING", "VERIFYING", "FAILED", "ROLLING_BACK", "FAILED"]
    return {
        "migration_error": "Verification failed: flow_mod",
        "rollback_error": "Rollback verification failed: timeout",
        "transaction": _transaction(switch_id, source, target, states, "rollback timeout"),
    }


def attempt_inputs(
    kind: str = "committed",
    attempt_id: str = "mig-stable-middle-dst45-c1-c2-r01",
    after_count: int = 5,
) -> dict[str, Any]:
    """Keyword arguments for writer.write_attempt, for one synthetic attempt.

    kind: 'committed', 'restored', 'failed', 'skipped' or 'no_op'.
    """
    # Imported here: fixtures stay usable without the writer (Day 1 tests).
    from src.experiments.migration.metrics import compute_outcome, summarize_quality
    from src.experiments.migration.schema import (
        AttemptMetadata,
        MigrationAction,
        StateSnapshotView,
    )
    from src.experiments.migration.writer import raw_transaction, transaction_from_raw

    before_raw = make_window([{"c1": 0.80, "c2": 0.45}] * 5)
    moved = dict(default_ownership())
    moved["s1"] = "c2"
    if kind == "committed":
        after_raw = make_window([{"c1": 0.72, "c2": 0.53}] * after_count, 10,
                                ownership=moved, ownership_version=2)
    else:
        after_raw = make_window([{"c1": 0.80, "c2": 0.45}] * after_count, 10)
    if kind == "skipped":
        after_raw = []

    if kind == "no_op":
        action = MigrationAction("NO_OP", "c1", "c2")
        tx_raw = raw_transaction()
    else:
        action = MigrationAction("MIGRATE", "c1", "c2", switch_id="s1",
                                 failure_injected=(kind == "restored"),
                                 selection_rule="seeded_random")
        tx_raw = {
            "committed": raw_transaction(200, committed_response()),
            "restored": raw_transaction(200, restored_response()),
            "failed": raw_transaction(500, {"detail": failed_detail()}),
            "skipped": raw_transaction(skip_reason="source_not_master"),
        }[kind]
    if kind == "no_op":
        attempt_id = attempt_id.replace("mig-", "noop-", 1)

    before = [StateSnapshotView.from_twin_state(x) for x in before_raw]
    after = [StateSnapshotView.from_twin_state(x) for x in after_raw]
    transaction = transaction_from_raw(tx_raw)
    metadata = AttemptMetadata(
        attempt_id=attempt_id,
        workload="stable",
        phase="middle",
        destination_load_level=0.45,
        source_controller="c1",
        target_controller="c2",
        repeat="r01",
        seed=9201,
        topology="2C20S",
        git_commit="0" * 40,
        config_sha256="f" * 64,
        capacity_version="capacity-v3",
        quality_version="quality-score-v1",
        started_at=T0.isoformat(),
    )
    return {
        "metadata": metadata,
        "before_raw": before_raw,
        "transition_raw": make_window([{"c1": 0.78, "c2": 0.47}], 6),
        "after_raw": after_raw,
        "forecast_raw": [],
        "action": action,
        "transaction_raw": tx_raw,
        "outcome": compute_outcome(before, after, action, transaction),
        "quality": summarize_quality(before, after),
    }


def unlock_tree(root: Path) -> None:
    """Make a tree written by write_attempt deletable again (test cleanup)."""
    for dirpath, _dirnames, filenames in os.walk(root):
        os.chmod(dirpath, stat.S_IRWXU)
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), stat.S_IRUSR | stat.S_IWUSR)
