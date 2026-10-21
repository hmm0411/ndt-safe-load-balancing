"""State-Action-Outcome record of one scripted migration attempt (Week 11).

This module is the contract between the runner (writes an attempt directory)
and the validator / dataset report (read it). Every view is built from the raw
orchestrator payloads the runner saves: ``GET /api/v1/twin/state`` and the
response of ``POST /api/v1/migrations``. Nothing here talks to the network.

Rules (Week 11 plan):
- controller-agnostic: always ``source_controller`` / ``target_controller``,
  never a hard-coded "c1"; works for 2C20S and 4C40S;
- ``transaction.status`` in COMMITTED, RESTORED, FAILED, SKIPPED;
  ``action_type`` in MIGRATE, NO_OP;
- no Week 12 prediction field (no predicted U, candidate or safety score).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "migration-outcome-v1"
ACTION_TYPES = ("MIGRATE", "NO_OP")
TRANSACTION_STATUSES = ("COMMITTED", "RESTORED", "FAILED", "SKIPPED")
# Statuses that come with a real orchestrator transaction (id + history).
_STATUSES_WITH_TRANSACTION = ("COMMITTED", "RESTORED")


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ValueError(f"{where}: missing field '{key}'")
    return data[key]


def _float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _optional_float(value: Any, name: str) -> float | None:
    return None if value is None else _float(value, name)


@dataclass(frozen=True)
class ControllerStateView:
    """One controller inside a TwinState (``twin_state["controllers"][i]``)."""

    controller_id: str
    utilization: float
    processed_packet_in_rate: float
    safe_capacity_pps: float
    response_p95_ms: float
    flow_mod_rate: float
    process_cpu_percent: float
    managed_switch_count: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ControllerStateView:
        where = "controller"
        cid = str(_require(data, "controller_id", where))
        where = f"controller {cid}"
        return cls(
            controller_id=cid,
            utilization=_float(_require(data, "utilization", where), f"{where}.utilization"),
            processed_packet_in_rate=_float(
                _require(data, "processed_packet_in_rate", where),
                f"{where}.processed_packet_in_rate",
            ),
            safe_capacity_pps=_float(
                _require(data, "safe_capacity_pps", where), f"{where}.safe_capacity_pps"
            ),
            response_p95_ms=_float(
                _require(data, "response_p95_ms", where), f"{where}.response_p95_ms"
            ),
            flow_mod_rate=_float(_require(data, "flow_mod_rate", where), f"{where}.flow_mod_rate"),
            process_cpu_percent=_float(
                _require(data, "process_cpu_percent", where), f"{where}.process_cpu_percent"
            ),
            managed_switch_count=int(_require(data, "managed_switch_count", where)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "utilization": self.utilization,
            "processed_packet_in_rate": self.processed_packet_in_rate,
            "safe_capacity_pps": self.safe_capacity_pps,
            "response_p95_ms": self.response_p95_ms,
            "flow_mod_rate": self.flow_mod_rate,
            "process_cpu_percent": self.process_cpu_percent,
            "managed_switch_count": self.managed_switch_count,
        }


@dataclass(frozen=True)
class SwitchStateView:
    """Load of one switch as seen by ONE controller.

    A TwinState can hold the same switch twice: every controller that ever
    received Packet-In from it keeps reporting it (``controller_id`` in the
    raw payload). The migrated switch's load must therefore be read from the
    entry of its owner at that time (source before, target after).
    ``processed_packet_in_rate`` is absolute (pps); ``control_load_share`` is
    that rate divided by the reporting controller's processed rate.
    """

    switch_id: str
    reporting_controller: str
    processed_packet_in_rate: float
    packet_in_rate: float
    flow_mod_rate: float
    control_load_share: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SwitchStateView:
        sid = str(_require(data, "switch_id", "switch"))
        where = f"switch {sid}"
        reporter = data.get("reporting_controller", data.get("controller_id"))
        if reporter is None:
            raise ValueError(f"{where}: missing field 'controller_id'")
        return cls(
            switch_id=sid,
            reporting_controller=str(reporter),
            processed_packet_in_rate=_float(
                _require(data, "processed_packet_in_rate", where),
                f"{where}.processed_packet_in_rate",
            ),
            packet_in_rate=_float(_require(data, "packet_in_rate", where), f"{where}.packet_in_rate"),
            flow_mod_rate=_float(_require(data, "flow_mod_rate", where), f"{where}.flow_mod_rate"),
            control_load_share=_float(
                _require(data, "control_load_share", where), f"{where}.control_load_share"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "switch_id": self.switch_id,
            "reporting_controller": self.reporting_controller,
            "processed_packet_in_rate": self.processed_packet_in_rate,
            "packet_in_rate": self.packet_in_rate,
            "flow_mod_rate": self.flow_mod_rate,
            "control_load_share": self.control_load_share,
        }


@dataclass(frozen=True)
class OwnershipView:
    switch_id: str
    owner_controller: str
    generation_id: int
    ownership_version: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OwnershipView:
        sid = str(_require(data, "switch_id", "ownership"))
        owner = data.get("owner_controller", data.get("owner_controller_id"))
        if owner is None:
            raise ValueError(f"ownership {sid}: missing field 'owner_controller_id'")
        return cls(
            switch_id=sid,
            owner_controller=str(owner),
            generation_id=int(_require(data, "generation_id", f"ownership {sid}")),
            ownership_version=int(_require(data, "ownership_version", f"ownership {sid}")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "switch_id": self.switch_id,
            "owner_controller": self.owner_controller,
            "generation_id": self.generation_id,
            "ownership_version": self.ownership_version,
        }


@dataclass(frozen=True)
class QualityView:
    """Hard Gate (``valid``, ``consistent``) and the soft scores of a TwinState."""

    valid: bool
    consistent: bool
    consistency_score: float
    quality_score: float | None
    completeness_ratio: float
    age_of_twin_ms: float
    synchronization_jitter_ms: float
    twinning_rate: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualityView:
        where = "quality"
        return cls(
            valid=bool(_require(data, "valid", where)),
            consistent=bool(_require(data, "consistent", where)),
            consistency_score=_float(
                _require(data, "consistency_score", where), "quality.consistency_score"
            ),
            quality_score=_optional_float(data.get("quality_score"), "quality.quality_score"),
            completeness_ratio=_float(
                _require(data, "completeness_ratio", where), "quality.completeness_ratio"
            ),
            age_of_twin_ms=_float(_require(data, "age_of_twin_ms", where), "quality.age_of_twin_ms"),
            synchronization_jitter_ms=_float(
                _require(data, "synchronization_jitter_ms", where),
                "quality.synchronization_jitter_ms",
            ),
            twinning_rate=_float(_require(data, "twinning_rate", where), "quality.twinning_rate"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "consistent": self.consistent,
            "consistency_score": self.consistency_score,
            "quality_score": self.quality_score,
            "completeness_ratio": self.completeness_ratio,
            "age_of_twin_ms": self.age_of_twin_ms,
            "synchronization_jitter_ms": self.synchronization_jitter_ms,
            "twinning_rate": self.twinning_rate,
        }


@dataclass(frozen=True)
class StateSnapshotView:
    """The parts of one TwinState the outcome needs (one line of *_samples.jsonl)."""

    snapshot_id: str
    created_at: str
    ownership_version: int
    controllers: tuple[ControllerStateView, ...]
    switches: tuple[SwitchStateView, ...]
    ownership: tuple[OwnershipView, ...]
    quality: QualityView

    @classmethod
    def from_twin_state(cls, payload: Mapping[str, Any]) -> StateSnapshotView:
        """Build from the raw ``GET /api/v1/twin/state`` payload (TwinState.to_dict)."""
        if payload.get("status") == "NOT_READY":
            raise ValueError("twin state is NOT_READY")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateSnapshotView:
        where = "twin_state"
        controllers = tuple(
            ControllerStateView.from_dict(x) for x in _require(data, "controllers", where)
        )
        if not controllers:
            raise ValueError("twin_state: no controllers")
        ids = [c.controller_id for c in controllers]
        if len(set(ids)) != len(ids):
            raise ValueError(f"twin_state: duplicate controller ids {ids}")
        return cls(
            snapshot_id=str(_require(data, "snapshot_id", where)),
            created_at=str(_require(data, "created_at", where)),
            ownership_version=int(_require(data, "ownership_version", where)),
            controllers=controllers,
            switches=tuple(SwitchStateView.from_dict(x) for x in _require(data, "switches", where)),
            ownership=tuple(OwnershipView.from_dict(x) for x in _require(data, "ownership", where)),
            quality=QualityView.from_dict(_require(data, "quality", where)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "ownership_version": self.ownership_version,
            "controllers": [c.to_dict() for c in self.controllers],
            "switches": [s.to_dict() for s in self.switches],
            "ownership": [o.to_dict() for o in self.ownership],
            "quality": self.quality.to_dict(),
        }

    def utilization(self) -> dict[str, float]:
        return {c.controller_id: c.utilization for c in self.controllers}

    def controller(self, controller_id: str) -> ControllerStateView:
        for item in self.controllers:
            if item.controller_id == controller_id:
                return item
        raise ValueError(f"snapshot {self.snapshot_id}: no controller '{controller_id}'")

    def owner_of(self, switch_id: str) -> str | None:
        for item in self.ownership:
            if item.switch_id == switch_id:
                return item.owner_controller
        return None

    def switch_load(self, switch_id: str, controller_id: str) -> SwitchStateView | None:
        """Entry of ``switch_id`` reported by ``controller_id`` (None if absent)."""
        for item in self.switches:
            if item.switch_id == switch_id and item.reporting_controller == controller_id:
                return item
        return None


@dataclass(frozen=True)
class MigrationAction:
    """A_t: what the runner asked for. For NO_OP the pair is the one that
    would have been migrated (same matrix cell), and nothing is sent."""

    action_type: str
    source_controller: str
    target_controller: str
    switch_id: str | None = None
    failure_injected: bool = False
    selection_rule: str = ""
    target_hot: bool = False

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}, got {self.action_type!r}")
        if not self.source_controller or not self.target_controller:
            raise ValueError("source_controller and target_controller are required")
        if self.source_controller == self.target_controller:
            raise ValueError("source_controller and target_controller must differ")
        if self.action_type == "MIGRATE" and not self.switch_id:
            raise ValueError("MIGRATE requires switch_id")
        if self.action_type == "NO_OP" and self.failure_injected:
            raise ValueError("NO_OP cannot inject a failure")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MigrationAction:
        where = "action"
        switch_id = data.get("switch_id")
        return cls(
            action_type=str(_require(data, "action_type", where)),
            source_controller=str(_require(data, "source_controller", where)),
            target_controller=str(_require(data, "target_controller", where)),
            switch_id=None if switch_id is None else str(switch_id),
            failure_injected=bool(data.get("failure_injected", False)),
            selection_rule=str(data.get("selection_rule", "")),
            target_hot=bool(data.get("target_hot", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "source_controller": self.source_controller,
            "target_controller": self.target_controller,
            "switch_id": self.switch_id,
            "failure_injected": self.failure_injected,
            "selection_rule": self.selection_rule,
            "target_hot": self.target_hot,
        }


@dataclass(frozen=True)
class TransactionEventView:
    state: str
    at: str
    detail: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransactionEventView:
        detail = data.get("detail")
        return cls(
            state=str(_require(data, "state", "transaction.history")),
            at=str(_require(data, "at", "transaction.history")),
            detail=None if detail is None else str(detail),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"state": self.state, "at": self.at, "detail": self.detail}


@dataclass(frozen=True)
class MigrationTransactionView:
    """Outcome of ``POST /api/v1/migrations`` for a MIGRATE action.

    COMMITTED / RESTORED: the endpoint answered with a transaction (id, history).
    FAILED: the endpoint returned HTTP 500 (rollback also failed); the
    transaction is kept when the error detail carries it.
    SKIPPED: a precondition failed, nothing was sent.
    """

    status: str
    transaction_id: str | None = None
    generation_id: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    failure_reason: str | None = None
    history: tuple[TransactionEventView, ...] = ()
    ownership_version_after: int | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in TRANSACTION_STATUSES:
            raise ValueError(
                f"transaction.status must be one of {TRANSACTION_STATUSES}, got {self.status!r}"
            )
        if self.status in _STATUSES_WITH_TRANSACTION:
            if not self.transaction_id:
                raise ValueError(f"{self.status} transaction needs transaction_id")
            if not self.history:
                raise ValueError(f"{self.status} transaction needs a history")
            if self.history[-1].state != self.status:
                raise ValueError(
                    f"{self.status} transaction ends in state {self.history[-1].state!r}"
                )
        if self.status == "SKIPPED" and not self.skip_reason:
            raise ValueError("SKIPPED transaction needs skip_reason")

    @classmethod
    def from_response(cls, payload: Mapping[str, Any]) -> MigrationTransactionView:
        """From a 200 response of ``POST /api/v1/migrations`` (COMMITTED or RESTORED)."""
        status = str(_require(payload, "status", "migration response"))
        tx = _require(payload, "transaction", "migration response")
        version = payload.get("ownership_version")
        return cls._from_transaction(
            status, tx, None if version is None else int(version), payload.get("failure_reason")
        )

    @classmethod
    def failed(cls, error_detail: Mapping[str, Any] | str) -> MigrationTransactionView:
        """From the HTTP 500 detail (migration and rollback both failed)."""
        if isinstance(error_detail, Mapping) and isinstance(
            error_detail.get("transaction"), Mapping
        ):
            reason = f"{error_detail.get('migration_error')}; {error_detail.get('rollback_error')}"
            view = cls._from_transaction("FAILED", error_detail["transaction"], None, reason)
            return view
        return cls(status="FAILED", failure_reason=str(error_detail))

    @classmethod
    def skipped(cls, reason: str) -> MigrationTransactionView:
        return cls(status="SKIPPED", skip_reason=reason)

    @classmethod
    def _from_transaction(
        cls,
        status: str,
        tx: Mapping[str, Any],
        ownership_version_after: int | None,
        failure_reason: Any,
    ) -> MigrationTransactionView:
        generation = tx.get("generation_id")
        reason = failure_reason if failure_reason is not None else tx.get("failure_reason")
        return cls(
            status=status,
            transaction_id=str(_require(tx, "transaction_id", "transaction")),
            generation_id=None if generation is None else int(generation),
            started_at=tx.get("started_at"),
            finished_at=tx.get("finished_at"),
            failure_reason=None if reason is None else str(reason),
            history=tuple(TransactionEventView.from_dict(x) for x in tx.get("history", [])),
            ownership_version_after=ownership_version_after,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MigrationTransactionView:
        generation = data.get("generation_id")
        version = data.get("ownership_version_after")
        return cls(
            status=str(_require(data, "status", "transaction")),
            transaction_id=data.get("transaction_id"),
            generation_id=None if generation is None else int(generation),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            failure_reason=data.get("failure_reason"),
            history=tuple(TransactionEventView.from_dict(x) for x in data.get("history", [])),
            ownership_version_after=None if version is None else int(version),
            skip_reason=data.get("skip_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "transaction_id": self.transaction_id,
            "generation_id": self.generation_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure_reason": self.failure_reason,
            "history": [e.to_dict() for e in self.history],
            "ownership_version_after": self.ownership_version_after,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True)
class AttemptMetadata:
    """Who/what/when of one attempt (metadata.json)."""

    attempt_id: str
    workload: str
    phase: str
    destination_load_level: float
    source_controller: str
    target_controller: str
    repeat: str
    seed: int
    topology: str
    git_commit: str
    config_sha256: str
    capacity_version: str
    quality_version: str
    started_at: str
    finished_at: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if not 0.0 < self.destination_load_level < 1.0:
            raise ValueError("destination_load_level must be in (0, 1)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttemptMetadata:
        where = "metadata"
        finished = data.get("finished_at")
        return cls(
            attempt_id=str(_require(data, "attempt_id", where)),
            workload=str(_require(data, "workload", where)),
            phase=str(_require(data, "phase", where)),
            destination_load_level=_float(
                _require(data, "destination_load_level", where), "metadata.destination_load_level"
            ),
            source_controller=str(_require(data, "source_controller", where)),
            target_controller=str(_require(data, "target_controller", where)),
            repeat=str(_require(data, "repeat", where)),
            seed=int(_require(data, "seed", where)),
            topology=str(_require(data, "topology", where)),
            git_commit=str(_require(data, "git_commit", where)),
            config_sha256=str(_require(data, "config_sha256", where)),
            capacity_version=str(_require(data, "capacity_version", where)),
            quality_version=str(_require(data, "quality_version", where)),
            started_at=str(_require(data, "started_at", where)),
            finished_at=None if finished is None else str(finished),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attempt_id": self.attempt_id,
            "workload": self.workload,
            "phase": self.phase,
            "destination_load_level": self.destination_load_level,
            "source_controller": self.source_controller,
            "target_controller": self.target_controller,
            "repeat": self.repeat,
            "seed": self.seed,
            "topology": self.topology,
            "git_commit": self.git_commit,
            "config_sha256": self.config_sha256,
            "capacity_version": self.capacity_version,
            "quality_version": self.quality_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass(frozen=True)
class QualitySummary:
    """Hard-Gate view of the two windows (record.quality)."""

    required_window: int
    before_count: int
    after_count: int
    before_all_valid: bool
    after_all_valid: bool
    post_window_complete: bool
    before_quality_min: float | None
    after_quality_min: float | None
    before_consistency_min: float | None
    after_consistency_min: float | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QualitySummary:
        where = "quality_summary"

        def opt(key: str) -> float | None:
            return _optional_float(data.get(key), f"{where}.{key}")

        return cls(
            required_window=int(_require(data, "required_window", where)),
            before_count=int(_require(data, "before_count", where)),
            after_count=int(_require(data, "after_count", where)),
            before_all_valid=bool(_require(data, "before_all_valid", where)),
            after_all_valid=bool(_require(data, "after_all_valid", where)),
            post_window_complete=bool(_require(data, "post_window_complete", where)),
            before_quality_min=opt("before_quality_min"),
            after_quality_min=opt("after_quality_min"),
            before_consistency_min=opt("before_consistency_min"),
            after_consistency_min=opt("after_consistency_min"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_window": self.required_window,
            "before_count": self.before_count,
            "after_count": self.after_count,
            "before_all_valid": self.before_all_valid,
            "after_all_valid": self.after_all_valid,
            "post_window_complete": self.post_window_complete,
            "before_quality_min": self.before_quality_min,
            "after_quality_min": self.after_quality_min,
            "before_consistency_min": self.before_consistency_min,
            "after_consistency_min": self.after_consistency_min,
        }


# Fields of PostMigrationOutcome that may be None (empty after-window or NO_OP).
_OUTCOME_FLOATS = (
    "src_u_before_last", "src_u_before_mean", "src_u_after_mean",
    "delta_src_u_mean", "delta_src_u_last",
    "dst_u_before_last", "dst_u_before_mean", "dst_u_after_mean",
    "delta_dst_u_mean", "delta_dst_u_last",
    "imbalance_before_last", "imbalance_before_mean", "imbalance_after_mean",
    "delta_imbalance_mean", "max_u_before_mean", "max_u_after_mean",
    "dst_response_p95_before_mean", "dst_response_p95_after_mean",
    "switch_rate_before_mean", "switch_share_before_mean",
)


@dataclass(frozen=True)
class PostMigrationOutcome:
    """Observed effect of the action (outcome.json), from metrics.compute_outcome.

    Both aggregations are kept so Week 12 can choose without re-collecting:
    ``*_before_last`` (last pre-window sample) and ``*_before_mean`` (mean of
    the pre-window). ``delta_*_mean`` = after_mean - before_mean,
    ``delta_*_last`` = after_mean - before_last. Deltas of RESTORED / FAILED
    attempts are what was observed, NOT the effect of a successful migration;
    those attempts carry ``training_eligible = False``.
    """

    action_type: str
    transaction_status: str | None
    training_eligible: bool
    ineligible_reasons: tuple[str, ...] = ()
    src_u_before_last: float | None = None
    src_u_before_mean: float | None = None
    src_u_after_mean: float | None = None
    delta_src_u_mean: float | None = None
    delta_src_u_last: float | None = None
    dst_u_before_last: float | None = None
    dst_u_before_mean: float | None = None
    dst_u_after_mean: float | None = None
    delta_dst_u_mean: float | None = None
    delta_dst_u_last: float | None = None
    imbalance_before_last: float | None = None
    imbalance_before_mean: float | None = None
    imbalance_after_mean: float | None = None
    delta_imbalance_mean: float | None = None
    max_u_before_mean: float | None = None
    max_u_after_mean: float | None = None
    dst_response_p95_before_mean: float | None = None
    dst_response_p95_after_mean: float | None = None
    switch_rate_before_mean: float | None = None
    switch_share_before_mean: float | None = None

    def __post_init__(self) -> None:
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"action_type must be one of {ACTION_TYPES}")
        if self.transaction_status is not None and (
            self.transaction_status not in TRANSACTION_STATUSES
        ):
            raise ValueError(f"transaction_status must be one of {TRANSACTION_STATUSES}")
        if self.training_eligible and self.ineligible_reasons:
            raise ValueError("an eligible outcome cannot carry ineligible_reasons")
        if not self.training_eligible and not self.ineligible_reasons:
            raise ValueError("an ineligible outcome needs at least one reason")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PostMigrationOutcome:
        where = "outcome"
        values = {name: _optional_float(data.get(name), f"{where}.{name}") for name in _OUTCOME_FLOATS}
        status = data.get("transaction_status")
        return cls(
            action_type=str(_require(data, "action_type", where)),
            transaction_status=None if status is None else str(status),
            training_eligible=bool(_require(data, "training_eligible", where)),
            ineligible_reasons=tuple(str(x) for x in data.get("ineligible_reasons", [])),
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action_type": self.action_type,
            "transaction_status": self.transaction_status,
            "training_eligible": self.training_eligible,
            "ineligible_reasons": list(self.ineligible_reasons),
        }
        for name in _OUTCOME_FLOATS:
            data[name] = getattr(self, name)
        return data


@dataclass(frozen=True)
class MigrationOutcomeRecord:
    """One (S_t, A_t, S_t+Delta) sample = one attempt directory, assembled."""

    metadata: AttemptMetadata
    before_state: tuple[StateSnapshotView, ...]
    action: MigrationAction
    transaction: MigrationTransactionView | None
    after_state: tuple[StateSnapshotView, ...]
    outcome: PostMigrationOutcome
    quality: QualitySummary
    schema_version: str = SCHEMA_VERSION
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        if self.action.action_type == "MIGRATE" and self.transaction is None:
            raise ValueError("a MIGRATE record needs a transaction (use SKIPPED if none ran)")
        if self.action.action_type == "NO_OP" and self.transaction is not None:
            raise ValueError("a NO_OP record has no transaction")
        if (self.action.source_controller, self.action.target_controller) != (
            self.metadata.source_controller,
            self.metadata.target_controller,
        ):
            raise ValueError("action and metadata disagree on source/target controller")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MigrationOutcomeRecord:
        where = "record"
        tx = data.get("transaction")
        return cls(
            schema_version=str(_require(data, "schema_version", where)),
            metadata=AttemptMetadata.from_dict(_require(data, "metadata", where)),
            before_state=_snapshots(_require(data, "before_state", where)),
            action=MigrationAction.from_dict(_require(data, "action", where)),
            transaction=None if tx is None else MigrationTransactionView.from_dict(tx),
            after_state=_snapshots(_require(data, "after_state", where)),
            outcome=PostMigrationOutcome.from_dict(_require(data, "outcome", where)),
            quality=QualitySummary.from_dict(_require(data, "quality", where)),
            notes=tuple(str(x) for x in data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "metadata": self.metadata.to_dict(),
            "before_state": [s.to_dict() for s in self.before_state],
            "action": self.action.to_dict(),
            "transaction": None if self.transaction is None else self.transaction.to_dict(),
            "after_state": [s.to_dict() for s in self.after_state],
            "outcome": self.outcome.to_dict(),
            "quality": self.quality.to_dict(),
            "notes": list(self.notes),
        }


def _snapshots(items: Sequence[Mapping[str, Any]]) -> tuple[StateSnapshotView, ...]:
    return tuple(StateSnapshotView.from_dict(x) for x in items)
