"""Validate migration-outcome attempt directories (Week 11).

``validate_attempt(dir)`` checks one attempt written by ``writer.write_attempt``;
``validate_dataset(root)`` checks every attempt under a root plus cross-attempt
rules (duplicate transactions, coverage of the experiment matrix).

Two outcomes, never mixed up:
- ``valid = False``: the directory is broken (missing file, bad field, wrong
  ownership, outcome that does not match the raw samples...). Usually a bug:
  fix the code and re-collect.
- ``valid = True, training_eligible = False``: a correct record of something
  that did not give a usable migration sample (RESTORED, FAILED, SKIPPED,
  post window timed out...). It is data and stays in the dataset.
The validator never repairs or fills in anything.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from src.experiments.migration.config import OutcomeConfig, expand_attempts
from src.experiments.migration.metrics import DEFAULT_WINDOW, compute_outcome
from src.experiments.migration.schema import (
    SCHEMA_VERSION,
    AttemptMetadata,
    MigrationAction,
    PostMigrationOutcome,
    StateSnapshotView,
)
from src.experiments.migration.writer import (
    CHECKSUM_FILE,
    DATA_FILES,
    INVALID_DIR,
    read_checksums,
    sha256_file,
    transaction_from_raw,
)

TOLERANCE = 1e-9

EXECUTED_STATUSES = frozenset({"COMMITTED", "RESTORED", "FAILED"})


@dataclass
class _Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
    return rows


def _snapshots(
    rows: list[dict[str, Any]], name: str, report: _Report
) -> list[StateSnapshotView]:
    views = []
    for index, row in enumerate(rows):
        try:
            views.append(StateSnapshotView.from_twin_state(row))
        except ValueError as exc:
            report.error(f"{name}[{index}]: {exc}")
    return views


def _check_window_order(views: list[StateSnapshotView], name: str, report: _Report) -> None:
    ids = [v.snapshot_id for v in views]
    if len(set(ids)) != len(ids):
        report.error(f"{name}: duplicate snapshot_id (samples must not be repeated)")
    times = [v.created_at for v in views]
    if any(b <= a for a, b in zip(times, times[1:])):
        report.error(f"{name}: created_at is not strictly increasing")


def _check_snapshot_fields(views: list[StateSnapshotView], name: str, report: _Report) -> None:
    for index, view in enumerate(views):
        quality = view.quality
        if quality.quality_score is None:
            report.error(f"{name}[{index}]: quality_score missing")
        elif not 0.0 <= quality.quality_score <= 1.0:
            report.error(f"{name}[{index}]: quality_score out of [0, 1]")
        if not 0.0 <= quality.consistency_score <= 1.0:
            report.error(f"{name}[{index}]: consistency_score out of [0, 1]")
        for controller in view.controllers:
            if controller.safe_capacity_pps <= 0:
                report.error(
                    f"{name}[{index}]: safe capacity missing for {controller.controller_id}"
                )


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=0.0, abs_tol=TOLERANCE)


def _compare_outcome(stored: PostMigrationOutcome, fresh: PostMigrationOutcome,
                     report: _Report) -> None:
    left, right = stored.to_dict(), fresh.to_dict()
    for key, value in right.items():
        other = left.get(key)
        if isinstance(value, float) or isinstance(other, float):
            if not _same(other, value):
                report.error(f"outcome.{key}: stored {other} != recomputed {value}")
        elif other != value:
            report.error(f"outcome.{key}: stored {other!r} != recomputed {value!r}")


def validate_attempt(
    attempt_dir: Path,
    required_window: int = DEFAULT_WINDOW,
    check_checksums: bool = True,
) -> dict[str, Any]:
    """Check one attempt directory; returns the content of validation.json."""
    report = _Report()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_dir.name.removeprefix(".tmp-"),
        "action_type": None,
        "transaction_status": None,
        "transaction_id": None,
    }

    def finish(eligible: bool = False) -> dict[str, Any]:
        result["valid"] = not report.errors
        result["training_eligible"] = bool(eligible and not report.errors)
        result["errors"] = report.errors
        result["warnings"] = report.warnings
        return result

    missing = [name for name in DATA_FILES if not (attempt_dir / name).is_file()]
    if missing:
        report.error(f"missing files: {missing}")
        return finish()

    try:
        metadata = AttemptMetadata.from_dict(_read_json(attempt_dir / "metadata.json"))
        action = MigrationAction.from_dict(_read_json(attempt_dir / "action.json"))
        transaction_raw = _read_json(attempt_dir / "transaction.json")
        transaction = transaction_from_raw(transaction_raw)
        stored = _read_json(attempt_dir / "outcome.json")
        stored_outcome = PostMigrationOutcome.from_dict(stored["outcome"])
        before_rows = _read_jsonl(attempt_dir / "before_samples.jsonl")
        after_rows = _read_jsonl(attempt_dir / "after_samples.jsonl")
        _read_jsonl(attempt_dir / "transition_samples.jsonl")
        _read_jsonl(attempt_dir / "forecast_samples.jsonl")
    except (ValueError, KeyError, TypeError) as exc:
        report.error(f"unreadable attempt: {exc}")
        return finish()

    result["attempt_id"] = metadata.attempt_id
    result["action_type"] = action.action_type
    result["transaction_status"] = None if transaction is None else transaction.status
    result["transaction_id"] = None if transaction is None else transaction.transaction_id
    if not attempt_dir.name.startswith(".tmp-") and attempt_dir.name != metadata.attempt_id:
        report.error(f"directory name {attempt_dir.name} != attempt_id {metadata.attempt_id}")
    if stored.get("schema_version") != SCHEMA_VERSION:
        report.error("outcome.json: wrong schema_version")
    if (action.source_controller, action.target_controller) != (
        metadata.source_controller, metadata.target_controller
    ):
        report.error("action and metadata disagree on source/target")
    if action.action_type == "MIGRATE" and transaction is None:
        report.error("MIGRATE attempt without a transaction (record SKIPPED instead)")
    if action.action_type == "NO_OP" and transaction is not None:
        report.error("NO_OP attempt has a transaction")

    before = _snapshots(before_rows, "before", report)
    after = _snapshots(after_rows, "after", report)
    if report.errors:
        return finish()

    # Pre window: the runner must not act without a full, valid pre window.
    # Only an executed MIGRATE (COMMITTED / RESTORED / FAILED) acted on the testbed; a
    # SKIPPED MIGRATE or a NO_OP keeps whatever pre window it had and is simply not
    # eligible (compute_outcome already says why), so these are warnings there.
    status = None if transaction is None else transaction.status
    executed = action.action_type == "MIGRATE" and status in EXECUTED_STATUSES
    pre_problem = report.error if executed else report.warn
    if len(before) != required_window:
        pre_problem(f"before window has {len(before)} samples, expected {required_window}")
    if not all(v.quality.valid for v in before):
        pre_problem("before window contains a snapshot that failed the Hard Gate")
    _check_window_order(before, "before", report)
    _check_window_order(after, "after", report)
    if before and after and after[0].created_at <= before[-1].created_at:
        report.error("after window starts before the end of the before window")
    _check_snapshot_fields(before, "before", report)
    _check_snapshot_fields(after, "after", report)
    for view in before + after:
        ids = {c.controller_id for c in view.controllers}
        if not {action.source_controller, action.target_controller} <= ids:
            report.error(f"snapshot {view.snapshot_id}: source/target controller missing")
            return finish()

    # Ownership and roles around the transaction.
    if executed and action.switch_id is not None:
        sid = action.switch_id
        if any(v.owner_of(sid) != action.source_controller for v in before):
            report.error(f"before window: {sid} is not owned by the source")
        expected_after = {
            "COMMITTED": action.target_controller,
            "RESTORED": action.source_controller,
        }.get(str(status))
        if expected_after is not None:
            if not after:
                report.warn("no after samples to check ownership")
            elif any(v.owner_of(sid) != expected_after for v in after):
                report.error(f"{status}: {sid} should be owned by {expected_after} after")
        if status == "COMMITTED" and transaction is not None and before:
            version = transaction.ownership_version_after
            if version is not None and version <= before[-1].ownership_version:
                report.error("COMMITTED but ownership_version did not increase")
        if status == "FAILED":
            report.warn("FAILED transaction: ownership after rollback is not checked")

    if len(after) < required_window:
        report.warn(f"after window has {len(after)} of {required_window} samples")
    elif not all(v.quality.valid for v in after):
        report.warn("after window contains a snapshot that failed the Hard Gate")

    # Outcome must be exactly what the raw samples give (no edited numbers).
    try:
        fresh = compute_outcome(before, after, action, transaction, required_window)
    except ValueError as exc:
        report.error(f"outcome cannot be recomputed: {exc}")
        return finish()
    _compare_outcome(stored_outcome, fresh, report)

    if check_checksums:
        if not (attempt_dir / CHECKSUM_FILE).is_file():
            report.error(f"missing {CHECKSUM_FILE}")
        else:
            for name, digest in read_checksums(attempt_dir).items():
                path = attempt_dir / name
                if not path.is_file() or sha256_file(path) != digest:
                    report.error(f"checksum mismatch: {name}")
    return finish(fresh.training_eligible)


def _attempt_dirs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def validate_dataset(
    root: Path,
    config: OutcomeConfig | None = None,
    required_window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Validate every attempt under ``root`` and summarize the dataset.

    With ``config``, also reports which planned attempts are missing and which
    directories are not in the plan.
    """
    attempts: dict[str, dict[str, Any]] = {}
    for path in _attempt_dirs(root):
        try:
            attempts[path.name] = validate_attempt(path, required_window)
        except Exception as exc:  # a broken attempt must not stop the scan
            attempts[path.name] = {"attempt_id": path.name, "valid": False,
                                   "training_eligible": False,
                                   "errors": [f"validator crashed: {exc!r}"], "warnings": []}

    tx_owner: dict[str, list[str]] = {}
    for name, report in attempts.items():
        tx_id = report.get("transaction_id")
        if tx_id:
            tx_owner.setdefault(str(tx_id), []).append(name)
    duplicates = {tx: names for tx, names in tx_owner.items() if len(names) > 1}
    for names in duplicates.values():
        for name in names:
            attempts[name]["valid"] = False
            attempts[name]["training_eligible"] = False
            attempts[name]["errors"] = [*attempts[name]["errors"], "duplicate transaction_id"]

    counts: dict[str, Counter[str]] = {
        "workload": Counter(), "phase": Counter(), "direction": Counter(),
        "destination_level": Counter(), "action_type": Counter(),
        "transaction_status": Counter(),
    }
    for name, report in attempts.items():
        parts = name.split("-")
        if len(parts) == 7:
            counts["workload"][parts[1]] += 1
            counts["phase"][f"{parts[1]}:{parts[2]}"] += 1
            counts["destination_level"][parts[3]] += 1
            counts["direction"][f"{parts[4]}->{parts[5]}"] += 1
        counts["action_type"][str(report.get("action_type"))] += 1
        counts["transaction_status"][str(report.get("transaction_status"))] += 1

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "attempts": len(attempts),
        "valid": sum(1 for r in attempts.values() if r["valid"]),
        "invalid": sum(1 for r in attempts.values() if not r["valid"]),
        "training_eligible": sum(1 for r in attempts.values() if r["training_eligible"]),
        "duplicate_transactions": duplicates,
        "counts": {key: dict(sorted(value.items())) for key, value in counts.items()},
        "invalid_attempts": {
            name: r["errors"] for name, r in attempts.items() if not r["valid"]
        },
        "quarantined": (
            sorted(p.name for p in (root / INVALID_DIR).iterdir())
            if (root / INVALID_DIR).is_dir() else []
        ),
    }
    if config is not None:
        planned = {plan.attempt_id for plan in expand_attempts(config)}
        summary["planned"] = len(planned)
        summary["missing"] = sorted(planned - set(attempts))
        summary["unplanned"] = sorted(set(attempts) - planned)
    return summary


def load_validation(attempt_dir: Path) -> Mapping[str, Any]:
    return _read_json(attempt_dir / "validation.json")
