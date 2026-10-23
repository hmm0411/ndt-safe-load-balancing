"""Pilot report of the migration-outcome collection (Week 11, Day 3).

The pilot exists to find bugs before the official 340 attempts, not to collect
data. This module reads a pilot root written by the runner, runs the validator
and adds the Day 3 checks of the plan that are NOT validator rules:

* bugs (block the freeze): validator invalid, transaction not terminal, post
  window taken before the transaction ended, controller roles disagreeing with
  the orchestrator ownership (roles not reset / not updated), switch load lost
  after the owner change, duplicate transaction, initial ownership different
  between attempts, collector versions (commit, config, capacity, quality)
  changing inside the pilot;
* things to look at by hand: SKIPPED / unintended RESTORED or FAILED,
  destination U far from the planned level, hot switch share far from its
  level (MIGRATE hot_switch runs skip the workload's own load check), total
  load dropping (workload stopped?), Delta-U with the wrong sign or a magnitude
  far from the switch load.

It also measures what Day 3 must decide: minutes per attempt, time from the end
of the transaction to the 5th valid post snapshot (-> post_window_timeout_s),
and the estimated time of the official collection.

Pure reading: never writes into the pilot root.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import yaml

from src.experiments.migration.config import OutcomeConfig, expand_attempts
from src.experiments.migration.metrics import DEFAULT_WINDOW
from src.experiments.migration.schema import (
    AttemptMetadata,
    MigrationAction,
    MigrationTransactionView,
    PostMigrationOutcome,
    StateSnapshotView,
)
from src.experiments.migration.validator import (
    load_validation,
    validate_attempt,
    validate_dataset,
)
from src.experiments.migration.writer import VALIDATION_FILE, transaction_from_raw
from src.experiments.workloads.hot_switch import HOT_SWITCH_LEVELS

EXECUTED = ("COMMITTED", "RESTORED", "FAILED")
FREEZE_KEYS = ("git_commit", "config_sha256", "capacity_version", "quality_version", "topology")


@dataclass(frozen=True)
class PilotSettings:
    pilot_root: str
    min_attempts: int
    collection_budget_hours: float
    budget_utilisation: float
    cleanup_overhead_s: float
    destination_level_tolerance: float
    hot_switch_share_tolerance: float
    workload_drop_ratio: float
    delta_ratio_range: tuple[float, float]
    timeout_headroom: float


def load_pilot_settings(path: Path) -> PilotSettings:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return validate_pilot_settings(data)


def validate_pilot_settings(data: Mapping[str, Any]) -> PilotSettings:
    def need(key: str) -> Any:
        if key not in data:
            raise ValueError(f"migration_pilot config: missing '{key}'")
        return data[key]

    low, high = (float(x) for x in need("delta_ratio_range"))
    settings = PilotSettings(
        pilot_root=str(need("pilot_root")),
        min_attempts=int(need("min_attempts")),
        collection_budget_hours=float(need("collection_budget_hours")),
        budget_utilisation=float(need("budget_utilisation")),
        cleanup_overhead_s=float(need("cleanup_overhead_s")),
        destination_level_tolerance=float(need("destination_level_tolerance")),
        hot_switch_share_tolerance=float(need("hot_switch_share_tolerance")),
        workload_drop_ratio=float(need("workload_drop_ratio")),
        delta_ratio_range=(low, high),
        timeout_headroom=float(need("timeout_headroom")),
    )
    if settings.min_attempts < 1:
        raise ValueError("min_attempts must be >= 1")
    if settings.collection_budget_hours <= 0:
        raise ValueError("collection_budget_hours must be > 0")
    if not 0 < settings.budget_utilisation <= 1:
        raise ValueError("budget_utilisation must be in (0, 1]")
    if settings.cleanup_overhead_s < 0 or settings.destination_level_tolerance <= 0:
        raise ValueError("cleanup_overhead_s must be >= 0 and the level tolerance > 0")
    if settings.hot_switch_share_tolerance <= 0:
        raise ValueError("hot_switch_share_tolerance must be > 0")
    if not 0 < settings.workload_drop_ratio < 1:
        raise ValueError("workload_drop_ratio must be in (0, 1)")
    if not 0 < low < 1 < high:
        raise ValueError("delta_ratio_range must be [low < 1 < high], low > 0")
    if settings.timeout_headroom < 1:
        raise ValueError("timeout_headroom must be >= 1")
    return settings


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None


def seconds_between(start: str | None, end: str | None) -> float | None:
    a, b = parse_time(start), parse_time(end)
    if a is None or b is None:
        return None
    try:
        return (b - a).total_seconds()
    except TypeError:  # naive vs aware timestamps
        return None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[rank - 1]


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "median": median(values) if values else None,
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_views(path: Path) -> tuple[list[StateSnapshotView], list[dict[str, Any]]]:
    views: list[StateSnapshotView] = []
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return views, rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        try:
            views.append(StateSnapshotView.from_twin_state(row))
        except (ValueError, KeyError, TypeError):
            continue
    return views, rows


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def role_mismatches(view: StateSnapshotView) -> list[str]:
    """Controllers whose MASTER count (controller telemetry) differs from the
    number of switches the orchestrator says they own."""
    owned = Counter(o.owner_controller for o in view.ownership)
    return [
        f"{c.controller_id}: managed {c.managed_switch_count} != owned {owned.get(c.controller_id, 0)}"
        for c in view.controllers
        if c.managed_switch_count != owned.get(c.controller_id, 0)
    ]


def ownership_map(view: StateSnapshotView) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((o.switch_id, o.owner_controller) for o in view.ownership))


# --------------------------------------------------------------------------
# one attempt
# --------------------------------------------------------------------------
def analyze_attempt(
    attempt_dir: Path,
    settings: PilotSettings,
    required_window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Measurements and pilot flags of one attempt directory."""
    validation = (
        load_validation(attempt_dir)
        if (attempt_dir / VALIDATION_FILE).is_file()
        else validate_attempt(attempt_dir, required_window)
    )
    row: dict[str, Any] = {
        "attempt_id": attempt_dir.name,
        "valid": bool(validation.get("valid")),
        "training_eligible": bool(validation.get("training_eligible")),
        "errors": list(validation.get("errors", [])),
        "warnings": list(validation.get("warnings", [])),
        "bugs": [],
        "review": [],
    }
    bugs: list[str] = row["bugs"]
    review: list[str] = row["review"]
    try:
        metadata = AttemptMetadata.from_dict(_read_json(attempt_dir / "metadata.json"))
        action = MigrationAction.from_dict(_read_json(attempt_dir / "action.json"))
        tx = transaction_from_raw(_read_json(attempt_dir / "transaction.json"))
        outcome = PostMigrationOutcome.from_dict(_read_json(attempt_dir / "outcome.json")["outcome"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        bugs.append(f"unreadable attempt: {exc}")
        return row
    before, _ = _read_views(attempt_dir / "before_samples.jsonl")
    after, _ = _read_views(attempt_dir / "after_samples.jsonl")
    _, transition_rows = _read_views(attempt_dir / "transition_samples.jsonl")

    status = None if tx is None else tx.status
    src, dst = action.source_controller, action.target_controller
    row.update({
        "workload": metadata.workload,
        "phase": metadata.phase,
        "destination_load_level": metadata.destination_load_level,
        "direction": f"{src}->{dst}",
        "repeat": metadata.repeat,
        "action_type": action.action_type,
        "switch_id": action.switch_id,
        "failure_injected": action.failure_injected,
        "target_hot": action.target_hot,
        "transaction_status": status,
        "started_at": metadata.started_at,
        "freeze": {key: getattr(metadata, key) for key in FREEZE_KEYS},
        "initial_ownership": [list(p) for p in ownership_map(before[0])] if before else None,
    })

    # --- time -------------------------------------------------------------
    row["duration_s"] = seconds_between(metadata.started_at, metadata.finished_at)
    row["transaction_s"] = None if tx is None else seconds_between(tx.started_at, tx.finished_at)
    action_end = tx.finished_at if (tx is not None and status in EXECUTED) else (
        before[-1].created_at if before else None)
    complete = len(after) >= required_window
    row["post_window_complete"] = complete
    row["first_post_gap_s"] = seconds_between(action_end, after[0].created_at) if after else None
    row["settle_s"] = seconds_between(action_end, after[-1].created_at) if complete else None
    row["transition_samples"] = len(transition_rows)
    row["transition_hard_gate_blocked"] = sum(
        1 for r in transition_rows
        if not (isinstance(r.get("quality"), Mapping) and r["quality"].get("valid") is True)
    )

    # --- load levels and Delta-U -------------------------------------------
    row["src_u_before_mean"] = outcome.src_u_before_mean
    row["dst_u_before_mean"] = outcome.dst_u_before_mean
    row["delta_src_u_mean"] = outcome.delta_src_u_mean
    row["delta_dst_u_mean"] = outcome.delta_dst_u_mean
    row["switch_share_before_mean"] = outcome.switch_share_before_mean
    row["switch_rate_before_mean"] = outcome.switch_rate_before_mean
    if outcome.dst_u_before_mean is not None:
        error = outcome.dst_u_before_mean - metadata.destination_load_level
        row["destination_level_error"] = error
        if abs(error) > settings.destination_level_tolerance:
            review.append(
                f"destination U {outcome.dst_u_before_mean:.3f} vs planned "
                f"{metadata.destination_load_level:.2f}")
    planned_share = HOT_SWITCH_LEVELS.get(metadata.phase)
    share = outcome.switch_share_before_mean
    if (metadata.workload == "hot_switch" and action.target_hot and planned_share is not None
            and share is not None):
        row["hot_switch_share_error"] = share - planned_share
        if abs(share - planned_share) > settings.hot_switch_share_tolerance:
            review.append(f"hot switch share {share:.3f} vs level {planned_share:.2f}")
    if before and after:
        rate_before = _mean([sum(c.processed_packet_in_rate for c in v.controllers) for v in before])
        rate_after = _mean([sum(c.processed_packet_in_rate for c in v.controllers) for v in after])
        if rate_before and rate_after is not None:
            row["total_rate_ratio"] = rate_after / rate_before
            if row["total_rate_ratio"] < settings.workload_drop_ratio:
                review.append(f"total load fell to {row['total_rate_ratio']:.2f}x: workload stopped?")

    if status == "COMMITTED" and before and outcome.switch_rate_before_mean is not None:
        expected_src = -outcome.switch_rate_before_mean / before[-1].controller(src).safe_capacity_pps
        expected_dst = outcome.switch_rate_before_mean / before[-1].controller(dst).safe_capacity_pps
        row["expected_delta_src"] = expected_src
        row["expected_delta_dst"] = expected_dst
        low, high = settings.delta_ratio_range
        for side, observed, expected in (
            ("src", outcome.delta_src_u_mean, expected_src),
            ("dst", outcome.delta_dst_u_mean, expected_dst),
        ):
            if observed is None or expected == 0:
                continue
            ratio = observed / expected
            row[f"delta_ratio_{side}"] = ratio
            if ratio <= 0:
                review.append(f"Delta-U {side} has the wrong sign ({observed:+.3f})")
            elif not low <= ratio <= high:
                review.append(f"Delta-U {side} is {ratio:.2f}x the switch load")

    # --- pilot bugs (the Day 3 list of the plan) ----------------------------
    if not row["valid"]:
        message = f"validator invalid: {row['errors']}"
        (review if action.failure_injected else bugs).append(message)
    if tx is not None and status in EXECUTED:
        bugs.extend(_transaction_problems(tx))
    if after and tx is not None and status in EXECUTED and tx.finished_at:
        gap = seconds_between(tx.finished_at, after[0].created_at)
        if gap is not None and gap <= 0:
            bugs.append("post window starts before the transaction ended")
    for view in before:
        problems = role_mismatches(view)
        if problems:
            bugs.append(f"before window: roles disagree with ownership ({'; '.join(problems)})")
            break
    for view in after:
        problems = role_mismatches(view)
        if problems:
            bugs.append(f"after window: roles not updated ({'; '.join(problems)})")
            break
    if status == "COMMITTED" and action.switch_id is not None:
        for view in after:
            entry = view.switch_load(action.switch_id, dst)
            if entry is None:
                bugs.append(f"{action.switch_id} load missing from {dst} after the move")
                break
            if entry.processed_packet_in_rate <= 0 and (outcome.switch_rate_before_mean or 0) > 0:
                review.append(f"{action.switch_id} reports zero load on {dst} after the move")
                break

    # --- outcomes to look at -------------------------------------------------
    if status == "SKIPPED":
        review.append(f"SKIPPED: {None if tx is None else tx.skip_reason}")
    elif status in ("RESTORED", "FAILED") and not action.failure_injected:
        review.append(f"{status} without failure injection")
    if not complete and status != "SKIPPED":
        review.append(f"post window incomplete ({len(after)}/{required_window})")
    return row


def _transaction_problems(tx: MigrationTransactionView) -> list[str]:
    problems: list[str] = []
    if not tx.finished_at:
        problems.append(f"{tx.status} transaction without finished_at (not terminal?)")
    if tx.history and tx.history[-1].state != tx.status:
        problems.append(f"history ends in {tx.history[-1].state}, status is {tx.status}")
    if not tx.history and tx.status in ("COMMITTED", "RESTORED"):
        problems.append("transaction history is empty")
    duration = seconds_between(tx.started_at, tx.finished_at)
    if duration is not None and duration < 0:
        problems.append("transaction finished before it started")
    return problems


# --------------------------------------------------------------------------
# the whole pilot
# --------------------------------------------------------------------------
def _attempt_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def _cycle_times(rows: list[dict[str, Any]], overhead_s: float) -> None:
    """Wall time per attempt including cleanup: the start of the next attempt when
    it follows right after (same session), else duration + overhead."""
    timed: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        started = parse_time(row.get("started_at"))
        if started is not None:
            timed.append((started, row))
    timed.sort(key=lambda item: item[0])
    for index, (started, row) in enumerate(timed):
        duration = row.get("duration_s")
        cycle = None if duration is None else duration + overhead_s
        if index + 1 < len(timed) and duration is not None:
            gap = (timed[index + 1][0] - started).total_seconds()
            if duration <= gap <= duration + 10 * max(overhead_s, 1.0):
                cycle = gap
        row["cycle_s"] = cycle


def build_pilot_report(
    root: Path,
    config: OutcomeConfig,
    settings: PilotSettings,
) -> dict[str, Any]:
    """Everything Day 3 needs to decide the freeze, as one JSON-able dict."""
    window = config.post_window_valid_snapshots
    dataset = validate_dataset(root, config, window)
    rows = [analyze_attempt(d, settings, window) for d in _attempt_dirs(root)]
    _cycle_times(rows, settings.cleanup_overhead_s)

    dataset_bugs: list[str] = []
    for tx_id, attempts in dataset["duplicate_transactions"].items():
        dataset_bugs.append(f"duplicate transaction {tx_id}: {attempts}")
    freeze: dict[str, list[str]] = {}
    for key in FREEZE_KEYS:
        values = sorted({str(r["freeze"][key]) for r in rows if "freeze" in r})
        freeze[key] = values
        if len(values) > 1:
            dataset_bugs.append(f"{key} changed inside the pilot: {values}")
    initial = Counter(
        json.dumps(r["initial_ownership"]) for r in rows if r.get("initial_ownership")
    )
    if len(initial) > 1:
        baseline = initial.most_common(1)[0][0]
        for r in rows:
            if r.get("initial_ownership") and json.dumps(r["initial_ownership"]) != baseline:
                r["bugs"].append("initial ownership differs from the other attempts (not reset)")

    # coverage of the matrix the plan asks the pilot to touch
    migrate = [r for r in rows if r.get("action_type") == "MIGRATE"]
    seen: dict[str, list[Any]] = {
        "workload": sorted({r["workload"] for r in migrate}),
        "direction": sorted({r["direction"] for r in migrate}),
        "destination_load_level": sorted({r["destination_load_level"] for r in migrate}),
    }
    wanted: dict[str, list[Any]] = {
        "workload": sorted(config.workloads),
        "direction": sorted(f"{s}->{d}" for s, d in config.directions),
        "destination_load_level": sorted(config.destination_load_levels),
    }
    missing = {k: [v for v in wanted[k] if v not in seen[k]] for k in wanted}
    no_op = sum(1 for r in rows if r.get("action_type") == "NO_OP")
    coverage_ok = not any(missing.values()) and no_op > 0

    # time
    durations = [r["duration_s"] for r in rows if r.get("duration_s") is not None]
    cycles = [r["cycle_s"] for r in rows if r.get("cycle_s") is not None]
    settles = [r["settle_s"] for r in rows if r.get("settle_s") is not None]
    blocked = [r["transition_hard_gate_blocked"] for r in rows if "transition_hard_gate_blocked" in r]
    planned = Counter(plan.workload for plan in expand_attempts(config))
    per_workload: dict[str, Any] = {}
    estimate_s = 0.0
    p95_estimate_s = 0.0
    extrapolated: list[str] = []
    overall_median = median(cycles) if cycles else None
    overall_p95 = percentile(cycles, 95)
    for workload, count in sorted(planned.items()):
        own: list[float] = [float(r["cycle_s"]) for r in rows
                            if r.get("workload") == workload and r.get("cycle_s") is not None]
        med = median(own) if own else overall_median
        p95 = percentile(own, 95) if own else overall_p95
        if not own:
            extrapolated.append(workload)
        per_workload[workload] = {"planned": count, "pilot_attempts": len(own),
                                  "cycle_median_s": med, "cycle_p95_s": p95}
        if med is not None and p95 is not None:
            estimate_s += count * med
            p95_estimate_s += count * p95
    budget_h = settings.collection_budget_hours * settings.budget_utilisation
    estimate_h = estimate_s / 3600 if cycles else None
    p95_h = p95_estimate_s / 3600 if cycles else None
    if estimate_h is None:
        mode = "unknown: no timed attempt"
    elif p95_h is not None and p95_h <= budget_h:
        mode = "one attempt per run (p95 estimate fits the budget)"
    elif estimate_h <= budget_h:
        mode = "one attempt per run, tight: median fits, p95 does not; start early"
    else:
        mode = "group the phases of one workload per run: one attempt per run does not fit"

    current_timeout = config.post_window_timeout_s
    # The runner stops waiting as soon as 5 valid post snapshots exist, so a long
    # timeout costs nothing on a good attempt: only raise it, never lower it.
    needed = math.ceil(max(settles) * settings.timeout_headroom / 5.0) * 5 if settles else None
    suggested_timeout = None if needed is None else max(current_timeout, float(needed))
    incomplete = sum(
        1 for r in rows
        if r.get("transaction_status") != "SKIPPED" and r.get("post_window_complete") is False
    )

    statuses = Counter(str(r.get("transaction_status")) for r in rows)
    unintended = sum(
        1 for r in rows
        if r.get("transaction_status") in ("RESTORED", "FAILED") and not r.get("failure_injected")
    )
    attempt_bugs = {r["attempt_id"]: r["bugs"] for r in rows if r["bugs"]}
    to_review = {r["attempt_id"]: r["review"] for r in rows if r["review"]}
    invalid_not_injected = [
        r["attempt_id"] for r in rows if not r["valid"] and not r.get("failure_injected")
    ]
    gate = {
        "enough_attempts": len(rows) >= settings.min_attempts,
        "coverage": coverage_ok,
        "no_code_bugs": not attempt_bugs and not dataset_bugs,
        "validator_pass": not invalid_not_injected,
        "time_known": bool(cycles),
    }
    gate["passed"] = all(gate.values())
    return {
        "root": str(root),
        "attempts": len(rows),
        "gate": gate,
        "validator": {k: dataset[k] for k in ("valid", "invalid", "training_eligible",
                                              "invalid_attempts", "quarantined")},
        "dataset_bugs": dataset_bugs,
        "attempt_bugs": attempt_bugs,
        "review": to_review,
        "coverage": {"seen": seen, "missing": missing, "no_op_attempts": no_op},
        "status_counts": dict(sorted(statuses.items())),
        "skipped_rate": statuses.get("SKIPPED", 0) / len(rows) if rows else None,
        "unintended_rollbacks": unintended,
        "freeze": freeze,
        "timing": {
            "duration_s": _stats(durations),
            "cycle_s": _stats(cycles),
            "settle_s": _stats(settles),
            "transition_hard_gate_blocked": _stats([float(x) for x in blocked]),
            "per_workload": per_workload,
            "extrapolated_workloads": extrapolated,
            "planned_attempts": sum(planned.values()),
            "estimate_hours": estimate_h,
            "estimate_hours_p95": p95_h,
            "budget_hours": budget_h,
        },
        "decisions": {
            "collection_mode": mode,
            "post_window_timeout_s": {"current": current_timeout, "needed": needed,
                                      "suggested": suggested_timeout,
                                      "incomplete_post_windows": incomplete},
            "no_op_controls": config.no_op_controls,
        },
        "attempt_rows": rows,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    """Short human summary (PR evidence / Data Card)."""
    gate = report["gate"]
    timing = report["timing"]
    decisions = report["decisions"]
    lines = [
        f"# Pilot report: {report['root']}",
        "",
        f"Gate Day 3: **{'PASS' if gate['passed'] else 'FAIL'}** "
        + ", ".join(f"{k}={'ok' if v else 'NO'}" for k, v in gate.items() if k != "passed"),
        "",
        f"Attempts {report['attempts']}; status {report['status_counts']}; "
        f"validator valid {report['validator']['valid']}, invalid {report['validator']['invalid']}, "
        f"eligible {report['validator']['training_eligible']}.",
        "",
        "## Decisions",
        "",
        f"- Collection mode: {decisions['collection_mode']}.",
        f"- Estimate for {timing['planned_attempts']} attempts: median "
        f"{_fmt(timing['estimate_hours'])} h, p95 {_fmt(timing['estimate_hours_p95'])} h "
        f"(budget {_fmt(timing['budget_hours'])} h).",
        f"- post_window_timeout_s: current {_fmt(decisions['post_window_timeout_s']['current'], 0)}, "
        f"needed (max settle x headroom) {_fmt(decisions['post_window_timeout_s']['needed'])}, "
        f"suggested {_fmt(decisions['post_window_timeout_s']['suggested'], 0)}, "
        f"incomplete post windows {decisions['post_window_timeout_s']['incomplete_post_windows']}.",
        f"- NO_OP controls in the matrix: {decisions['no_op_controls']}.",
        "",
        "## Time (s)",
        "",
        "| Measure | n | median | p95 | max |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ("duration_s", "cycle_s", "settle_s", "transition_hard_gate_blocked"):
        st = timing[key]
        lines.append(f"| {key} | {st['n']} | {_fmt(st['median'])} | {_fmt(st['p95'])} | "
                     f"{_fmt(st['max'])} |")
    lines += ["", "## Bugs (block the freeze)", ""]
    bugs = list(report["dataset_bugs"]) + [
        f"{a}: {b}" for a, items in report["attempt_bugs"].items() for b in items
    ]
    lines += [f"- {b}" for b in bugs] or ["- none"]
    lines += ["", "## Look at by hand", ""]
    lines += [f"- {a}: {r}" for a, items in report["review"].items() for r in items] or ["- none"]
    missing = {k: v for k, v in report["coverage"]["missing"].items() if v}
    lines += ["", "## Coverage", "",
              f"Missing: {missing or 'none'}; NO_OP attempts: {report['coverage']['no_op_attempts']}.",
              "", "## Freeze (must be one value each)", ""]
    lines += [f"- {k}: {v}" for k, v in report["freeze"].items()]
    return "\n".join(lines) + "\n"
