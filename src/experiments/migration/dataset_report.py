"""Dataset report of the migration-outcome collection (Week 11, Day 4-6).

Reads an attempt root (pilot or official), never writes into it, and builds
what plan sections 27, 28, 36 and 44-46 ask for:

* ``migration_outcome_manifest.json``  one entry per attempt (identity, split,
  status, validity, a fingerprint of its SHA256SUMS);
* ``split_manifest.json``               split by repeat (r01-r03 train, r04
  validation, r05 test); every sample of an attempt stays in its attempt's
  split, NO_OP controls follow their repeat;
* ``dataset_quality_report.json``       the section-36 counters, progress per
  repeat and the freeze values;
* tables: ``attempt_summary.csv``, ``committed_summary.csv``,
  ``rollback_summary.csv``, ``workload_distribution.csv``,
  ``noop_vs_migrate.csv``;
* ``migration_latency.json``            transaction time, post-window settle
  time, snapshots blocked by the Hard Gate;
* ``analytic_check.json``               -Delta U_src against the switch's
  control-load share and against the analytic estimate rate / C_safe (the
  evidence Week 12's analytic outcome model starts from).

Pure functions and the standard library only (figures are drawn by
``scripts/analysis/migration_dataset_report.py``). Nothing here trains or
filters data: every attempt is reported, invalid ones included.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.experiments.migration.config import OutcomeConfig, config_sha256, dst_tag, expand_attempts
from src.experiments.migration.metrics import DEFAULT_WINDOW
from src.experiments.migration.pilot import (
    FREEZE_KEYS,
    PilotSettings,
    analyze_attempt,
    percentile,
)
from src.experiments.migration.schema import SCHEMA_VERSION
from src.experiments.migration.validator import validate_dataset
from src.experiments.migration.writer import CHECKSUM_FILE

STATIONARY_WORKLOADS = ("stable", "hot_switch")
STATUSES = ("COMMITTED", "RESTORED", "FAILED", "SKIPPED")

ATTEMPT_COLUMNS = (
    "attempt_id", "action_type", "workload", "phase", "destination_load_level", "direction",
    "repeat", "split", "switch_id", "target_hot", "failure_injected", "transaction_status",
    "valid", "training_eligible", "ineligible_reasons", "switch_share_before_mean",
    "switch_rate_before_mean", "src_u_before_mean", "src_u_after_mean", "delta_src_u_mean",
    "delta_src_u_last", "dst_u_before_mean", "dst_u_after_mean", "delta_dst_u_mean",
    "delta_dst_u_last", "expected_delta_src", "expected_delta_dst", "imbalance_before_mean",
    "imbalance_after_mean", "delta_imbalance_mean", "dst_response_p95_before_mean",
    "dst_response_p95_after_mean", "before_quality_min", "after_quality_min",
    "before_consistency_min", "after_consistency_min", "post_window_complete",
    "transaction_s", "settle_s", "transition_hard_gate_blocked", "duration_s",
)
ROLLBACK_COLUMNS = (
    "attempt_id", "workload", "phase", "direction", "repeat", "switch_id", "failure_injected",
    "transaction_status", "failure_reason", "history", "transaction_s", "valid",
    "post_window_complete",
)
DISTRIBUTION_KEYS = (
    "workload", "phase", "destination_load_level", "direction", "action_type",
    "transaction_status",
)


# --------------------------------------------------------------------------
# reading one attempt
# --------------------------------------------------------------------------
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(attempt_dir: Path) -> str | None:
    path = attempt_dir / CHECKSUM_FILE
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attempt_record(
    attempt_dir: Path,
    config: OutcomeConfig,
    settings: PilotSettings,
    required_window: int = DEFAULT_WINDOW,
) -> dict[str, Any]:
    """Pilot measurements of one attempt + the full outcome, quality and transaction."""
    row = analyze_attempt(attempt_dir, settings, required_window)
    row["fingerprint"] = _fingerprint(attempt_dir)
    try:
        metadata = _read_json(attempt_dir / "metadata.json")
        stored = _read_json(attempt_dir / "outcome.json")
        raw_tx = _read_json(attempt_dir / "transaction.json")
    except (OSError, ValueError):
        return row
    row["seed"] = metadata.get("seed")
    outcome = stored.get("outcome", {}) if isinstance(stored, Mapping) else {}
    quality = stored.get("quality", {}) if isinstance(stored, Mapping) else {}
    for key, value in outcome.items():
        row.setdefault(key, value)
    row["ineligible_reasons"] = ";".join(outcome.get("ineligible_reasons", []))
    for key in ("before_count", "after_count", "before_all_valid", "after_all_valid",
                "before_quality_min", "after_quality_min", "before_consistency_min",
                "after_consistency_min"):
        row[key] = quality.get(key)
    repeat = row.get("repeat")
    row["split"] = _split_of(config, repeat)
    body = raw_tx.get("body") if isinstance(raw_tx, Mapping) else None
    tx = body.get("transaction") if isinstance(body, Mapping) else None
    if isinstance(body, Mapping) and isinstance(body.get("detail"), Mapping):
        tx = body["detail"].get("transaction", tx)
    history = [str(h.get("state")) for h in tx.get("history", [])] if isinstance(tx, Mapping) else []
    row["history"] = ">".join(history)
    row["failure_reason"] = (
        (body.get("failure_reason") if isinstance(body, Mapping) else None)
        or (tx.get("failure_reason") if isinstance(tx, Mapping) else None)
    )
    row["transaction_id"] = tx.get("transaction_id") if isinstance(tx, Mapping) else None
    return row


def _split_of(config: OutcomeConfig, repeat: Any) -> str | None:
    if repeat is None:
        return None
    try:
        return config.split_of(str(repeat))
    except ValueError:
        return None


def _attempt_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


# --------------------------------------------------------------------------
# small statistics (standard library only)
# --------------------------------------------------------------------------
def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def describe(values: Sequence[float]) -> dict[str, float | int | None]:
    """count, mean, std (population), min, p25, median, p75, p95, max (nearest rank)."""
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "p25": None,
                "median": None, "p75": None, "p95": None, "max": None}
    ordered = sorted(values)

    def rank(q: float) -> float:
        value = percentile(ordered, q)
        assert value is not None
        return value

    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p25": rank(25),
        "median": statistics.median(ordered),
        "p75": rank(75),
        "p95": rank(95),
        "max": ordered[-1],
    }


def fit_line(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float | int | None]:
    """Least-squares y = a + b x, Pearson r and the slope through the origin."""
    n = len(xs)
    result: dict[str, float | int | None] = {
        "n": n, "intercept": None, "slope": None, "pearson_r": None, "slope_through_origin": None,
    }
    if n < 3 or len(set(xs)) < 2:
        return result
    slope, intercept = statistics.linear_regression(xs, ys)
    result["intercept"] = intercept
    result["slope"] = slope
    if len(set(ys)) > 1:
        result["pearson_r"] = statistics.correlation(xs, ys)
    sxx = sum(x * x for x in xs)
    result["slope_through_origin"] = sum(x * y for x, y in zip(xs, ys)) / sxx if sxx else None
    return result


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def build_dataset_report(
    root: Path,
    config: OutcomeConfig,
    settings: PilotSettings,
    config_path: Path | None = None,
) -> dict[str, Any]:
    window = config.post_window_valid_snapshots
    dataset = validate_dataset(root, config, window)
    rows = [attempt_record(d, config, settings, window) for d in _attempt_dirs(root)]
    readable = [r for r in rows if r.get("workload") is not None]
    planned = expand_attempts(config)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "config_sha256": config_sha256(config_path) if config_path else None,
        "attempts": [
            {key: r.get(key) for key in (
                "attempt_id", "action_type", "workload", "phase", "destination_load_level",
                "direction", "repeat", "seed", "split", "switch_id", "target_hot",
                "failure_injected", "transaction_status", "transaction_id", "valid",
                "training_eligible", "fingerprint")}
            for r in rows
        ],
    }
    return {
        "manifest": manifest,
        "split_manifest": split_manifest(rows, config),
        "quality_report": quality_report(rows, readable, dataset, planned, window),
        "latency": latency_report(readable),
        "analytic_check": analytic_check(readable),
        "tables": {
            "attempt_summary": table(readable, ATTEMPT_COLUMNS),
            "committed_summary": table(
                [r for r in readable if r.get("transaction_status") == "COMMITTED"], ATTEMPT_COLUMNS),
            "rollback_summary": table(
                [r for r in readable if r.get("transaction_status") in ("RESTORED", "FAILED")],
                ROLLBACK_COLUMNS),
            "workload_distribution": workload_distribution(readable),
            "noop_vs_migrate": noop_vs_migrate(readable),
        },
        "rows": rows,
    }


def split_manifest(rows: Sequence[Mapping[str, Any]], config: OutcomeConfig) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for name, repeats in config.split.items():
        members = sorted(r["attempt_id"] for r in rows if r.get("split") == name)
        chosen = [r for r in rows if r.get("split") == name]
        splits[name] = {
            "repeats": list(repeats),
            "attempts": members,
            "count": len(members),
            "migrate": sum(1 for r in chosen if r.get("action_type") == "MIGRATE"),
            "no_op": sum(1 for r in chosen if r.get("action_type") == "NO_OP"),
            "training_eligible": sum(1 for r in chosen if r.get("training_eligible")),
        }
    owners = Counter(a for split in splits.values() for a in split["attempts"])
    return {
        "rule": ("split by repeat (scenario + repeat), never by row: every pre/post sample of "
                 "an attempt stays in that attempt's split; NO_OP controls follow their repeat"),
        "splits": splits,
        "unassigned": sorted(r["attempt_id"] for r in rows if r.get("split") is None),
        "attempts_in_more_than_one_split": sorted(a for a, n in owners.items() if n > 1),
    }


def quality_report(
    rows: Sequence[Mapping[str, Any]],
    readable: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    planned: Sequence[Any],
    window: int,
) -> dict[str, Any]:
    statuses = Counter(str(r.get("transaction_status")) for r in readable)

    def count(pred: Any) -> int:
        return sum(1 for r in readable if pred(r))

    def mentions(r: Mapping[str, Any], words: tuple[str, ...]) -> bool:
        text = " ".join(list(r.get("errors", [])) + list(r.get("bugs", []))).lower()
        return any(w in text for w in words)

    planned_by_repeat = Counter(p.repeat for p in planned)
    present = Counter(r.get("repeat") for r in readable)
    valid = Counter(r.get("repeat") for r in readable if r.get("valid"))
    eligible = Counter(r.get("repeat") for r in readable if r.get("training_eligible"))
    freeze = {
        key: sorted({str(r["freeze"][key]) for r in readable if "freeze" in r})
        for key in FREEZE_KEYS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "total_attempts": len(rows),
        "planned_attempts": len(planned),
        "missing_attempts": len(dataset.get("missing", [])),
        "unplanned_attempts": len(dataset.get("unplanned", [])),
        "quarantined": len(dataset.get("quarantined", [])),
        "valid_count": count(lambda r: r.get("valid")),
        "invalid_count": len(rows) - count(lambda r: r.get("valid")),
        "migrate_count": count(lambda r: r.get("action_type") == "MIGRATE"),
        "no_op_count": count(lambda r: r.get("action_type") == "NO_OP"),
        "committed_count": statuses.get("COMMITTED", 0),
        "restored_count": statuses.get("RESTORED", 0),
        "failed_count": statuses.get("FAILED", 0),
        "skipped_count": statuses.get("SKIPPED", 0),
        "failure_injected_count": count(lambda r: r.get("failure_injected")),
        "valid_pre_count": count(
            lambda r: (r.get("before_count") or 0) >= window and r.get("before_all_valid") is True),
        "valid_post_count": count(
            lambda r: r.get("post_window_complete") is True and r.get("after_all_valid") is True),
        "post_window_incomplete_count": count(
            lambda r: r.get("transaction_status") != "SKIPPED"
            and r.get("post_window_complete") is False),
        "training_eligible_count": count(lambda r: r.get("training_eligible")),
        "training_eligible_committed_count": count(
            lambda r: r.get("training_eligible") and r.get("transaction_status") == "COMMITTED"),
        "training_eligible_no_op_count": count(
            lambda r: r.get("training_eligible") and r.get("action_type") == "NO_OP"),
        "workload_counts": dict(sorted(Counter(r["workload"] for r in readable).items())),
        "phase_counts": dict(sorted(
            Counter(f"{r['workload']}:{r['phase']}" for r in readable).items())),
        "direction_counts": dict(sorted(Counter(r["direction"] for r in readable).items())),
        "destination_load_counts": dict(sorted(
            Counter(dst_tag(float(r["destination_load_level"])) for r in readable).items())),
        "split_counts": dict(sorted(Counter(str(r.get("split")) for r in readable).items())),
        "rollback_count": count(lambda r: "ROLLING_BACK" in str(r.get("history", ""))),
        "verification_failure_count": count(
            lambda r: "verification failed" in str(r.get("failure_reason") or "").lower()),
        "missing_field_count": count(lambda r: mentions(r, ("missing", "unreadable")))
        + (len(rows) - len(readable)),
        "invalid_quality_count": count(
            lambda r: r.get("before_all_valid") is False
            or ((r.get("after_count") or 0) > 0 and r.get("after_all_valid") is False)),
        "duplicate_transaction_count": len(dataset.get("duplicate_transactions", {})),
        "ownership_mismatch_count": count(
            lambda r: mentions(r, ("owned by", "not owned", "ownership", "roles"))),
        "code_bug_attempts": count(lambda r: bool(r.get("bugs"))),
        "progress_by_repeat": {
            repeat: {"planned": planned_by_repeat[repeat], "present": present.get(repeat, 0),
                     "valid": valid.get(repeat, 0), "training_eligible": eligible.get(repeat, 0)}
            for repeat in sorted(planned_by_repeat)
        },
        "freeze": freeze,
        "freeze_consistent": all(len(v) <= 1 for v in freeze.values()),
        "invalid_attempts": dict(sorted(dataset.get("invalid_attempts", {}).items())),
    }


def latency_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    executed = [r for r in rows if r.get("transaction_status") in ("COMMITTED", "RESTORED", "FAILED")]
    by_workload: dict[str, Any] = {}
    for workload in sorted({str(r["workload"]) for r in executed}):
        own = [r for r in executed if r["workload"] == workload]
        by_workload[workload] = {
            "transaction_s": describe(_numbers(own, "transaction_s")),
            "settle_s": describe(_numbers(own, "settle_s")),
        }
    return {
        "definition": {
            "transaction_s": "orchestrator transaction started_at -> finished_at",
            "settle_s": "transaction finished_at -> 5th Hard-Gate-valid post snapshot",
            "transition_hard_gate_blocked": "snapshots between the action and the post window "
                                            "that failed the Hard Gate",
        },
        "transaction_s": describe(_numbers(executed, "transaction_s")),
        "transaction_s_by_status": {
            status: describe(_numbers([r for r in executed if r.get("transaction_status") == status],
                                      "transaction_s"))
            for status in ("COMMITTED", "RESTORED", "FAILED")
        },
        "settle_s": describe(_numbers(executed, "settle_s")),
        "transition_hard_gate_blocked": describe(_numbers(executed, "transition_hard_gate_blocked")),
        "attempt_duration_s": describe(_numbers(rows, "duration_s")),
        "by_workload": by_workload,
    }


def analytic_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """-Delta U_src vs the moved switch's share and vs the analytic rate / C_safe."""
    eligible = [r for r in rows if r.get("transaction_status") == "COMMITTED"
                and r.get("training_eligible") and r.get("delta_src_u_mean") is not None]

    def group(sub: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        pairs = [(float(r["switch_share_before_mean"]), -float(r["delta_src_u_mean"]))
                 for r in sub if r.get("switch_share_before_mean") is not None]
        expected = [(-float(r["expected_delta_src"]), -float(r["delta_src_u_mean"]))
                    for r in sub if r.get("expected_delta_src") is not None]
        ratios = [o / e for e, o in expected if e]
        return {
            "n": len(sub),
            "share_vs_neg_delta_src": fit_line([p[0] for p in pairs], [p[1] for p in pairs]),
            "expected_vs_observed_neg_delta_src": fit_line(
                [p[0] for p in expected], [p[1] for p in expected]),
            "observed_over_expected": describe(ratios),
        }

    stationary = [r for r in eligible if r.get("workload") in STATIONARY_WORKLOADS]
    return {
        "definition": ("x = control-load share of the moved switch (pre window), y = -Delta U_src "
                       "(after_mean - before_mean); expected = switch rate / C_safe of the source"),
        "all_committed": group(eligible),
        "stationary_workloads": {"workloads": list(STATIONARY_WORKLOADS), **group(stationary)},
        "by_workload": {w: group([r for r in eligible if r.get("workload") == w])
                        for w in sorted({str(r["workload"]) for r in eligible})},
    }


def workload_distribution(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[Any, ...]] = Counter()
    eligible: Counter[tuple[Any, ...]] = Counter()
    for r in rows:
        key = tuple(r.get(k) if k != "destination_load_level" else dst_tag(float(r[k]))
                    for k in DISTRIBUTION_KEYS)
        counts[key] += 1
        eligible[key] += int(bool(r.get("training_eligible")))
    return [
        {**dict(zip(DISTRIBUTION_KEYS, key)), "attempts": n, "training_eligible": eligible[key]}
        for key, n in sorted(counts.items(), key=lambda kv: tuple(str(x) for x in kv[0]))
    ]


def noop_vs_migrate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Section 28: how much the load moves by itself (NO_OP) vs with a migration."""
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"MIGRATE": [], "NO_OP": []})
    for r in rows:
        if not r.get("training_eligible") or r.get("delta_src_u_mean") is None:
            continue
        if r.get("action_type") == "MIGRATE" and r.get("transaction_status") != "COMMITTED":
            continue
        cells[(str(r["workload"]), str(r["phase"]))][str(r["action_type"])].append(
            float(r["delta_src_u_mean"]))
    out = []
    for (workload, phase), values in sorted(cells.items()):
        mig, noop = values["MIGRATE"], values["NO_OP"]
        mig_mean = statistics.fmean(mig) if mig else None
        noop_mean = statistics.fmean(noop) if noop else None
        out.append({
            "workload": workload, "phase": phase,
            "migrate_n": len(mig), "migrate_delta_src_mean": mig_mean,
            "no_op_n": len(noop), "no_op_delta_src_mean": noop_mean,
            "migration_effect_src": (mig_mean - noop_mean
                                     if mig_mean is not None and noop_mean is not None else None),
        })
    return out


def table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    return [{c: r.get(c) for c in columns} for r in sorted(rows, key=lambda r: r["attempt_id"])]


def to_csv(records: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> str:
    """CSV text with LF line ends (same as week10_evidence)."""
    cols = list(columns) if columns is not None else (list(records[0]) if records else [])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow({c: _cell(record.get(c)) for c in cols})
    return buffer.getvalue()


def _cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return value


def write_report(report: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Write the JSON and CSV files; returns the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def dump(name: str, value: Any) -> None:
        path = out_dir / name
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
                        encoding="utf-8")
        written.append(path)

    dump("migration_outcome_manifest.json", report["manifest"])
    dump("split_manifest.json", report["split_manifest"])
    dump("dataset_quality_report.json", report["quality_report"])
    dump("migration_latency.json", report["latency"])
    dump("analytic_check.json", report["analytic_check"])
    columns = {
        "attempt_summary": ATTEMPT_COLUMNS,
        "committed_summary": ATTEMPT_COLUMNS,
        "rollback_summary": ROLLBACK_COLUMNS,
        "workload_distribution": (*DISTRIBUTION_KEYS, "attempts", "training_eligible"),
        "noop_vs_migrate": ("workload", "phase", "migrate_n", "migrate_delta_src_mean", "no_op_n",
                            "no_op_delta_src_mean", "migration_effect_src"),
    }
    for name, records in report["tables"].items():
        path = out_dir / f"{name}.csv"
        path.write_text(to_csv(records, columns[name]), encoding="utf-8")
        written.append(path)
    return written
