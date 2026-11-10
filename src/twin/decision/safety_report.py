"""Tables and the one-shot lock of the final Safety Gate evaluation (Week 13, Day 4).

Pure formatting and file helpers: the Gate and the metrics are computed elsewhere. The lock
file is created atomically before the test and safety-stress-v1 results are computed, so the
evaluation cannot be run twice by accident.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.twin.decision.safety_metrics import Rate, SafetyMetrics
from src.twin.decision.safety_replay import ReplayResult

RATE_NAMES = ("ucrr", "uar", "frr", "sap", "doar")
RATE_PARTS = ("num", "den", "value", "low", "high")
CONFUSION_COLUMNS = (
    "split", "gate", "records", "unsafe_rejected", "unsafe_accepted",
    "safe_rejected", "safe_accepted",
)
ABLATION_COLUMNS = ("split", "gate", "m_u", "m_r", "records", "accepted", "unsafe") + tuple(
    f"{name}_{part}" for name in RATE_NAMES for part in RATE_PARTS
)
REASON_COLUMNS = ("split", "gate", "kind", "reason", "count")


def _rate_cells(name: str, rate: Rate) -> dict[str, Any]:
    return {
        f"{name}_num": rate.numerator, f"{name}_den": rate.denominator,
        f"{name}_value": rate.value, f"{name}_low": rate.low, f"{name}_high": rate.high,
    }


def confusion_row(split: str, gate: str, metrics: SafetyMetrics) -> dict[str, Any]:
    """Unsafe = observed U_dst_post above the limit; rejected = the Gate said no."""
    return {
        "split": split, "gate": gate, "records": metrics.records,
        "unsafe_rejected": metrics.unsafe_rejected, "unsafe_accepted": metrics.unsafe_accepted,
        "safe_rejected": metrics.safe_rejected, "safe_accepted": metrics.safe_accepted,
    }


def ablation_row(
    split: str, gate: str, m_u: float, m_r: float, metrics: SafetyMetrics
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split, "gate": gate, "m_u": m_u, "m_r": m_r, "records": metrics.records,
        "accepted": metrics.accepted, "unsafe": metrics.unsafe,
    }
    for name in RATE_NAMES:
        row.update(_rate_cells(name, getattr(metrics, name)))
    return row


def reason_rows(split: str, gate: str, result: ReplayResult) -> list[dict[str, Any]]:
    rows = [
        {"split": split, "gate": gate, "kind": "failed", "reason": reason, "count": count}
        for reason, count in sorted(result.failed_reasons.items())
    ]
    rows += [
        {"split": split, "gate": gate, "kind": "skipped", "reason": reason, "count": count}
        for reason, count in sorted(result.skipped_checks.items())
    ]
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Write the rows (None -> empty cell) and return the sha256 of the file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: "" if row.get(c) is None else row[c] for c in columns})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_lock(path: Path, policy_sha12: str, rule: str) -> None:
    """Create the lock exclusively; FileExistsError when the evaluation was already opened."""
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(
            {"status": "opened", "attempt": 1, "opened_at": _now(),
             "policy_sha12": policy_sha12, "rule": rule}, indent=2) + "\n")


def complete_lock(path: Path, outputs_sha256: Mapping[str, str]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({"status": "completed", "completed_at": _now(),
                 "outputs_sha256": dict(outputs_sha256)})
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
