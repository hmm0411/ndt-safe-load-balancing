"""Frozen model selection and the one-time test lock (Week 12, Day 4).

``model_selection.json`` is written from validation only, BEFORE any test row is
read. Its ``frozen`` part (chosen models, validation scores, sha256 of every model
file and config) is hashed; ``test_lock.json`` records that hash when the test split
is opened and is completed once every output is written.

A second evaluation is refused once a lock is completed. If a run crashed after
opening the test (lock status ``opened``), it may be run again only with the SAME
frozen selection: nothing chosen can change, and the attempt count is recorded.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LOCK_FILE = "test_lock.json"
SELECTION_FILE = "model_selection.json"


class LockedTestError(RuntimeError):
    """The test split was already evaluated for this output directory."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(data: Any) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_sha256(paths: Mapping[str, Path]) -> dict[str, str]:
    """sha256 of each file, keyed by a stable name; missing files are an error."""
    out = {}
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        out[name] = file_sha256(path)
    return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_lock(out: Path) -> dict[str, Any] | None:
    path = out / LOCK_FILE
    if not path.is_file():
        return None
    return dict(json.loads(path.read_text(encoding="utf-8")))


def check_can_open(out: Path, selection_sha: str) -> int:
    """Attempt number of the next opening; raises if the test may not be opened."""
    lock = read_lock(out)
    if lock is None:
        return 1
    if lock.get("status") == "completed":
        raise LockedTestError(
            f"{out / LOCK_FILE}: test already evaluated at {lock.get('completed_at')} "
            f"(selection {str(lock.get('selection_sha256'))[:12]}); it is opened only once")
    if lock.get("selection_sha256") != selection_sha:
        raise LockedTestError(
            f"{out / LOCK_FILE}: an earlier run opened the test with another selection "
            f"({str(lock.get('selection_sha256'))[:12]}); refusing to evaluate a new one")
    return int(lock.get("attempt", 1)) + 1


def open_test(out: Path, selection_sha: str) -> dict[str, Any]:
    """Record that the test split is being opened; refuse a second evaluation."""
    attempt = check_can_open(out, selection_sha)
    lock = read_lock(out)
    new = {"status": "opened", "selection_sha256": selection_sha, "attempt": attempt,
           "opened_at": _now(),
           "rule": "test evaluated once; selection frozen before the test rows are read"}
    if lock is not None:
        new["previous_opened_at"] = lock.get("opened_at")
    (out / LOCK_FILE).write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return new


def complete_test(out: Path, outputs: Mapping[str, Path],
                  summary: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_lock(out)
    if lock is None or lock.get("status") != "opened":
        raise LockedTestError("complete_test called without an opened lock")
    lock.update({"status": "completed", "completed_at": _now(),
                 "outputs_sha256": tree_sha256(outputs), "test_summary": dict(summary)})
    (out / LOCK_FILE).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock
