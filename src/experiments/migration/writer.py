"""Write one migration attempt directory, immutably (Week 11).

The runner calls ``write_attempt`` once per attempt. The directory is built
under a temporary name, validated, given a ``validation.json`` and a
``SHA256SUMS`` file (``sha256sum -c SHA256SUMS`` works), renamed to
``<output_root>/<attempt_id>`` and made read-only. An existing attempt is never
overwritten. If writing fails half-way (a bug), the partial directory is moved
to ``<output_root>/_invalid/`` so it can be inspected, and the error is raised.

Raw payloads are stored as received from the orchestrator; ``transaction.json``
wraps the HTTP exchange so the status mapping lives in one place
(``transaction_from_raw``), used by both the runner and the validator.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.experiments.migration.schema import (
    SCHEMA_VERSION,
    AttemptMetadata,
    MigrationAction,
    MigrationTransactionView,
    PostMigrationOutcome,
    QualitySummary,
)

CHECKSUM_FILE = "SHA256SUMS"
VALIDATION_FILE = "validation.json"
INVALID_DIR = "_invalid"
DATA_FILES = (
    "metadata.json",
    "before_samples.jsonl",
    "action.json",
    "transaction.json",
    "transition_samples.jsonl",
    "after_samples.jsonl",
    "forecast_samples.jsonl",
    "outcome.json",
)


def raw_transaction(
    http_status: int | None = None,
    body: Any = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """What goes into transaction.json.

    MIGRATE that reached the orchestrator: ``http_status`` + response ``body``.
    MIGRATE stopped by a precondition: only ``skip_reason``.
    NO_OP: all None.
    """
    return {"http_status": http_status, "body": body, "skip_reason": skip_reason}


def transaction_from_raw(raw: Mapping[str, Any]) -> MigrationTransactionView | None:
    """Map transaction.json to the view (None = NO_OP, nothing was attempted).

    200 -> COMMITTED / RESTORED from the body; 500 -> FAILED from body['detail'];
    any other HTTP status -> SKIPPED 'http_<code>' (the endpoint refused before
    creating a transaction); no HTTP call + skip_reason -> SKIPPED.
    """
    code = raw.get("http_status")
    body = raw.get("body")
    reason = raw.get("skip_reason")
    if code is None:
        return None if reason is None else MigrationTransactionView.skipped(str(reason))
    code = int(code)
    if code == 200:
        if not isinstance(body, Mapping):
            raise ValueError("HTTP 200 transaction without a JSON body")
        return MigrationTransactionView.from_response(body)
    if code >= 500:
        detail = body.get("detail", body) if isinstance(body, Mapping) else body
        return MigrationTransactionView.failed(detail if detail is not None else f"http_{code}")
    return MigrationTransactionView.skipped(f"http_{code}")


@dataclass(frozen=True)
class WrittenAttempt:
    path: Path
    validation: dict[str, Any]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(directory: Path) -> None:
    lines = [
        f"{sha256_file(p)}  {p.name}"
        for p in sorted(directory.iterdir())
        if p.is_file() and p.name != CHECKSUM_FILE
    ]
    (directory / CHECKSUM_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_checksums(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (directory / CHECKSUM_FILE).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            result[name.strip()] = digest
    return result


def make_read_only(directory: Path) -> None:
    for path in directory.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
                    | stat.S_IROTH | stat.S_IXOTH)


def _make_writable(directory: Path) -> None:
    directory.chmod(stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    for path in directory.iterdir():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP)


def move_to_invalid(attempt_dir: Path, reason: str) -> Path:
    """Move an attempt broken by a code bug to <root>/_invalid/, never delete it."""
    root = attempt_dir.parent
    target_root = root / INVALID_DIR
    target_root.mkdir(exist_ok=True)
    name = attempt_dir.name.removeprefix(".tmp-")
    target = target_root / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}"
    if attempt_dir.exists() and not os.access(attempt_dir, os.W_OK):
        _make_writable(attempt_dir)
    shutil.move(str(attempt_dir), str(target))
    (target / "INVALID_REASON.txt").write_text(reason + "\n", encoding="utf-8")
    return target


def write_attempt(
    output_root: Path,
    metadata: AttemptMetadata,
    before_raw: Sequence[Mapping[str, Any]],
    transition_raw: Sequence[Mapping[str, Any]],
    after_raw: Sequence[Mapping[str, Any]],
    forecast_raw: Sequence[Mapping[str, Any]],
    action: MigrationAction,
    transaction_raw: Mapping[str, Any],
    outcome: PostMigrationOutcome,
    quality: QualitySummary,
    required_window: int = 5,
) -> WrittenAttempt:
    """Write, validate, checksum and lock one attempt directory."""
    from src.experiments.migration.validator import validate_attempt

    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / metadata.attempt_id
    if final.exists():
        raise FileExistsError(f"attempt already written: {final}")
    tmp = output_root / f".tmp-{metadata.attempt_id}-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _write_json(tmp / "metadata.json", metadata.to_dict())
        _write_jsonl(tmp / "before_samples.jsonl", before_raw)
        _write_json(tmp / "action.json", action.to_dict())
        _write_json(tmp / "transaction.json", dict(transaction_raw))
        _write_jsonl(tmp / "transition_samples.jsonl", transition_raw)
        _write_jsonl(tmp / "after_samples.jsonl", after_raw)
        _write_jsonl(tmp / "forecast_samples.jsonl", forecast_raw)
        _write_json(
            tmp / "outcome.json",
            {"schema_version": SCHEMA_VERSION, "outcome": outcome.to_dict(),
             "quality": quality.to_dict()},
        )
        validation = validate_attempt(tmp, required_window=required_window,
                                      check_checksums=False)
        _write_json(tmp / VALIDATION_FILE, validation)
        write_checksums(tmp)
        tmp.rename(final)
    except Exception as exc:
        if tmp.exists():
            move_to_invalid(tmp, f"write_attempt failed: {exc!r}")
        raise
    make_read_only(final)
    return WrittenAttempt(final, validation)
