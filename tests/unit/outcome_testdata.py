"""Synthetic raw attempts for the outcome-model tests (written with the Week 11 writer)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Sequence

from src.experiments.migration import fixtures as fx
from src.experiments.migration.metrics import compute_outcome, summarize_quality
from src.experiments.migration.schema import StateSnapshotView
from src.experiments.migration.writer import transaction_from_raw, write_attempt

CAPACITY = fx.SAFE_CAPACITY


def pre_window(src_u: Sequence[float], dst_u: Sequence[float], share: float = 0.2,
               start: int = 0) -> list[dict[str, Any]]:
    return [fx.make_twin_state({"c1": s, "c2": d}, index=start + i, hot_switch=("s1", share))
            for i, (s, d) in enumerate(zip(src_u, dst_u))]


def views(raw: Sequence[dict[str, Any]]) -> list[StateSnapshotView]:
    return [StateSnapshotView.from_twin_state(x) for x in raw]


def write_raw_attempt(
    root: Path,
    attempt_id: str,
    repeat: str,
    kind: str = "committed",
    src_u: Sequence[float] = (0.80, 0.80, 0.80, 0.80, 0.80),
    dst_u: Sequence[float] = (0.45, 0.45, 0.45, 0.45, 0.45),
    share: float = 0.2,
    after: tuple[float, float] = (0.64, 0.61),
    workload: str = "stable",
) -> None:
    """kind: committed, restored or no_op; source c1, target c2, moved switch s1."""
    base = fx.attempt_inputs(kind, attempt_id=attempt_id)
    before_raw = pre_window(src_u, dst_u, share)
    moved = dict(fx.default_ownership())
    moved["s1"] = "c2"
    if kind == "committed":
        after_raw = fx.make_window([{"c1": after[0], "c2": after[1]}] * 5, 10,
                                   ownership=moved, ownership_version=2)
    else:
        after_raw = fx.make_window([{"c1": after[0], "c2": after[1]}] * 5, 10)
    action = base["action"]
    transaction = transaction_from_raw(base["transaction_raw"])
    before, after_views = views(before_raw), views(after_raw)
    metadata = dataclasses.replace(base["metadata"], attempt_id=attempt_id, repeat=repeat,
                                   workload=workload)
    write_attempt(
        root, metadata=metadata, before_raw=before_raw, transition_raw=base["transition_raw"],
        after_raw=after_raw, forecast_raw=[], action=action,
        transaction_raw=base["transaction_raw"],
        outcome=compute_outcome(before, after_views, action, transaction),
        quality=summarize_quality(before, after_views),
    )


def write_split(path: Path, splits: dict[str, list[str]]) -> Path:
    path.write_text(json.dumps({"splits": {k: {"attempts": v} for k, v in splits.items()}}))
    return path


def write_capacity(path: Path, c1: float = CAPACITY, c2: float = CAPACITY) -> Path:
    path.write_text(json.dumps({"controllers": {"c1": {"c_safe_pps": c1},
                                                "c2": {"c_safe_pps": c2}}}))
    return path


def standard_raw(root: Path) -> dict[str, list[str]]:
    """Four committed attempts (2 train, validation, test), one NO_OP, one RESTORED."""
    write_raw_attempt(root, "mig-stable-middle-dst45-c1-c2-r01", "r01",
                      src_u=(0.78, 0.79, 0.80, 0.81, 0.82), share=0.20, after=(0.64, 0.61))
    write_raw_attempt(root, "mig-gradual-u60-dst75-c1-c2-r03", "r03", workload="gradual",
                      src_u=(0.60, 0.62, 0.64, 0.66, 0.68), dst_u=(0.70, 0.71, 0.72, 0.73, 0.74),
                      share=0.10, after=(0.58, 0.79))
    write_raw_attempt(root, "mig-burst-burst_1-dst45-c1-c2-r04", "r04", workload="burst",
                      src_u=(0.70, 0.75, 0.80, 0.85, 0.90), share=0.30, after=(0.70, 0.72))
    write_raw_attempt(root, "mig-hot_switch-peak-dst75-c1-c2-r05", "r05", workload="hot_switch",
                      dst_u=(0.74, 0.75, 0.75, 0.76, 0.76), share=0.45, after=(0.45, 1.08))
    write_raw_attempt(root, "noop-stable-middle-dst45-c1-c2-r01", "r01", kind="no_op",
                      after=(0.80, 0.45))
    write_raw_attempt(root, "mig-stable-early-dst45-c1-c2-r02", "r02", kind="restored",
                      after=(0.80, 0.45))
    return {
        "train": ["mig-stable-middle-dst45-c1-c2-r01", "noop-stable-middle-dst45-c1-c2-r01",
                  "mig-stable-early-dst45-c1-c2-r02", "mig-gradual-u60-dst75-c1-c2-r03"],
        "validation": ["mig-burst-burst_1-dst45-c1-c2-r04"],
        "test": ["mig-hot_switch-peak-dst75-c1-c2-r05"],
    }
