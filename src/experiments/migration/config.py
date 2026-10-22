"""Experiment matrix of the Week 11 migration-outcome collection.

Loads and validates ``configs/experiments/migration_outcome.yaml`` and expands
it into the ordered list of attempts the runner executes. Pure functions: no
network, no file writes.

Attempt ids (one directory each):
  MIGRATE: ``mig-<workload>-<phase>-dst<45|75>-<src>-<dst>-r<NN>``
  NO_OP:   ``noop-<workload>-<phase>-dst<45|75>-<src>-<dst>-r<NN>``
Workload and phase names must not contain '-', so an id splits cleanly.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.experiments.migration.schema import SCHEMA_VERSION

_REPEAT = re.compile(r"^r\d{2}$")
_NAME = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class OutcomeConfig:
    schema_version: str
    topology: str
    controllers_config: str
    output_root: str
    pre_window_valid_snapshots: int
    post_window_valid_snapshots: int
    post_window_timeout_s: float
    workloads: dict[str, tuple[str, ...]]
    destination_load_levels: tuple[float, ...]
    directions: tuple[tuple[str, str], ...]
    repeats: dict[str, int]
    split: dict[str, tuple[str, ...]]
    no_op_controls: int

    def split_of(self, repeat: str) -> str:
        for name, repeats in self.split.items():
            if repeat in repeats:
                return name
        raise ValueError(f"repeat {repeat} is in no split")


@dataclass(frozen=True)
class AttemptPlan:
    attempt_id: str
    action_type: str
    workload: str
    phase: str
    destination_load_level: float
    source_controller: str
    target_controller: str
    repeat: str
    seed: int
    split: str


def dst_tag(level: float) -> str:
    return f"dst{int(round(level * 100)):02d}"


def attempt_id(
    action_type: str,
    workload: str,
    phase: str,
    destination_load_level: float,
    source: str,
    target: str,
    repeat: str,
) -> str:
    prefix = {"MIGRATE": "mig", "NO_OP": "noop"}.get(action_type)
    if prefix is None:
        raise ValueError(f"unknown action_type {action_type!r}")
    return "-".join(
        (prefix, workload, phase, dst_tag(destination_load_level), source, target, repeat)
    )


def config_sha256(path: Path) -> str:
    """Hash recorded in metadata.json so every attempt names its exact config."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_outcome_config(path: Path) -> OutcomeConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return validate_outcome_config(data)


def validate_outcome_config(data: Mapping[str, Any]) -> OutcomeConfig:
    def need(key: str) -> Any:
        if key not in data:
            raise ValueError(f"migration_outcome config: missing '{key}'")
        return data[key]

    if need("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    pre = int(need("pre_window_valid_snapshots"))
    post = int(need("post_window_valid_snapshots"))
    timeout = float(need("post_window_timeout_s"))
    if pre < 1 or post < 1:
        raise ValueError("window sizes must be >= 1")
    if timeout <= 0:
        raise ValueError("post_window_timeout_s must be > 0")

    workloads: dict[str, tuple[str, ...]] = {}
    for name, phases in dict(need("workloads")).items():
        name = str(name)
        phases = tuple(str(p) for p in phases)
        if not _NAME.match(name) or not all(_NAME.match(p) for p in phases):
            raise ValueError(f"workload/phase names must match {_NAME.pattern}: {name} {phases}")
        if not phases or len(set(phases)) != len(phases):
            raise ValueError(f"workload {name}: phases must be non-empty and unique")
        workloads[name] = phases
    if not workloads:
        raise ValueError("no workloads")

    levels = tuple(float(x) for x in need("destination_load_levels"))
    if not levels or not all(0.0 < x < 1.0 for x in levels):
        raise ValueError("destination_load_levels must be in (0, 1)")
    if len({dst_tag(x) for x in levels}) != len(levels):
        raise ValueError("destination_load_levels collide after rounding to a tag")

    directions: list[tuple[str, str]] = []
    for pair in need("directions"):
        src, dst = (str(x) for x in pair)
        if src == dst or not _NAME.match(src) or not _NAME.match(dst):
            raise ValueError(f"bad direction {pair}")
        directions.append((src, dst))
    if len(set(directions)) != len(directions):
        raise ValueError("duplicate directions")

    repeats = {str(k): int(v) for k, v in dict(need("repeats")).items()}
    if not repeats or not all(_REPEAT.match(k) for k in repeats):
        raise ValueError("repeat keys must look like r01")
    if len(set(repeats.values())) != len(repeats):
        raise ValueError("repeat seeds must be unique")

    split = {str(k): tuple(str(r) for r in v) for k, v in dict(need("split")).items()}
    assigned = [r for rs in split.values() for r in rs]
    if sorted(assigned) != sorted(repeats):
        raise ValueError("split must assign every repeat to exactly one set")

    no_op = int(need("no_op_controls"))
    if no_op < 0:
        raise ValueError("no_op_controls must be >= 0")

    return OutcomeConfig(
        schema_version=SCHEMA_VERSION,
        topology=str(need("topology")),
        controllers_config=str(need("controllers_config")),
        output_root=str(need("output_root")),
        pre_window_valid_snapshots=pre,
        post_window_valid_snapshots=post,
        post_window_timeout_s=timeout,
        workloads=workloads,
        destination_load_levels=levels,
        directions=tuple(directions),
        repeats=repeats,
        split=split,
        no_op_controls=no_op,
    )


def expand_attempts(config: OutcomeConfig) -> list[AttemptPlan]:
    """Fixed-order attempt list: every MIGRATE cell, then the NO_OP controls.

    NO_OP plan: one cell per (workload, destination level, direction) at the
    workload's middle phase; ``no_op_controls`` must be a multiple of the
    number of cells. Cell i takes repeats i, i+stride, ... (mod n) so the
    controls spread evenly over repeats and therefore over the splits.
    """
    plans: list[AttemptPlan] = []
    repeat_ids = sorted(config.repeats)
    for repeat in repeat_ids:
        for workload, phases in config.workloads.items():
            for phase in phases:
                for level in config.destination_load_levels:
                    for src, dst in config.directions:
                        plans.append(_plan("MIGRATE", config, workload, phase, level, src, dst, repeat))

    if config.no_op_controls:
        cells = [
            (workload, phases[len(phases) // 2], level, src, dst)
            for workload, phases in config.workloads.items()
            for level in config.destination_load_levels
            for src, dst in config.directions
        ]
        per_cell, remainder = divmod(config.no_op_controls, len(cells))
        if remainder or per_cell > len(repeat_ids):
            raise ValueError(
                f"no_op_controls={config.no_op_controls} must be a multiple of {len(cells)} "
                f"cells and at most {len(cells) * len(repeat_ids)}"
            )
        stride = max(1, len(repeat_ids) // per_cell)
        for i, (workload, phase, level, src, dst) in enumerate(cells):
            for j in range(per_cell):
                repeat = repeat_ids[(i + j * stride) % len(repeat_ids)]
                plans.append(_plan("NO_OP", config, workload, phase, level, src, dst, repeat))

    ids = [p.attempt_id for p in plans]
    if len(set(ids)) != len(ids):
        raise ValueError("expanded attempt ids are not unique")
    return plans


def _plan(
    action_type: str,
    config: OutcomeConfig,
    workload: str,
    phase: str,
    level: float,
    src: str,
    dst: str,
    repeat: str,
) -> AttemptPlan:
    return AttemptPlan(
        attempt_id=attempt_id(action_type, workload, phase, level, src, dst, repeat),
        action_type=action_type,
        workload=workload,
        phase=phase,
        destination_load_level=level,
        source_controller=src,
        target_controller=dst,
        repeat=repeat,
        seed=config.repeats[repeat],
        split=config.split_of(repeat),
    )
