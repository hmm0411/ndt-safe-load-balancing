"""Per-controller workload hosts derived from a controllers config (Week 15, Day 1).

The forecast runners used literal 2C20S tables (c1: h1 -> h10, c2: h11 -> h20). Both
topologies number hosts like their switches (h<i> = 10.0.0.<i> on s<i>), and every
controller owns one contiguous slice of the switch chain, so the tables follow from the
config: the source host is on the first switch of the slice, the sink on the last one.
configs/controllers_2c20s.yaml gives back exactly the old literals (tested).

NDT_CONTROLLERS_CONFIG selects the config; unset keeps the 2C20S default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG = Path("configs/controllers_2c20s.yaml")
ENV = "NDT_CONTROLLERS_CONFIG"
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ControllerPath:
    controller: str
    switch_indices: tuple[int, ...]

    @property
    def source(self) -> tuple[str, str]:
        first = self.switch_indices[0]
        return f"h{first}", f"10.0.0.{first}"

    @property
    def sink(self) -> tuple[str, str]:
        last = self.switch_indices[-1]
        return f"h{last}", f"10.0.0.{last}"


def _index(switch: str) -> int:
    if not switch.startswith("s") or not switch[1:].isdigit():
        raise ValueError(f"switch ids must look like s<number>, got {switch!r}")
    return int(switch[1:])


def paths_from_config(config: Mapping[str, Any]) -> tuple[ControllerPath, ...]:
    owners: dict[str, list[int]] = {}
    for switch, owner in config["initial_ownership"].items():
        owners.setdefault(str(owner), []).append(_index(str(switch)))
    unknown = set(owners) - {str(c) for c in config["controllers"]}
    if unknown:
        raise ValueError(f"initial_ownership names unknown controllers: {sorted(unknown)}")
    paths = []
    for controller in sorted(owners):
        indices = tuple(sorted(owners[controller]))
        if len(indices) < 2 or indices != tuple(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"{controller}: owned switches must be one contiguous slice "
                             f"of at least 2, got {indices}")
        paths.append(ControllerPath(controller, indices))
    return tuple(paths)


def config_path() -> Path:
    raw = os.environ.get(ENV)
    path = Path(raw) if raw else DEFAULT_CONFIG
    return path if path.is_absolute() else REPO_ROOT / path


def load_paths(path: Path | None = None) -> tuple[ControllerPath, ...]:
    with (path or config_path()).open(encoding="utf-8") as handle:
        return paths_from_config(yaml.safe_load(handle))


def controller_ids(paths: tuple[ControllerPath, ...]) -> tuple[str, ...]:
    return tuple(p.controller for p in paths)


def sinks(paths: tuple[ControllerPath, ...], port: int = 9000) -> tuple[tuple[str, str, int], ...]:
    return tuple((*p.sink, port) for p in paths)


def generators(paths: tuple[ControllerPath, ...]) -> tuple[tuple[str, str, str, str], ...]:
    return tuple((*p.source, *p.sink) for p in paths)


def controller_hosts(paths: tuple[ControllerPath, ...]) -> dict[str, tuple[int, ...]]:
    return {p.controller: p.switch_indices for p in paths}


def hot_hosts(paths: tuple[ControllerPath, ...]) -> dict[str, int]:
    return {p.controller: p.switch_indices[0] for p in paths}
