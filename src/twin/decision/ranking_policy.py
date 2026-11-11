"""Ranking Policy v1: versioned weights of the ranking score, with provenance.

Every component keeps its configured weight; a disabled one needs a reason and the
enabled weights are renormalized to sum to 1 (the rule the selection score S uses when
r95_post is excluded). Weight sets for the sensitivity analysis live in the same file
so they are fixed before any result is seen (CLAUDE.md section 6).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "configs" / "decision" / "ranking_policy_v1.yaml"
# Fixed order: the sensitivity weight lists in the config follow it.
COMPONENTS: tuple[str, ...] = (
    "source_relief",
    "destination_headroom",
    "imbalance",
    "response",
    "global_improvement",
    "migration_cost",
)
NORMALIZATION = "minmax_per_epoch"
TIE_BREAK = ("total_score_desc", "dst_upper_bound_asc", "candidate_id_asc")
WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Component:
    name: str
    weight: float
    enabled: bool
    reason: str = ""


@dataclass(frozen=True)
class RankingPolicy:
    version: str
    safety_policy_version: str
    components: tuple[Component, ...]
    require_positive: tuple[str, ...]
    sensitivity: tuple[tuple[str, tuple[float, ...]], ...]

    def component(self, name: str) -> Component:
        for item in self.components:
            if item.name == name:
                return item
        raise KeyError(name)

    @property
    def enabled(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components if c.enabled)

    def effective_weights(self) -> dict[str, float]:
        """Weights of the enabled components, renormalized to sum to 1."""
        total = sum(c.weight for c in self.components if c.enabled)
        return {c.name: c.weight / total for c in self.components if c.enabled}

    def with_weights(self, name: str, weights: Sequence[float]) -> RankingPolicy:
        """The same policy with another weight set (sensitivity); enabled flags stay."""
        values = _weights(weights, f"sensitivity.{name}")
        components = tuple(
            dataclasses.replace(c, weight=w) for c, w in zip(self.components, values)
        )
        if sum(c.weight for c in components if c.enabled) <= 0:
            raise ValueError(f"sensitivity.{name}: enabled weights sum to 0")
        return dataclasses.replace(self, version=f"{self.version}+{name}", components=components)


def _weights(values: Any, where: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(f"{where}: expected a list of {len(COMPONENTS)} weights")
    if len(values) != len(COMPONENTS):
        raise ValueError(f"{where}: expected {len(COMPONENTS)} weights, got {len(values)}")
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: weight {value!r} is not a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{where}: weight {value!r} must be finite and >= 0")
        out.append(float(value))
    if abs(sum(out) - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"{where}: weights sum to {sum(out)}, not 1")
    return tuple(out)


def _component(name: str, data: Any) -> Component:
    where = f"components.{name}"
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: expected a mapping")
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"{where}.enabled must be true or false")
    reason = str(data.get("reason") or "").strip()
    if not enabled and not reason:
        raise ValueError(f"{where}: a disabled component needs a 'reason'")
    return Component(name, float(data.get("weight", -1.0)), enabled, reason)


def parse_ranking_policy(data: Mapping[str, Any]) -> RankingPolicy:
    version = str(data.get("version") or "").strip()
    safety = str(data.get("safety_policy_version") or "").strip()
    if not version or not safety:
        raise ValueError("ranking policy needs 'version' and 'safety_policy_version'")
    if data.get("normalization") != NORMALIZATION:
        raise ValueError(f"normalization must be {NORMALIZATION!r}")
    if tuple(data.get("tie_break") or ()) != TIE_BREAK:
        raise ValueError(f"tie_break must be {list(TIE_BREAK)}")
    raw = data.get("components")
    if not isinstance(raw, Mapping) or set(raw) != set(COMPONENTS):
        raise ValueError(f"components must be exactly {list(COMPONENTS)}")
    components = tuple(_component(name, raw[name]) for name in COMPONENTS)
    _weights([c.weight for c in components], "components")
    if not any(c.enabled and c.weight > 0 for c in components):
        raise ValueError("at least one enabled component needs a weight > 0")
    for c in components:
        if c.enabled and c.weight <= 0:
            raise ValueError(f"components.{c.name}: an enabled component needs a weight > 0")
    no_migration = data.get("no_migration") or {}
    if not isinstance(no_migration, Mapping):
        raise ValueError("no_migration must be a mapping")
    positive = tuple(str(x) for x in no_migration.get("require_positive") or ())
    enabled = {c.name for c in components if c.enabled}
    unknown = sorted(set(positive) - enabled)
    if unknown:
        raise ValueError(f"no_migration.require_positive: not enabled components {unknown}")
    sets = data.get("sensitivity") or {}
    if not isinstance(sets, Mapping) or "base" not in sets:
        raise ValueError("sensitivity needs at least a 'base' weight set")
    sensitivity = tuple((str(k), _weights(v, f"sensitivity.{k}")) for k, v in sets.items())
    if dict(sensitivity)["base"] != tuple(c.weight for c in components):
        raise ValueError("sensitivity.base must equal the component weights")
    return RankingPolicy(version, safety, components, positive, sensitivity)


def load_ranking_policy(path: Path = POLICY_PATH) -> RankingPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return parse_ranking_policy(data)
