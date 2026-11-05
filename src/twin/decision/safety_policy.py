"""Safety Policy v1: the hard limits of the Safety Gate, each with its provenance.

Every limit is a number with the evidence it comes from, or null with the reason it is
unset; a null limit makes its gate check SKIPPED (NOT_CONFIGURED), never PASSED. The
margins m_U, m_R are read from the frozen outcome artifact, never copied by hand.
Boundary (CLAUDE.md section 1): value + margin <= limit passes.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.twin.forecasting.margin import finite_sample_quantile

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "configs" / "decision" / "safety_policy_v1.yaml"
EPSILON = 1e-9  # float error at the boundary, as in overload_detector
COOLDOWN_SCOPES = ("global",)


@dataclass(frozen=True)
class Limit:
    value: float | None
    source: str = ""
    reason: str = ""

    @property
    def configured(self) -> bool:
        return self.value is not None


def parse_limit(data: Any, where: str, minimum: float = 0.0) -> Limit:
    if not isinstance(data, Mapping) or "value" not in data:
        raise ValueError(f"{where}: needs 'value' plus 'source' (set) or 'reason' (null)")
    value = data["value"]
    source = str(data.get("source") or "").strip()
    reason = str(data.get("reason") or "").strip()
    if value is None:
        if not reason:
            raise ValueError(f"{where}: a null limit needs a 'reason'")
        return Limit(None, reason=reason)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where}: value must be a finite number or null, got {value!r}")
    if value < minimum:
        raise ValueError(f"{where}: value must be >= {minimum}, got {value}")
    if not source:
        raise ValueError(f"{where}: a set limit needs its 'source'")
    return Limit(float(value), source=source)


@dataclass(frozen=True)
class SafetyPolicy:
    version: str
    boundary: str
    require_valid_snapshot: bool
    minimum_quality_score: Limit
    require_destination_connected: bool
    reject_if_destination_at_risk: bool
    utilization_safe_limit: Limit
    response_p95_max_ms: Limit
    outcome_model_version: str
    margins_file: str
    max_concurrent_transactions: int
    cooldown_scope: str
    cooldown_seconds: Limit
    require_positive_source_relief: bool
    minimum_source_relief: Limit


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"safety policy: missing section '{key}'")
    return value


def _flag(section: Mapping[str, Any], key: str, where: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}.{key} must be true or false")
    return value


def parse_safety_policy(data: Mapping[str, Any]) -> SafetyPolicy:
    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError("safety policy: missing 'version'")
    if data.get("boundary") != "inclusive":
        raise ValueError("boundary must be 'inclusive' (value + margin <= limit passes)")
    state, dst = _section(data, "state"), _section(data, "destination")
    unc, mig = _section(data, "uncertainty"), _section(data, "migration")
    imp = _section(data, "improvement")
    if _flag(state, "require_valid_snapshot", "state") is not True:
        raise ValueError("state.require_valid_snapshot must be true: the Hard Gate is not optional")
    quality = parse_limit(state.get("minimum_quality_score"), "state.minimum_quality_score")
    if quality.value is not None and quality.value > 1.0:
        raise ValueError("state.minimum_quality_score must be <= 1")
    utilization = parse_limit(dst.get("utilization_safe_limit"),
                              "destination.utilization_safe_limit")
    if utilization.value is None or utilization.value <= 0:
        raise ValueError("destination.utilization_safe_limit must be set and > 0")
    concurrent = mig.get("max_concurrent_transactions")
    if concurrent != 1:
        raise ValueError("migration.max_concurrent_transactions must be 1 (one per cycle)")
    scope = mig.get("cooldown_scope")
    if scope not in COOLDOWN_SCOPES:
        raise ValueError(f"migration.cooldown_scope must be one of {COOLDOWN_SCOPES}")
    for key in ("outcome_model_version", "margins_file"):
        if not str(unc.get(key) or "").strip():
            raise ValueError(f"uncertainty.{key} is required")
    return SafetyPolicy(
        version=version,
        boundary="inclusive",
        require_valid_snapshot=True,
        minimum_quality_score=quality,
        require_destination_connected=_flag(dst, "require_connected", "destination"),
        reject_if_destination_at_risk=_flag(dst, "reject_if_forecast_risk", "destination"),
        utilization_safe_limit=utilization,
        response_p95_max_ms=parse_limit(dst.get("response_p95_max_ms"),
                                        "destination.response_p95_max_ms"),
        outcome_model_version=str(unc["outcome_model_version"]),
        margins_file=str(unc["margins_file"]),
        max_concurrent_transactions=1,
        cooldown_scope=str(scope),
        cooldown_seconds=parse_limit(mig.get("cooldown_seconds"), "migration.cooldown_seconds"),
        require_positive_source_relief=_flag(imp, "require_positive_source_relief",
                                             "improvement"),
        minimum_source_relief=parse_limit(imp.get("minimum_source_relief"),
                                          "improvement.minimum_source_relief"),
    )


def load_safety_policy(path: Path = POLICY_PATH) -> SafetyPolicy:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected a mapping")
    return parse_safety_policy(data)


def load_margins(policy: SafetyPolicy, root: Path = REPO_ROOT) -> tuple[float, float]:
    """(m_U, m_R) of the outcome artifact named by the policy."""
    data = json.loads((Path(root) / policy.margins_file).read_text(encoding="utf-8"))
    if data.get("model_version") != policy.outcome_model_version:
        raise ValueError(f"{policy.margins_file}: model_version {data.get('model_version')!r}"
                         f" is not {policy.outcome_model_version!r}")
    margins = []
    for key in ("m_U", "m_R"):
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{policy.margins_file}: {key} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{policy.margins_file}: {key} must be finite and >= 0")
        margins.append(float(value))
    return margins[0], margins[1]


def within_limit(value: float, margin: float, limit: float) -> bool:
    """Inclusive boundary: value + margin <= limit passes; float error is absorbed."""
    if not all(math.isfinite(x) for x in (value, margin, limit)):
        raise ValueError("value, margin and limit must be finite")
    if margin < 0:
        raise ValueError("margin must be >= 0")
    return value + margin <= limit + EPSILON


def cooldown_lower_bound(transaction_s: Sequence[float], settle_s: Sequence[float],
                         level: float = 0.95) -> float:
    """Finite-sample P95 of transaction + settle time of committed migrations, in s."""
    if len(transaction_s) != len(settle_s):
        raise ValueError("transaction_s and settle_s must have the same length")
    totals = [float(a) + float(b) for a, b in zip(transaction_s, settle_s)]
    if any(not math.isfinite(t) or t < 0 for t in totals):
        raise ValueError("durations must be finite and >= 0")
    return finite_sample_quantile(totals, level)
