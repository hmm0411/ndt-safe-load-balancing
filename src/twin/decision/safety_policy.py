from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_POLICY_PATH = Path("configs/decision/safety_policy_v1.yaml")

_ALLOWED: dict[str, set[str]] = {
    "state": {"require_valid_snapshot", "minimum_quality_score"},
    "destination": {
        "require_connected",
        "reject_if_forecast_risk",
        "utilization_safe_limit",
        "response_p95_max_ms",
    },
    "uncertainty": {"utilization_margin_source", "response_margin_source"},
    "migration": {"require_transaction_slot", "cooldown_seconds"},
    "improvement": {"require_positive_source_relief", "minimum_source_relief"},
}


class PolicyError(ValueError):
    """Raised when a safety policy file is malformed."""


@dataclass(frozen=True)
class SafetyPolicy:
    version: str
    require_valid_snapshot: bool
    minimum_quality_score: float | None
    require_connected: bool
    reject_if_forecast_risk: bool
    utilization_safe_limit: float
    response_p95_max_ms: float | None
    utilization_margin_source: str
    response_margin_source: str
    require_transaction_slot: bool
    cooldown_seconds: float | None
    require_positive_source_relief: bool
    minimum_source_relief: float | None

    def not_configured(self) -> tuple[str, ...]:
        """Names of optional thresholds that are still null."""
        names = (
            "minimum_quality_score",
            "response_p95_max_ms",
            "cooldown_seconds",
            "minimum_source_relief",
        )
        return tuple(n for n in names if getattr(self, n) is None)


def _num(value: Any, name: str, *, nullable: bool) -> float | None:
    if value is None:
        if nullable:
            return None
        raise PolicyError(f"{name} must not be null")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"{name} must be a number, got {value!r}")
    return float(value)


def _flag(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{name} must be true or false, got {value!r}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{name} must be a non-empty string")
    return value


def parse_policy(raw: Any) -> SafetyPolicy:
    if not isinstance(raw, dict):
        raise PolicyError("policy must be a mapping")
    unknown = set(raw) - set(_ALLOWED) - {"version"}
    if unknown:
        raise PolicyError(f"unknown sections: {sorted(unknown)}")
    sec: dict[str, dict[str, Any]] = {}
    for name, allowed in _ALLOWED.items():
        body = raw.get(name)
        if not isinstance(body, dict):
            raise PolicyError(f"section {name} is missing")
        if set(body) != allowed:
            raise PolicyError(f"section {name} keys must be {sorted(allowed)}")
        sec[name] = body
    s, d, u, m, i = (sec[k] for k in _ALLOWED)

    def flag(body: dict[str, Any], key: str) -> bool:
        return _flag(body[key], key)

    def opt(body: dict[str, Any], key: str) -> float | None:
        return _num(body[key], key, nullable=True)

    return SafetyPolicy(
        version=_text(raw.get("version"), "version"),
        require_valid_snapshot=flag(s, "require_valid_snapshot"),
        minimum_quality_score=opt(s, "minimum_quality_score"),
        require_connected=flag(d, "require_connected"),
        reject_if_forecast_risk=flag(d, "reject_if_forecast_risk"),
        utilization_safe_limit=_num(
            d["utilization_safe_limit"], "utilization_safe_limit", nullable=False
        ),
        response_p95_max_ms=opt(d, "response_p95_max_ms"),
        utilization_margin_source=_text(
            u["utilization_margin_source"], "utilization_margin_source"
        ),
        response_margin_source=_text(u["response_margin_source"], "response_margin_source"),
        require_transaction_slot=flag(m, "require_transaction_slot"),
        cooldown_seconds=opt(m, "cooldown_seconds"),
        require_positive_source_relief=flag(i, "require_positive_source_relief"),
        minimum_source_relief=opt(i, "minimum_source_relief"),
    )


def load_safety_policy(path: str | Path = DEFAULT_POLICY_PATH) -> SafetyPolicy:
    with open(path, encoding="utf-8") as handle:
        return parse_policy(yaml.safe_load(handle))
