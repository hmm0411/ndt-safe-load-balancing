"""Decision methods of the campaign: Full, its ablations and the baselines (protocol 3-4).

A method is data (configs/decision/methods.yaml): what triggers a decision, how the action
is chosen, which outcome predictions and margins the Safety Gate sees, how safe candidates
are ranked, and the cooldown. DecisionService reads one MethodSpec; everything else (the
executor, one migration at a time, telemetry, records) is shared, so the methods differ
only where the protocol says they do.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "decision" / "methods.yaml"
DEFAULT_METHOD = "ndt_no_feedback"

TRIGGERS = ("none", "measured", "forecast")
SELECTORS = ("heuristic", "pipeline")
OUTCOMES = ("model", "analytical")
MARGINS = ("fixed", "adaptive", "zero")
RANKINGS = ("policy", "random")


@dataclass(frozen=True)
class FeedbackConfig:
    alpha: float
    gamma_forecast: float
    gamma_outcome: float


@dataclass(frozen=True)
class MethodSpec:
    name: str
    trigger: str
    selector: str
    outcome: str
    margins: str
    ranking: str
    cooldown_seconds: float | None
    trigger_utilization: float
    room_limit: float
    feedback: FeedbackConfig
    version: str

    def __post_init__(self) -> None:
        for value, allowed, field_name in (
                (self.trigger, TRIGGERS, "trigger"), (self.selector, SELECTORS, "selector"),
                (self.outcome, OUTCOMES, "outcome"), (self.margins, MARGINS, "margins"),
                (self.ranking, RANKINGS, "ranking")):
            if value not in allowed:
                raise ValueError(f"method {self.name}: {field_name} {value!r} not in {allowed}")
        if self.selector == "heuristic" and (self.outcome, self.margins, self.ranking) != (
                "model", "fixed", "policy"):
            raise ValueError(f"method {self.name}: a heuristic method uses no outcome, margin "
                             "or ranking option; keep the defaults")
        if self.cooldown_seconds is not None and self.cooldown_seconds < 0:
            raise ValueError(f"method {self.name}: cooldown_seconds must be >= 0")
        if not 0 < self.trigger_utilization <= 2 or not 0 < self.room_limit <= 2:
            raise ValueError(f"method {self.name}: trigger_utilization / room_limit out of range")

    @property
    def migrates(self) -> bool:
        return self.trigger != "none"

    @property
    def uses_forecast(self) -> bool:
        return self.trigger == "forecast"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_methods(data: Mapping[str, Any]) -> dict[str, MethodSpec]:
    version = str(data.get("version") or "").strip()
    if not version:
        raise ValueError("methods config needs a version")
    fb = data.get("feedback") or {}
    feedback = FeedbackConfig(float(fb["alpha"]), float(fb["gamma_forecast"]),
                              float(fb["gamma_outcome"]))
    defaults = data.get("defaults") or {}
    out = {}
    for name, spec in (data.get("methods") or {}).items():
        cooldown = spec.get("cooldown_seconds")
        out[str(name)] = MethodSpec(
            name=str(name), trigger=str(spec["trigger"]), selector=str(spec["selector"]),
            outcome=str(spec["outcome"]), margins=str(spec["margins"]),
            ranking=str(spec["ranking"]),
            cooldown_seconds=None if cooldown is None else float(cooldown),
            trigger_utilization=float(spec.get("trigger_utilization",
                                               defaults["trigger_utilization"])),
            room_limit=float(spec.get("room_limit", defaults["room_limit"])),
            feedback=feedback, version=version)
    if DEFAULT_METHOD not in out:
        raise ValueError(f"methods config must define {DEFAULT_METHOD}")
    return out


def load_method(name: str | None = None, path: Path = CONFIG_PATH) -> MethodSpec:
    """The method named by ``name``, else NDT_DECISION_METHOD, else ndt_no_feedback."""
    methods = parse_methods(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    chosen = name or os.environ.get("NDT_DECISION_METHOD") or DEFAULT_METHOD
    if chosen not in methods:
        raise ValueError(f"unknown decision method {chosen!r}; one of {sorted(methods)}")
    return methods[chosen]
