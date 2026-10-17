"""Twin Quality Score (Week 10, V1): soft quality in [0, 1].

The Hard Gate (TwinQuality.valid = fresh and complete and consistent and
twinning rate) still decides whether a snapshot may be used; this score only
grades HOW good a valid snapshot is (and how bad an invalid one is).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import yaml

COMPONENTS = ("consistency", "completeness", "freshness", "jitter", "collection", "twinning")


def piecewise(value: float, good: float, hard: float) -> float:
    """1.0 at or below `good`, 0.0 at or above `hard`, linear in between."""
    if not math.isfinite(value):
        return 0.0
    if hard <= good:
        raise ValueError("hard must be greater than good")
    if value <= good:
        return 1.0
    if value >= hard:
        return 0.0
    return (hard - value) / (hard - good)


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if set(config.get("weights", {})) != set(COMPONENTS):
        raise ValueError(f"quality weights must be exactly {COMPONENTS}")
    weights = {name: float(config["weights"][name]) for name in COMPONENTS}
    if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("quality weights must be >= 0 and sum to 1")
    thresholds = {
        key: {"good": float(v["good"]), "hard": float(v["hard"])}
        for key, v in config["thresholds"].items()
    }
    for key in ("age_ms", "jitter_ms", "collection_ms"):
        if thresholds[key]["hard"] <= thresholds[key]["good"]:
            raise ValueError(f"{key}: hard must be greater than good")
    return {"version": str(config.get("version", "")), "weights": weights,
            "thresholds": thresholds}


def load_quality_config(path: Path) -> dict[str, Any]:
    return validate_config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def quality_components(
    *,
    consistency_score: float,
    completeness_ratio: float,
    age_ms: float,
    jitter_ms: float,
    collection_ms: float,
    twinning_rate: float,
    thresholds: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    t = thresholds

    def clip(value: float) -> float:
        return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0

    return {
        "consistency": clip(consistency_score),
        "completeness": clip(completeness_ratio),
        "freshness": piecewise(age_ms, t["age_ms"]["good"], t["age_ms"]["hard"]),
        "jitter": piecewise(jitter_ms, t["jitter_ms"]["good"], t["jitter_ms"]["hard"]),
        "collection": piecewise(collection_ms, t["collection_ms"]["good"],
                                t["collection_ms"]["hard"]),
        "twinning": clip(twinning_rate),
    }


def quality_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    return round(sum(weights[name] * components[name] for name in COMPONENTS), 6)
