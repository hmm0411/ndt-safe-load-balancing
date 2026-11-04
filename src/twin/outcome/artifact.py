"""The post-migration-v1 artifact: configuration and file helpers (Week 12, Day 5).

Layout (plan section 46), read at runtime by ``src/twin/outcome/predictor.py``:

    analytical_config.json  direct_model/  hybrid_model/  residual_models/
    feature_schema.json  target_schema.json  hyperparameters.json
    validation_metrics.json  test_metrics.json  per_workload_metrics.json
    uncertainty_margin.json  inference_latency.json  training_manifest.json  model_card.json

``inference_latency.json`` is measured on the VM by the runtime side (Vy, Day 5) once the
rest of the artifact exists; until then the predictor answers ARTIFACT_SCHEMA_MISMATCH.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from src.twin.outcome.targets import ALL_TARGETS

LATENCY_FILE = "inference_latency.json"
MANIFEST_FILE = "training_manifest.json"


@dataclass(frozen=True)
class ArtifactConfig:
    model_version: str
    artifact_dir: Path
    deploy_rule: str
    margin_level: float
    margins: dict[str, str]
    coverage_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.deploy_rule != "lowest_validation_S":
            raise ValueError("deploy_rule must be lowest_validation_S")
        if not 0.5 < self.margin_level < 1.0:
            raise ValueError("margin_level must be in (0.5, 1)")
        if set(self.margins) != {"m_U", "m_R"}:
            raise ValueError("margins must define m_U and m_R")
        bad = [t for t in self.margins.values() if t not in ALL_TARGETS]
        if bad:
            raise ValueError(f"margin targets must be outcome targets, got {bad}")


def load_artifact_config(path: Path) -> ArtifactConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return ArtifactConfig(
            model_version=str(data["model_version"]),
            artifact_dir=Path(str(data["artifact_dir"])),
            deploy_rule=str(data["deploy_rule"]),
            margin_level=float(data["margin_level"]),
            margins={str(k): str(v) for k, v in dict(data["margins"]).items()},
            coverage_groups=tuple(str(g) for g in data.get("coverage_groups") or ()),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path}: missing or malformed key {exc}") from exc


def capacity_version(path: Path) -> str:
    """Same format as the dataset manifest and the orchestrator: <file>@sha256:<12 hex>."""
    return f"{path.name}@sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, data: Any) -> None:
    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): walk(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [walk(v) for v in obj]
        return _clean(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(walk(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, Any]]:
    """CSV rows with numbers parsed (empty cells stay '')."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value) if value not in ("", "True", "False") else value
            except ValueError:
                parsed[key] = value
        out.append(parsed)
    return out


def metrics_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """model -> target -> metrics, from validation_metrics.csv / test_metrics.csv rows."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r["model"]), {})[str(r["target"])] = {
            k: r[k] for k in ("n", "mae", "rmse", "p95_abs_error", "bias", "r2") if k in r}
    return out


def files_sha256(root: Path, skip: tuple[str, ...] = (MANIFEST_FILE,)) -> dict[str, str]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel not in skip:
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def write_manifest_files(root: Path, manifest: Mapping[str, Any]) -> None:
    write_json(root / MANIFEST_FILE, {**dict(manifest), "files_sha256": files_sha256(root)})
