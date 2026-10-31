"""The frozen Day 4 selection, read back for margins, the artifact and the figures (Day 5-6).

``train_outcome_models.py --stage select-and-test`` wrote ``model_selection.json``
(frozen choice + sha256 of every model file and config) and completed ``test_lock.json``.
Nothing after Day 4 may choose anything again: this module only re-reads that choice,
checks that every file it names is byte-identical, and rebuilds the predictions of
Analytical, Best Direct and Best Hybrid from the saved models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.twin.outcome import hybrid, residual, selection, trainer
from src.twin.outcome.targets import ALL_TARGETS, BASE_TARGETS

ROLES = ("analytical", "best_direct", "best_hybrid")


@dataclass(frozen=True)
class FinalPaths:
    dataset: Path
    config: Path
    models_config: Path
    eval_config: Path
    direct_dir: Path
    hybrid_dir: Path
    final_dir: Path

    def frozen_files(self, names: Sequence[str]) -> dict[str, Path]:
        """Paths of the names recorded by Day 4 in ``frozen.sha256``."""
        fixed = {
            "outcome_dataset.csv": self.dataset / "outcome_dataset.csv",
            "feature_schema.json": self.dataset / "feature_schema.json",
            "configs/dataset.yaml": self.config,
            "configs/models.yaml": self.models_config,
            "configs/evaluation.yaml": self.eval_config,
            "direct/hyperparameters.json": self.direct_dir / "hyperparameters.json",
            "hybrid/hyperparameters.json": self.hybrid_dir / "hyperparameters.json",
        }
        out = {}
        for name in names:
            if name in fixed:
                out[name] = fixed[name]
            elif name.startswith("models/direct_"):
                out[name] = self.direct_dir / name
            elif name.startswith("models/residual_"):
                out[name] = self.hybrid_dir / name
            else:
                raise ValueError(f"model_selection.json names an unknown file {name!r}")
        return out


@dataclass(frozen=True)
class Frozen:
    selection: dict[str, Any]
    lock: dict[str, Any]

    @property
    def frozen(self) -> dict[str, Any]:
        return dict(self.selection["frozen"])

    @property
    def best_direct(self) -> str:
        return str(self.frozen["best_direct"])

    @property
    def best_hybrid(self) -> str:
        return str(self.frozen["best_hybrid"])

    def name_of(self, role: str) -> str:
        return {"analytical": "analytical", "best_direct": self.best_direct,
                "best_hybrid": self.best_hybrid}[role]

    def validation_s(self, name: str) -> float:
        return float(self.frozen["validation_scores"][name]["S"])


def load_frozen(paths: FinalPaths) -> Frozen:
    """The Day 4 choice, after checking it is complete and that no file has changed."""
    sel_path = paths.final_dir / selection.SELECTION_FILE
    if not sel_path.is_file():
        raise FileNotFoundError(f"{sel_path} not found: run --stage select-and-test first")
    chosen = json.loads(sel_path.read_text(encoding="utf-8"))
    lock = selection.read_lock(paths.final_dir)
    if lock is None or lock.get("status") != "completed":
        raise ValueError(f"{paths.final_dir / selection.LOCK_FILE}: the Day 4 test run did not "
                         "complete; finish --stage select-and-test first")
    sha = selection.canonical_sha256(chosen["frozen"])
    if sha != chosen.get("frozen_sha256") or sha != lock.get("selection_sha256"):
        raise ValueError("model_selection.json was edited after the test was opened")
    recorded = dict(chosen["frozen"]["sha256"])
    current = selection.tree_sha256(paths.frozen_files(list(recorded)))
    changed = sorted(n for n in recorded if recorded[n] != current[n])
    if changed:
        raise ValueError(f"files changed since Day 4 froze the selection: {changed}")
    return Frozen(selection=chosen, lock=lock)


def deployed_model(frozen: Frozen) -> str:
    """Rule fixed in the plan before the test (lowest validation S among Analytical,
    Best Direct and Best Hybrid); it reads validation scores only."""
    names = [frozen.name_of(role) for role in ROLES]
    return min(names, key=frozen.validation_s)


class FrozenModels:
    """Predictions of the three roles from the saved Day 2 / Day 3 models."""

    def __init__(self, paths: FinalPaths, frozen: Frozen) -> None:
        schema = json.loads((paths.dataset / "feature_schema.json").read_text(encoding="utf-8"))
        self.model_features = [str(f) for f in schema["model_features"]]
        self.hybrid_features = residual.hybrid_features(self.model_features)
        self.frozen = frozen
        self.models: dict[str, dict[str, Any]] = {}
        for name in (frozen.best_direct, frozen.best_hybrid):
            family = family_of(name)
            if name.startswith("direct_"):
                self.models[name] = trainer.load_models(paths.direct_dir / "models", family,
                                                        BASE_TARGETS, prefix="direct")
            else:
                self.models[name] = hybrid.load(paths.hybrid_dir / "models", family)

    def predict(self, name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
        if name == "analytical":
            return {t: [float(r[f"analytical_{t}"]) for r in rows] for t in ALL_TARGETS}
        if name.startswith("direct_"):
            return trainer.predict_targets(self.models[name], rows, self.model_features)
        return hybrid.predict(self.models[name], rows, self.hybrid_features)

    def predict_roles(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
        """Model name -> target -> values, for Analytical, Best Direct, Best Hybrid."""
        return {self.frozen.name_of(role): self.predict(self.frozen.name_of(role), rows)
                for role in ROLES}


def family_of(name: str) -> str:
    if name.endswith("_rf"):
        return "random_forest"
    if name.endswith("_xgb"):
        return "xgboost"
    raise ValueError(f"{name} is not a learned model")
