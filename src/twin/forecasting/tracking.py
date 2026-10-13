"""MLflow tracking for forecast training, with a no-op mode.

Local files under data/models/forecasting/<version>/ stay the authoritative
record; MLflow adds a searchable history, run comparison and a model registry.

Layout of one training run:

    parent run   <model_version>-<dataset_version>   (tags, params, all metrics,
    |                                                  models, local artifacts)
    +-- child    random_forest #k                      (one per grid candidate:
    +-- child    xgboost #k                             params + validation metrics)
"""
from __future__ import annotations

import inspect
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "none"


def _code_changed() -> bool:
    """Uncommitted changes to tracked files, ignoring MLflow's own store.

    mlflow.db / mlruns/ are tracked in this repo and are rewritten by every
    run, so counting them would mark every run dirty.
    """
    status = _git(
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--",
        ".",
        ":(exclude)mlflow.db",
        ":(exclude)mlruns",
    )
    return status not in ("", "none", "unknown")


def git_tags() -> dict[str, str]:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_tag": _git("tag", "--points-at", "HEAD").replace("\n", ","),
        "git_dirty": "true" if _code_changed() else "false",
    }


def _flat(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """MLflow params are strings; None becomes 'None' rather than vanishing."""
    return {f"{prefix}{key}": str(value) for key, value in params.items()}


def _finite(metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                out[key] = number
    return out


class Tracker:
    """Thin wrapper so the training script reads the same with MLflow on or off."""

    def __init__(
        self,
        *,
        enabled: bool,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment: str = "ndt-forecasting",
        run_name: str | None = None,
        register_models: bool = False,
        registered_names: Mapping[str, str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.register_models = register_models
        self.registered_names = dict(registered_names or {})
        self.run_id: str | None = None
        self._mlflow: Any = None
        self._candidate_counter: dict[str, int] = {}

    # -- run lifecycle ---------------------------------------------------

    def start(self, tags: Mapping[str, str]) -> None:
        if not self.enabled:
            return
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        run = mlflow.start_run(run_name=self.run_name)
        self.run_id = run.info.run_id
        mlflow.set_tags({**git_tags(), **tags})

    def end(self, status: str = "FINISHED") -> None:
        if self._mlflow is not None and self._mlflow.active_run() is not None:
            self._mlflow.end_run(status=status)

    # -- logging -----------------------------------------------------------

    def log_params(self, params: Mapping[str, Any], prefix: str = "") -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(_flat(params, prefix))

    def log_metrics(self, metrics: Mapping[str, Any], prefix: str = "") -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(
                {f"{prefix}{key}": value for key, value in _finite(metrics).items()}
            )

    def log_candidate(
        self,
        model_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, Any],
        extra: Mapping[str, Any],
    ) -> None:
        """One grid candidate = one child run (compare them in the MLflow UI)."""
        if self._mlflow is None:
            return
        index = self._candidate_counter.get(model_name, 0) + 1
        self._candidate_counter[model_name] = index
        with self._mlflow.start_run(run_name=f"{model_name} #{index}", nested=True):
            self._mlflow.set_tags({"model": model_name, "role": "grid_candidate"})
            self._mlflow.log_params(_flat(params))
            self._mlflow.log_metrics(
                {
                    **{f"validation_{k}": v for k, v in _finite(metrics).items()},
                    **_finite(extra),
                }
            )

    def log_artifacts(self, directory: Path, artifact_path: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_artifacts(str(directory), artifact_path=artifact_path)

    def log_model(
        self,
        flavor: str,
        model: Any,
        name: str,
        input_example: Any,
        trusted_types: list[str] | None = None,
    ) -> None:
        """Log (and optionally register) a fitted model on MLflow 2.x or 3.x.

        Keyword arguments are chosen from the installed log_model signature:
        MLflow 3 uses ``name=`` and saves sklearn models with skops, which
        refuses tree internals unless they are listed in
        ``skops_trusted_types``; MLflow 2.x uses ``artifact_path=`` + pickle.
        """
        if self._mlflow is None:
            return
        module = getattr(self._mlflow, flavor)  # mlflow.sklearn / mlflow.xgboost
        accepted = inspect.signature(module.log_model).parameters
        kwargs: dict[str, Any] = {"input_example": input_example}
        kwargs["name" if "name" in accepted else "artifact_path"] = name
        if self.register_models and name in self.registered_names:
            kwargs["registered_model_name"] = self.registered_names[name]
        if trusted_types and "skops_trusted_types" in accepted:
            kwargs["skops_trusted_types"] = trusted_types
        module.log_model(model, **kwargs)
