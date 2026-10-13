"""MLflow tracking for forecast training, with a no-op mode.

Local files under data/models/forecasting/<version>/ stay the authoritative
record; MLflow adds a searchable history, run comparison and a model registry.

Layout of one training run:

    parent run   <model_version>-<dataset_version>   (tags, params, datasets, all
    |                                                  metrics, live grid progress,
    |                                                  system metrics, tables,
    |                                                  figures, models, artifacts)
    +-- child    random_forest #k                      (one per grid candidate:
    +-- child    xgboost #k                             params, validation metrics,
                                                        XGBoost learning curve)
"""
from __future__ import annotations

import inspect
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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
        system_metrics: bool = False,
        system_metrics_interval: float = 5.0,
        candidate_alias: str = "challenger",
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.register_models = register_models
        self.registered_names = dict(registered_names or {})
        self.system_metrics = system_metrics
        self.system_metrics_interval = system_metrics_interval
        self.candidate_alias = candidate_alias
        self.run_id: str | None = None
        self.registered_versions: dict[str, str] = {}
        self._mlflow: Any = None
        self._candidate_counter: dict[str, int] = {}
        self._best_so_far: dict[str, float] = {}

    # -- run lifecycle ---------------------------------------------------

    def start(self, tags: Mapping[str, str]) -> None:
        if not self.enabled:
            return
        import mlflow

        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment)
        kwargs: dict[str, Any] = {"run_name": self.run_name}
        if self.system_metrics:
            # CPU / RAM / disk / network of the training process, sampled
            # in the background (needs psutil) -> "System metrics" tab.
            mlflow.set_system_metrics_sampling_interval(self.system_metrics_interval)
            kwargs["log_system_metrics"] = True
        run = mlflow.start_run(**kwargs)
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
        curve: Sequence[float] | None = None,
    ) -> None:
        """One grid candidate = one child run (compare them in the MLflow UI).

        The parent run also receives step metrics
        ``grid_<model>_validation_rmse`` and ``grid_<model>_best_validation_rmse``
        (step = candidate index), so the grid search can be watched live.
        """
        if self._mlflow is None:
            return
        index = self._candidate_counter.get(model_name, 0) + 1
        self._candidate_counter[model_name] = index
        with self._mlflow.start_run(run_name=f"{model_name} #{index}", nested=True) as child:
            self._mlflow.set_tags({"model": model_name, "role": "grid_candidate"})
            self._mlflow.log_params(_flat(params))
            self._mlflow.log_metrics(
                {
                    **{f"validation_{k}": v for k, v in _finite(metrics).items()},
                    **_finite(extra),
                }
            )
            if curve:
                self._log_series(child.info.run_id, "learning_curve_validation_rmse", curve)

        rmse = _finite(metrics).get("rmse")
        if rmse is not None and self.run_id is not None:
            best = min(rmse, self._best_so_far.get(model_name, math.inf))
            self._best_so_far[model_name] = best
            client = self._mlflow.MlflowClient()
            client.log_metric(self.run_id, f"grid_{model_name}_validation_rmse", rmse, step=index)
            client.log_metric(self.run_id, f"grid_{model_name}_best_validation_rmse", best, step=index)

    def log_curve(self, key: str, values: Sequence[float]) -> None:
        """A per-step series on the parent run (e.g. the selected XGBoost curve)."""
        if self._mlflow is not None and self.run_id is not None and values:
            self._log_series(self.run_id, key, values)

    def _log_series(self, run_id: str, key: str, values: Sequence[float]) -> None:
        from mlflow.entities import Metric

        now = int(time.time() * 1000)
        points = [
            Metric(key, float(v), now, step)
            for step, v in enumerate(values)
            if math.isfinite(float(v))
        ]
        client = self._mlflow.MlflowClient()
        for start in range(0, len(points), 1000):  # log_batch limit
            client.log_batch(run_id, metrics=points[start : start + 1000])

    def log_dataset(
        self,
        frame: Any,
        name: str,
        context: str,
        source: str,
        targets: str | None = None,
    ) -> None:
        """Record a split as a dataset input (name, digest, schema, row count)."""
        if self._mlflow is None:
            return
        dataset = self._mlflow.data.from_pandas(
            frame, source=source, name=name, targets=targets
        )
        self._mlflow.log_input(dataset, context=context)

    def log_table(self, rows: Any, artifact_file: str) -> None:
        """A table artifact (dict of columns, DataFrame, or list of row dicts)."""
        if self._mlflow is None:
            return
        if isinstance(rows, list):
            import pandas as pd

            rows = pd.DataFrame(rows)
        self._mlflow.log_table(data=rows, artifact_file=artifact_file)

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        if self._mlflow is not None:
            self._mlflow.log_figure(figure, artifact_file)

    def write_pointer(self, path: Path) -> None:
        """Let later scripts (plots) attach artifacts to this run."""
        if self.run_id is None:
            return
        path.write_text(
            json.dumps(
                {
                    "tracking_uri": self.tracking_uri,
                    "experiment": self.experiment,
                    "run_id": self.run_id,
                    "registered_versions": self.registered_versions,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
        version_tags: Mapping[str, Any] | None = None,
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
        if "registered_model_name" in kwargs:
            self._tag_new_version(kwargs["registered_model_name"], name, version_tags)

    def _tag_new_version(
        self,
        registered_name: str,
        name: str,
        version_tags: Mapping[str, Any] | None,
    ) -> None:
        """Point the ``challenger`` alias at the version this run just created.

        Promotion to ``champion`` (the version the runtime predictor loads via
        ``models:/<name>@champion``) is a deliberate human step in the UI.
        """
        client = self._mlflow.MlflowClient()
        versions = client.search_model_versions(
            f"name='{registered_name}' and run_id='{self.run_id}'"
        )
        if not versions:
            return
        version = max(versions, key=lambda v: int(v.version)).version
        self.registered_versions[name] = str(version)
        if self.candidate_alias:
            client.set_registered_model_alias(registered_name, self.candidate_alias, version)
        for key, value in (version_tags or {}).items():
            client.set_model_version_tag(registered_name, version, key, str(value))
