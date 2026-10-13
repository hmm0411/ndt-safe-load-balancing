# Experiment Tracking with MLflow

`scripts/experiments/train_forecast_baselines.py` logs every training run to
MLflow. Files under `data/models/forecasting/<model_version>/` remain the
authoritative record; MLflow adds run history, side-by-side comparison, and an
optional model registry.

## Configuration

All settings are in `configs/forecasting/models.yaml`:

| Section | Purpose |
|---|---|
| `mlflow.enabled` | turn tracking on/off (`--no-mlflow` also disables it) |
| `mlflow.tracking_uri` | `sqlite:///mlflow.db` (file at repo root) or `http://host:port` |
| `mlflow.experiment` | experiment name, default `ndt-forecasting` |
| `mlflow.register_models` | register the selected RF/XGB models (or pass `--register`) |
| `moving_average.windows` | candidate MA windows |
| `random_forest.fixed` / `.grid` | fixed params and the search grid |
| `xgboost.fixed` / `.grid` | fixed params and the search grid |

Every grid combination is trained and scored on the **validation** split; the
lowest validation RMSE wins. Test and unseen (hot-switch) metrics are computed
once, after selection, so changing grids never uses test data for tuning.

## Running

```bash
pip install -r requirements-ml.txt
export PYTHONPATH=$PWD

python3 scripts/experiments/train_forecast_baselines.py \
  --run-name rf-xgb-h3-grid1 \
  --note "first grid on the 25 official runs"

# options
#   --config <yaml>   alternative settings file
#   --register        register selected models in the Model Registry
#   --no-mlflow       local artifacts only
```

## What is logged

One **parent run** per training invocation:

- tags: `git_commit`, `git_branch`, `git_tag`, `git_dirty`, `dataset_version`,
  `model_version`, `model_config`, `role=training`, the `--note` text;
- params: horizon, feature count, split sizes, selected MA window, selected and
  fixed RF/XGB hyperparameters;
- metrics: `validation_<model>_*`, `test_<model>_*`, `unseen_<model>_*`
  (mae, rmse, mape, r2, p95_absolute_error), per-workload
  `<split>_<workload>_<model>_{rmse,mae}`, skill vs persistence
  `skill_<split>_<workload>_<model>`, `xgboost_best_iteration`;
- models: `random_forest`, `xgboost`, `linear_regression` (scaler + model
  pipeline), each with an input example;
- artifacts: the whole local model directory (manifests, metrics JSON,
  feature importance, hyperparameter search, feature shift report).

One **child run** per grid candidate (`random_forest #k`, `xgboost #k`) with its
params and `validation_*` metrics, so candidates can be sorted or plotted
(e.g. `max_depth` vs `validation_rmse`) in the UI.

Note: the logged Random Forest / XGBoost models predict the **change**
`U(t+h) - U(t)`; the forecast is `U(t) + model.predict(X)`.

## Viewing the UI on the GCP VM

Start the UI bound to localhost only (do not open port 5000 in the firewall):

```bash
cd ~/ndt-safe-load-balancing
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

From your own computer, open an SSH tunnel and browse `http://localhost:5000`:

```bash
gcloud compute ssh kltn-sdn --zone asia-southeast1-b -- -N -L 5000:localhost:5000
```

## Querying runs from Python

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(
    experiment_names=["ndt-forecasting"],
    filter_string="tags.role = 'training'",
    order_by=["metrics.test_xgboost_rmse ASC"],
)
print(runs[["tags.mlflow.runName", "metrics.test_persistence_rmse",
            "metrics.test_random_forest_rmse", "metrics.test_xgboost_rmse"]])
```

`mlflow.db` and `mlruns/` are git-ignored. To share results, commit the local
model directory, or export them with `mlflow.search_runs(...).to_csv(...)`.
