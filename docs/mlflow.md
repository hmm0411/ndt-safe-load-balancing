# Experiment Tracking with MLflow

`scripts/experiments/train_forecast_baselines.py` logs every training run to
MLflow. Files under `data/models/forecasting/<model_version>/` remain the
authoritative record; MLflow adds run history, live progress, side-by-side
comparison, dataset lineage and a model registry.

## 1. Install

```bash
pip install -r requirements-ml.txt      # mlflow[auth], psutil, xgboost, ...
```

## 2. Tracking server reachable from several machines

Training writes straight into `sqlite:///mlflow.db` (repo root), so it never
depends on the server. The server only serves the UI/API over that store,
with a login.

### 2.1 Admin password (once)

```bash
mkdir -p ~/.config && umask 077
echo 'MLFLOW_AUTH_ADMIN_PASSWORD=<choose-a-long-password>' > ~/.config/ndt-mlflow.env
```

The password is only used the first time, to create user `admin` in
`basic_auth.db`. `scripts/mlflow/serve.sh` also creates a CSRF secret in
`.mlflow_flask_secret`. Both files are git-ignored.

### 2.2 Run it as a service (survives logout and reboots)

```bash
sudo cp deploy/mlflow/mlflow.service /etc/systemd/system/mlflow.service
# edit User/paths inside if your account or clone path differ
sudo systemctl daemon-reload
sudo systemctl enable --now mlflow
systemctl status mlflow            # journalctl -u mlflow -f for logs
```

Foreground alternative: `set -a; . ~/.config/ndt-mlflow.env; set +a; scripts/mlflow/serve.sh`.

### 2.3 Open port 5000 for your own IPs only (GCP firewall)

The UI uses HTTP basic auth over plain HTTP, so never open it to `0.0.0.0/0`.

```bash
# your current public IP: curl -s ifconfig.me
gcloud compute firewall-rules create allow-mlflow-5000 \
  --network default --direction INGRESS --action ALLOW --rules tcp:5000 \
  --source-ranges <IP_1>/32,<IP_2>/32 --target-tags mlflow
gcloud compute instances add-tags kltn-sdn --zone asia-southeast1-b --tags mlflow
```

Open `http://<VM_EXTERNAL_IP>:5000` and log in. When an IP changes, update
the rule with `gcloud compute firewall-rules update allow-mlflow-5000 --source-ranges ...`.

### 2.4 Accounts for teammates

```bash
python3 - <<'PY'
from mlflow.server.auth.client import AuthServiceClient
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<admin password>"
c = AuthServiceClient("http://127.0.0.1:5000")
c.create_user("my", "<password for My>")
PY
```

Every logged-in user can read every experiment (`default_permission = READ`
in `deploy/mlflow/basic_auth.ini`).

### 2.5 No firewall change at all (alternatives)

- **SSH tunnel** (one machine, needs gcloud):
  `gcloud compute ssh kltn-sdn --zone asia-southeast1-b -- -N -L 5000:localhost:5000`,
  then `http://localhost:5000`.
- **Cloud Shell Web Preview**: run the tunnel above inside Cloud Shell, then
  Web Preview → Change port → 5000.
- **Tailscale** (private network, any number of devices): install on the VM
  and on each laptop, then open `http://kltn-sdn:5000`.

## 3. Configuration (`configs/forecasting/models.yaml`)

| Key | Purpose |
|---|---|
| `mlflow.enabled` | tracking on/off (`--no-mlflow` also disables it) |
| `mlflow.tracking_uri` | `sqlite:///mlflow.db` (default) or `http://host:5000` |
| `mlflow.experiment` | experiment name, default `ndt-forecasting` |
| `mlflow.system_metrics` | CPU/RAM/disk/network of the training process |
| `mlflow.register_models` | register selected RF/XGB (or `--register`) |
| `mlflow.candidate_alias` | alias set on each new version (`challenger`) |
| `moving_average.windows` | candidate MA windows |
| `random_forest.fixed` / `.grid` | fixed params and search grid |
| `xgboost.fixed` / `.grid` | fixed params and search grid |

Every grid combination is scored on **validation**; the lowest RMSE wins.
Test and unseen (hot-switch) are evaluated once, after selection.

## 4. Train

```bash
export PYTHONPATH=$PWD
python3 scripts/experiments/train_forecast_baselines.py \
  --run-name rf-xgb-h3-grid1 --note "first grid on the 25 official runs" --register
python3 scripts/experiments/plot_forecast_baselines.py   # also attaches plots to the run
```

## 5. What you see in the UI

Switch the left toggle to **Model training**, open `ndt-forecasting`.

| Where | Content |
|---|---|
| Runs list | one parent run per training; expand it for 20 child runs (`random_forest #k`, `xgboost #k`) |
| Overview | tags (git commit/branch/dirty, dataset and model version, note), params, datasets used |
| Model metrics | `grid_<model>_validation_rmse` and `grid_<model>_best_validation_rmse` per candidate (live while the grid runs, Auto-refresh on), `xgboost_learning_curve_validation_rmse` per boosting round, validation/test/unseen and per-workload metrics, `skill_*` ratios vs persistence |
| System metrics | CPU, memory, disk, network of the training process |
| Artifacts | `figures/` (feature importance, skill vs persistence), `plots/` (actual vs forecast per workload), `tables/` (per-workload metrics, grid search), `forecast-models-v2/` (all local JSON/joblib files) and the logged models |
| Compare | select runs → Compare: parallel coordinates of grid params vs validation RMSE |
| Models | `ndt-forecast-random-forest`, `ndt-forecast-xgboost`; each new version carries alias `challenger` and tags `test_rmse`, `unseen_rmse`, `dataset_version` |

Promote a version after review: Models → version → Aliases → add `champion`.
The runtime predictor can then load `models:/ndt-forecast-xgboost@champion`.

The logged Random Forest / XGBoost models predict the **change**
`U(t+h) - U(t)`; the forecast is `U(t) + model.predict(X)`.

## 6. Querying runs from Python

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

`mlflow.db` and `mlruns/` are currently committed in this repository; every
training run modifies them. Agree as a team before untracking them
(`git rm -r --cached mlflow.db mlruns`).
