#!/usr/bin/env bash
# Start the MLflow tracking server for the forecasting experiments.
#
#   - reads the same SQLite store that train_forecast_baselines.py writes
#     (sqlite:///mlflow.db at the repo root), so training needs no server;
#   - basic-auth login (users/passwords in basic_auth.db);
#   - listens on all interfaces so several machines can open the UI; reach
#     it only through a GCP firewall rule limited to your IPs (docs/mlflow.md).
#
# Environment (all optional except the admin password on first start):
#   MLFLOW_AUTH_ADMIN_PASSWORD  admin password, used only to create the admin
#   MLFLOW_HOST                 default 0.0.0.0
#   MLFLOW_PORT                 default 5000
#   MLFLOW_ALLOWED_HOSTS        Host headers to accept, default "*"
#   MLFLOW_WORKERS              default 2
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export MLFLOW_AUTH_CONFIG_PATH="${MLFLOW_AUTH_CONFIG_PATH:-$REPO_ROOT/deploy/mlflow/basic_auth.ini}"

if [[ ! -f basic_auth.db && -z "${MLFLOW_AUTH_ADMIN_PASSWORD:-}" ]]; then
  echo "First start: set MLFLOW_AUTH_ADMIN_PASSWORD to create the admin user." >&2
  exit 1
fi

# The login form needs a stable CSRF secret; create one once (git-ignored).
SECRET_FILE="$REPO_ROOT/.mlflow_flask_secret"
if [[ -z "${MLFLOW_FLASK_SERVER_SECRET_KEY:-}" ]]; then
  if [[ ! -s "$SECRET_FILE" ]]; then
    (umask 077 && python3 -c 'import secrets; print(secrets.token_hex(32))' > "$SECRET_FILE")
  fi
  MLFLOW_FLASK_SERVER_SECRET_KEY="$(cat "$SECRET_FILE")"
  export MLFLOW_FLASK_SERVER_SECRET_KEY
fi

exec mlflow server \
  --app-name basic-auth \
  --backend-store-uri "sqlite:///$REPO_ROOT/mlflow.db" \
  --host "${MLFLOW_HOST:-0.0.0.0}" \
  --port "${MLFLOW_PORT:-5000}" \
  --workers "${MLFLOW_WORKERS:-2}" \
  --allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-*}"
