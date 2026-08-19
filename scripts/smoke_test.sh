#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/ci

PIDS=()
cleanup() {
  set +e
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  sudo mn -c >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_http() {
  local url="$1"
  for _ in $(seq 1 40); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

./scripts/start_c1.sh >logs/ci/c1.log 2>&1 & PIDS+=("$!")
./scripts/start_c2.sh >logs/ci/c2.log 2>&1 & PIDS+=("$!")
./scripts/start_orchestrator.sh >logs/ci/orchestrator.log 2>&1 & PIDS+=("$!")

wait_http http://127.0.0.1:8081/api/v1/state
wait_http http://127.0.0.1:8082/api/v1/state
wait_http http://127.0.0.1:9000/health

sudo -E env PYTHONPATH="$ROOT" python3 tests/integration/smoke_role_telemetry.py
