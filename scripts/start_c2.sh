#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${RYU_VENV:-$HOME/ryu-venv}/bin/activate"
export CONTROLLER_ID=c2
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec ryu-manager --ofp-tcp-listen-port 6654 --wsapi-port 8082 src/controller/reactive_controller.py
