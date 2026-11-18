#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${RYU_VENV:-$HOME/ryu-venv}/bin/activate"
export CONTROLLER_ID=c4
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec ryu-manager --ofp-tcp-listen-port 6656 --wsapi-port 8084 src/controller/reactive_controller.py
