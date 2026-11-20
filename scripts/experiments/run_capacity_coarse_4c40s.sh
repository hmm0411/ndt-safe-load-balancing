#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$0")/run_week15_vm_4c40s.sh" --stages coarse "$@"
