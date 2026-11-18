#!/usr/bin/env bash
# Week 15, Day 3-4 on the VM, in one pass (plan 16-22/11):
#   probe_off    Day 3  probe-induced load: all controllers at 30 fps, probe OFF   (5 runs)
#   probe_sweep  Day 3  each controller 30/90/140 fps, others 30, probe ON         (60 runs)
#   coarse       Day 4  capacity-v5 coarse, others idle, probe ON                  (56 runs)
#   fine         Day 4  capacity-v5 fine, 16 rates x 5 repeats, probe ON           (320 runs)
#
# Needs, already running: c1-c4 (scripts/start_c1.sh .. start_c4.sh) and the orchestrator
# with NDT_CONTROLLERS_CONFIG=configs/controllers_4c40s.yaml (+ NDT_CAPACITY_FILE).
# Valid runs are skipped, so the same command resumes after a stop. A failed run is
# reported and left in place; it does not stop the pass.
#
#   scripts/experiments/run_week15_vm_4c40s.sh [--stages probe_off,probe_sweep,coarse,fine]
#                                              [--shutdown] [--preflight-only]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGES="probe_off,probe_sweep,coarse,fine"
SHUTDOWN=0
PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --shutdown) SHUTDOWN=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

declare -A CONFIG=(
    [probe_off]=configs/experiments/probe_off_4c40s.yaml
    [probe_sweep]=configs/experiments/probe_sweep_4c40s.yaml
    [coarse]=configs/experiments/capacity_coarse_4c40s.yaml
    [fine]=configs/experiments/capacity_fine_4c40s.yaml
)
PY="$(command -v python3)"
export PYTHONPATH="$PWD"
mkdir -p data/experiment_runs
LOG="data/experiment_runs/week15_vm_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "LOG: $LOG"

fail() { echo "PREFLIGHT FAIL: $1"; exit 1; }

# ---------------------------------------------------------------- preflight
echo "== preflight $(date -u +%FT%TZ)"
[[ "$(nproc)" == "8" ]] || fail "nproc is $(nproc), capacity needs the 8-vCPU machine"
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
(( MEM_KB >= 30000000 )) || fail "MemTotal ${MEM_KB} kB; e2-standard-8 has 32 GB"
echo "cpu: $(lscpu | sed -n 's/^Model name:[[:space:]]*//p') | mem_kb: ${MEM_KB} | kernel: $(uname -r)"
for port in 6653 6654 6655 6656 8081 8082 8083 8084 9000; do
    ss -ltn "sport = :${port}" | grep -q LISTEN || fail "nothing listens on port ${port}"
done
CONTROLLERS=$(curl -fsS http://127.0.0.1:9000/api/v1/twin/state | "$PY" -c \
    'import json,sys; d=json.load(sys.stdin); print(",".join(sorted(c["controller_id"] for c in d.get("controllers",[]))))') \
    || fail "orchestrator twin/state not readable"
[[ "$CONTROLLERS" == "c1,c2,c3,c4" ]] || fail "orchestrator sees '${CONTROLLERS}', not c1..c4 (NDT_CONTROLLERS_CONFIG?)"
if pgrep -f "mininet:" >/dev/null; then
    fail "a Mininet network is already up (someone else's run?); check before cleaning"
fi
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || fail "tracked files changed (git status)"
echo "git: $(git rev-parse HEAD) $(git describe --tags --always 2>/dev/null)"
for stage in ${STAGES//,/ }; do
    [[ -n "${CONFIG[$stage]:-}" ]] || fail "unknown stage ${stage}"
    "$PY" -m src.experiments.capacity.sweep_4c40s --config "${CONFIG[$stage]}" \
        || echo "note: ${stage} already has failed or incomplete runs (listed with --list)"
done
echo "PREFLIGHT: PASS"
[[ "$PREFLIGHT_ONLY" -eq 1 ]] && exit 0

# ---------------------------------------------------------------- runs
# Leftover veth pairs of an interrupted Mininet (no process left) make the next
# net.build() fail with "File exists". Clean them only when no Mininet is running.
clean_leftover_links() {
    if ip -br link | grep -qE '^s[0-9]+-eth' && ! pgrep -f "mininet:" >/dev/null; then
        echo "cleaning leftover Mininet links"
        sudo mn -c >/dev/null 2>&1
    fi
}

FAILED_PROCESSES=0
CONSECUTIVE_FAILURES=0
MAX_CONSECUTIVE_FAILURES=3
ABORTED=0
for stage in ${STAGES//,/ }; do
    [[ "$ABORTED" -eq 1 ]] && break
    cfg="${CONFIG[$stage]}"
    echo "== stage ${stage} (${cfg}) $(date -u +%FT%TZ)"
    read -r -a targets <<< "$("$PY" -c "
from pathlib import Path
from src.experiments.common.config import load_yaml
from src.experiments.capacity.sweep_4c40s import sweep_controllers
c = load_yaml(Path('${cfg}')); cc = load_yaml(Path('configs/controllers_4c40s.yaml'))
print(' '.join(sweep_controllers(c, cc)))")"
    read -r -a rates <<< "$("$PY" -c "
from pathlib import Path
from src.experiments.common.config import load_yaml
print(' '.join(str(r) for r in load_yaml(Path('${cfg}'))['workload']['rates_fps']))")"
    for controller in "${targets[@]}"; do
        for rate in "${rates[@]}"; do
            echo "-- ${stage} ${controller} ${rate} fps $(date -u +%T)"
            clean_leftover_links
            if sudo -E env PYTHONPATH="$PWD" "$PY" -m src.experiments.topologies.capacity_4c40s \
                    --config "$cfg" --controller "$controller" --rate "$rate"; then
                CONSECUTIVE_FAILURES=0
            else
                FAILED_PROCESSES=$((FAILED_PROCESSES + 1))
                CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
                echo "!! ${stage} ${controller} ${rate}: a repeat failed (details above)"
                sudo mn -c >/dev/null 2>&1
                if (( CONSECUTIVE_FAILURES >= MAX_CONSECUTIVE_FAILURES )); then
                    echo "!! ${MAX_CONSECUTIVE_FAILURES} processes failed in a row: pass ABORTED"
                    ABORTED=1
                    break 2
                fi
            fi
        done
    done
done

# ---------------------------------------------------------------- summary
echo "== summary $(date -u +%FT%TZ)"
for stage in ${STAGES//,/ }; do
    "$PY" -m src.experiments.capacity.sweep_4c40s --config "${CONFIG[$stage]}" --list
done
echo "processes with a failed repeat: ${FAILED_PROCESSES}"
[[ "$ABORTED" -eq 1 ]] && echo "PASS ABORTED: see the first '!!' lines above"
sudo chown -R "$(id -un):$(id -gn)" data/experiment_runs 2>/dev/null || true
if [[ "$SHUTDOWN" -eq 1 ]]; then
    echo "shutting the VM down in 5 minutes (cancel: sudo shutdown -c)"
    sudo shutdown -h +5
fi
