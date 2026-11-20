#!/usr/bin/env bash
# CPU-pinning pilot (configs/experiments/cpu_pin_pilot_v1.yaml): 15 runs, ~35 min.
# Needs c1-c4 and the orchestrator already running (4C40S, any capacity file: the probe
# latencies do not use U). Pins the RUNNING processes, runs the 3 rates, scores, and puts every
# process back on all 8 cores at the end, so later runs are not pinned by accident.
#   scripts/experiments/run_cpu_pin_pilot.sh [--shutdown]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
SHUTDOWN=0; [[ "${1:-}" == "--shutdown" ]] && SHUTDOWN=1
CFG=configs/experiments/probe_pin_pilot_4c40s.yaml
PY="$(command -v python3)"; export PYTHONPATH="$PWD"
mkdir -p data/experiment_runs
LOG="data/experiment_runs/cpu_pin_pilot_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
fail() { echo "PREFLIGHT FAIL: $1"; exit 1; }

echo "== preflight $(date -u +%FT%TZ)"
[[ "$(nproc)" == "8" ]] || fail "nproc is $(nproc), the pinning plan needs 8"
pgrep -f "mininet:" >/dev/null && fail "a Mininet network is already up (someone else's run?)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || fail "tracked files changed"
declare -A CORE=([6653]=0 [6654]=1 [6655]=2 [6656]=3)
PIDS=()
for port in 6653 6654 6655 6656; do
    pid=$(pgrep -f "ryu-manager --ofp-tcp-listen-port ${port}" | head -1)
    [[ -n "$pid" ]] || fail "no ryu-manager on port ${port}"
    PIDS+=("$pid")
done
ORCH=$(pgrep -f "uvicorn src.orchestrator.app" | head -1); [[ -n "$ORCH" ]] || fail "no orchestrator"
OVS=$(pgrep -x ovs-vswitchd | head -1); [[ -n "$OVS" ]] || fail "no ovs-vswitchd"
OTHERS=$(pgrep -f "mlflow.server" | tr '\n' ' ')

unpin() {
    for pid in "${PIDS[@]}" "$ORCH" $OTHERS; do taskset -apc 0-7 "$pid" >/dev/null 2>&1; done
    sudo taskset -apc 0-7 "$OVS" >/dev/null 2>&1
    echo "affinity restored to 0-7"
}
trap unpin EXIT
i=0
for port in 6653 6654 6655 6656; do
    taskset -apc "${CORE[$port]}" "${PIDS[$i]}" >/dev/null || fail "cannot pin port ${port}"
    i=$((i + 1))
done
for pid in "$ORCH" $OTHERS; do taskset -apc 4-7 "$pid" >/dev/null || fail "cannot pin ${pid}"; done
sudo taskset -apc 4-7 "$OVS" >/dev/null || fail "cannot pin ovs-vswitchd"
for pid in "${PIDS[@]}" "$ORCH" "$OVS"; do echo "pinned: $(taskset -cp "$pid") $(ps -o comm= -p "$pid")"; done
echo "PREFLIGHT: PASS"

clean_leftover_links() {  # same rule as run_week15_vm_4c40s.sh: never mn -c
    local links bridges
    links=$(ip -o link show | awk -F': ' '{print $2}' | cut -d@ -f1 | grep -E '^s[0-9]+-eth[0-9]+$')
    bridges=$(sudo ovs-vsctl list-br 2>/dev/null | grep -E '^s[0-9]+$')
    [[ -z "$links" && -z "$bridges" ]] && return 0
    for link in $links; do sudo ip link del "$link" 2>/dev/null; done
    for bridge in $bridges; do sudo ovs-vsctl --if-exists del-br "$bridge"; done
}

for rate in 30 90 140; do
    echo "-- c1 ${rate} fps $(date -u +%T)"
    clean_leftover_links
    sudo -E env PYTHONPATH="$PWD" taskset -c 4-7 "$PY" -m src.experiments.topologies.capacity_4c40s \
        --config "$CFG" --controller c1 --rate "$rate" || echo "!! c1 ${rate}: a repeat failed"
done
sudo chown -R "$(id -un):$(id -gn)" data/experiment_runs 2>/dev/null || true
"$PY" -m src.experiments.capacity.sweep_4c40s --config "$CFG" --list
"$PY" -m src.experiments.capacity.pin_pilot
if [[ "$SHUTDOWN" -eq 1 ]]; then
    unpin; trap - EXIT
    echo "shutting the VM down in 5 minutes (cancel: sudo shutdown -c)"; sudo shutdown -h +5
fi
