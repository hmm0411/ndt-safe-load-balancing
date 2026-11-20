#!/usr/bin/env bash
# Week 15 Day 5 on the VM: forecast runs of the 2C -> 4C transfer test, one queue, unattended.
#   10 runs = 5 workloads x seeds 9701-9702 (configs/experiments/transfer_4c_v1.yaml), 180 s
#   each, about 40 min with setup. One Mininet at a time: two measurements in parallel on one
#   VM would change each other's load and latency, so the saving is in never idling instead.
#
# Needs, already running: c1-c4 and the orchestrator with NDT_CONTROLLERS_CONFIG=
# configs/controllers_4c40s.yaml and NDT_CAPACITY_FILE=data/benchmarks/controller_capacity_v5.json
# (restart it after capacity-v5 is derived), and data/benchmarks/capacity_runs_4c40s.csv.
# Existing run directories are skipped, so the same command resumes after a stop.
# Run it from its own worktree; see RUNS_ROOT below.
#
#   scripts/experiments/run_week15_day5_vm.sh [--dry-run] [--shutdown] [--preflight-only]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=0; SHUTDOWN=0; PREFLIGHT_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --shutdown) SHUTDOWN=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

CONFIG=configs/experiments/transfer_4c_v1.yaml
# Capacity the runs are made with (20/11: Huong 2 -> the campaign runs with the probe off, so
# the probe-off capacity, once derived). Override all three, e.g.
#   CAPACITY=data/benchmarks/controller_capacity_v6.json \
#   CAPACITY_EXPERIMENTS="capacity-4c40s-coarse capacity-4c40s-noprobe-fine" scripts/...
CAPACITY="${CAPACITY:-data/benchmarks/controller_capacity_v5.json}"
RUNS_CSV="${RUNS_CSV:-data/benchmarks/capacity_runs_4c40s.csv}"
CAPACITY_EXPERIMENTS="${CAPACITY_EXPERIMENTS:-capacity-4c40s-coarse capacity-4c40s-fine}"
EXP_ARGS=(); for e in $CAPACITY_EXPERIMENTS; do EXP_ARGS+=(--capacity-experiment "$e"); done
echo "capacity: $CAPACITY | rows: $RUNS_CSV | experiments: $CAPACITY_EXPERIMENTS"
# RUNS_ROOT: where the raw runs live. A second worktree (so nobody switches the branch under
# a running queue, 20/11) points it at the first one: RUNS_ROOT=~/w15-day34/data/experiment_runs
RUNS_ROOT="${RUNS_ROOT:-data/experiment_runs}"
OUT="${RUNS_ROOT}/transfer_4c"
PY="$(command -v python3)"
export PYTHONPATH="$PWD"
mkdir -p "$OUT"
LOG="$OUT/day5_vm_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
echo "LOG: $LOG"
fail() { echo "PREFLIGHT FAIL: $1"; exit 1; }

# ---------------------------------------------------------------- preflight
echo "== preflight $(date -u +%FT%TZ)"
mapfile -t PLAN < <("$PY" -m src.experiments.workloads.forecast_4c --config "$CONFIG")
(( ${#PLAN[@]} == 10 )) || fail "plan has ${#PLAN[@]} runs, expected 10"
if [[ "$DRY_RUN" -eq 1 ]]; then printf '%s\n' "${PLAN[@]}"; exit 0; fi
[[ -f "$CAPACITY" && -f "$RUNS_CSV" ]] || fail "capacity-v5 not derived yet ($CAPACITY, $RUNS_CSV)"
[[ "$(nproc)" == "8" ]] || fail "nproc is $(nproc), capacity-v5 was measured on 8 vCPU"
CPU_NOW="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
FINE_RUN="$(ls -d "${RUNS_ROOT}"/capacity-4c40s-fine-* 2>/dev/null | head -1)"
[[ -n "$FINE_RUN" && -f "$FINE_RUN/run_conditions.json" ]] || fail "no fine-sweep run_conditions.json"
CPU_V5="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['cpu_model'])" \
    "$FINE_RUN/run_conditions.json")"
echo "cpu now: ${CPU_NOW} | cpu of capacity-v5: ${CPU_V5}"
[[ "$CPU_NOW" == "$CPU_V5" ]] || fail "CPU model changed since capacity-v5: c_safe does not hold here"
for port in 6653 6654 6655 6656 8081 8082 8083 8084 9000; do
    ss -ltn "sport = :${port}" | grep -q LISTEN || fail "nothing listens on port ${port}"
done
CONTROLLERS=$(curl -fsS http://127.0.0.1:9000/api/v1/twin/state | "$PY" -c \
    'import json,sys; d=json.load(sys.stdin); print(",".join(sorted(c["controller_id"] for c in d.get("controllers",[]))))') \
    || fail "orchestrator twin/state not readable"
[[ "$CONTROLLERS" == "c1,c2,c3,c4" ]] || fail "orchestrator sees '${CONTROLLERS}', not c1..c4"
pgrep -f "mininet:" >/dev/null && fail "a Mininet network is already up (someone else's run?)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || fail "tracked files changed (git status)"
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
(( FREE_GB >= 3 )) || fail "only ${FREE_GB} GB free on the disk"
"$PY" -m src.experiments.workloads.forecast_4c --check-range "$RUNS_CSV" "$CAPACITY" "${EXP_ARGS[@]}" \
    || fail "a workload U falls outside the measured capacity range"
GIT_COMMIT="$(git rev-parse HEAD)"
GIT_BRANCH="$(git branch --show-current)"
echo "git: ${GIT_COMMIT} $(git describe --tags --always 2>/dev/null)"
echo "PREFLIGHT: PASS"
[[ "$PREFLIGHT_ONLY" -eq 1 ]] && exit 0

clean_leftover_links() {  # same rule as run_week15_vm_4c40s.sh: never mn -c
    local links bridges
    links=$(ip -o link show | awk -F': ' '{print $2}' | cut -d@ -f1 | grep -E '^s[0-9]+-eth[0-9]+$')
    bridges=$(sudo ovs-vsctl list-br 2>/dev/null | grep -E '^s[0-9]+$')
    [[ -z "$links" && -z "$bridges" ]] && return 0
    pgrep -f "mininet:" >/dev/null && { echo "!! Mininet processes still running; not cleaning"; return 1; }
    for link in $links; do sudo ip link del "$link" 2>/dev/null; done
    for bridge in $bridges; do sudo ovs-vsctl --if-exists del-br "$bridge"; done
}

# ---------------------------------------------------------------- runs
FAILED=0; CONSECUTIVE=0
for line in "${PLAN[@]}"; do
    IFS='|' read -r workload seed repeat run_id <<< "$line"
    if [[ -e "$OUT/$run_id" ]]; then echo "SKIP existing: $run_id"; continue; fi
    # 20/11: the worktree was switched to another branch in the middle of the first queue, and
    # the next runs ran other code. Same commit for every run, or stop.
    if [[ "$(git rev-parse HEAD)" != "$GIT_COMMIT" || "$(git branch --show-current)" != "$GIT_BRANCH" ]]; then
        echo "!! the worktree moved to $(git branch --show-current) $(git rev-parse --short HEAD)" \
             "(queue started on ${GIT_BRANCH} ${GIT_COMMIT:0:7}): queue ABORTED"
        break
    fi
    echo "-- $run_id $(date -u +%T)"
    clean_leftover_links
    if sudo -E env PYTHONPATH="$PWD" "$PY" -m src.experiments.topologies.runtime_4c40s \
            --forecast "$workload" --seed "$seed" --repeat-index "$repeat" --run-id "$run_id" \
            --output-dir "$OUT" --git-commit "$GIT_COMMIT" --capacity-runs "$RUNS_CSV" \
            --controller-capacity "$CAPACITY" "${EXP_ARGS[@]}"; then
        CONSECUTIVE=0; echo "PASS: $run_id"
    else
        FAILED=$((FAILED + 1)); CONSECUTIVE=$((CONSECUTIVE + 1))
        echo "!! FAIL: $run_id (details above)"
        sudo pkill -f "mininet:[hp][0-9]+" 2>/dev/null
        clean_leftover_links
        (( CONSECUTIVE >= 3 )) && { echo "!! 3 runs failed in a row: queue ABORTED"; break; }
    fi
done

echo "== summary $(date -u +%FT%TZ): failed ${FAILED}"
sudo chown -R "$(id -un):$(id -gn)" "$OUT" 2>/dev/null || true
if [[ "$SHUTDOWN" -eq 1 ]]; then
    echo "shutting the VM down in 5 minutes (cancel: sudo shutdown -c)"
    sudo shutdown -h +5
fi
