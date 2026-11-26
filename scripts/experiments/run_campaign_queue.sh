#!/usr/bin/env bash
# Closed-loop campaign queue on the VM: every episode of a plan, in the plan's (shuffled)
# order, unattended. Existing episode folders are skipped, so the same command resumes.
#   scripts/experiments/run_campaign_queue.sh results/campaign/plan_B.csv [--limit N] [--shutdown]
#   CONFIG=configs/experiments/campaign_2c_v1.yaml scripts/experiments/run_campaign_queue.sh \
#       results/campaign/plan_B2.csv     (2C20S; episodes go to the config's out_root)
# Needs: ports 6653-6656, 8081-8084, 9000 free (stop any other orchestrator/controller
# first), duration_s set in configs/experiments/campaign_4c_v1.yaml (after the pilot).
# Start it with `tmux new -d -s campaignB '...'` and never `tmux kill-session` first: a
# second start then fails on the duplicate session, and the lock below refuses a second
# queue (26/11: a doubled start killed the first queue and the second shut the VM down).
# --shutdown powers the VM off only when the whole plan was gone through.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PLAN="${1:?plan CSV}"; shift
LIMIT=0; SHUTDOWN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) LIMIT="$2"; shift 2 ;;
        --shutdown) SHUTDOWN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
# shellcheck disable=SC1090
source ~/ndt-venv/bin/activate
export PYTHONPATH="$PWD"
CONFIG="${CONFIG:-configs/experiments/campaign_4c_v1.yaml}"
[[ -f "$CONFIG" ]] || { echo "!! no config $CONFIG"; exit 2; }
CONFIG_OUT=$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get('out_root') or '')" "$CONFIG")
OUT_ROOT="${OUT_ROOT:-${CONFIG_OUT:-data/experiment_runs/campaign_4c}}"
mkdir -p "$OUT_ROOT"
# One queue at a time, whatever the topology (they share ports). The lock is inherited by
# the episode's processes, so it also stays held by an orphaned episode (controllers or
# Mininet left behind by a killed queue).
LOCK=data/experiment_runs/.campaign_queue.lock
exec 9>"$LOCK"
flock -n 9 || { echo "!! another queue (or an orphaned episode) holds $LOCK: STOP"; exit 1; }
LOG="$OUT_ROOT/queue_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
GIT_COMMIT="$(git rev-parse HEAD)"; GIT_BRANCH="$(git branch --show-current)"
echo "== queue $(date -u +%FT%TZ) plan $PLAN config $CONFIG out $OUT_ROOT git ${GIT_COMMIT:0:7} $GIT_BRANCH"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "!! tracked files changed: STOP"; exit 1; }

clean_leftover_links() {  # never mn -c (it would kill someone else's run)
    local links bridges
    links=$(ip -o link show | awk -F': ' '{print $2}' | cut -d@ -f1 | grep -E '^s[0-9]+-eth[0-9]+$')
    bridges=$(sudo ovs-vsctl list-br 2>/dev/null | grep -E '^s[0-9]+$')
    [[ -z "$links" && -z "$bridges" ]] && return 0
    pgrep -f "mininet:" >/dev/null && { echo "!! Mininet processes still running; not cleaning"; return 1; }
    for link in $links; do sudo ip link del "$link" 2>/dev/null; done
    for bridge in $bridges; do sudo ovs-vsctl --if-exists del-br "$bridge"; done
}

DONE=0; FAILED=0; CONSECUTIVE=0; STOPPED=0
while IFS=, read -r order episode_id method workload seed; do
    [[ "$order" == "order" ]] && continue
    if [[ -e "$OUT_ROOT/$episode_id" ]]; then echo "SKIP existing: $episode_id"; continue; fi
    if [[ "$(git rev-parse HEAD)" != "$GIT_COMMIT" || "$(git branch --show-current)" != "$GIT_BRANCH" ]]; then
        echo "!! the worktree moved: queue ABORTED"; STOPPED=1; break
    fi
    (( LIMIT > 0 && DONE + FAILED >= LIMIT )) && { echo "limit $LIMIT reached"; STOPPED=1; break; }
    echo "-- [$order] $episode_id $(date -u +%T)"
    clean_leftover_links
    if python3 scripts/experiments/run_campaign_episode.py --config "$CONFIG" --plan "$PLAN" \
            --episode-id "$episode_id" --out-root "$OUT_ROOT"; then
        DONE=$((DONE + 1)); CONSECUTIVE=0; echo "PASS: $episode_id"
    else
        FAILED=$((FAILED + 1)); CONSECUTIVE=$((CONSECUTIVE + 1)); echo "!! FAIL: $episode_id"
        sudo pkill -f "mininet:[hp][0-9]+" 2>/dev/null
        clean_leftover_links
        (( CONSECUTIVE >= 3 )) && { echo "!! 3 episodes failed in a row: queue ABORTED"; STOPPED=1; break; }
    fi
done < "$PLAN"
echo "== summary $(date -u +%FT%TZ): passed $DONE failed $FAILED"
if [[ "$SHUTDOWN" -eq 1 && "$STOPPED" -eq 1 ]]; then
    echo "queue stopped before the end of the plan: the VM stays on"
elif [[ "$SHUTDOWN" -eq 1 ]]; then
    echo "shutting the VM down in 5 minutes (cancel: sudo shutdown -c)"; sudo shutdown -h +5
fi
