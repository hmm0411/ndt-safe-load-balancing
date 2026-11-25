#!/usr/bin/env bash
# forecast-models-v4 on the VM, unattended (25/11): train -> m_f/coverage -> artifact -> summary.
# Needs the processed v4 dataset and split (build_forecast_dataset.py + build_forecast_splits.py
# with NDT_FORECAST_PROFILE=4c40s). Same grids and selection rule as v3 (models.yaml), no
# MLflow (mlflow.db is shared). Training is CPU-heavy, so every step waits while a Mininet
# network is up (someone measuring) instead of disturbing it.
#   tmux new -s v4 'scripts/experiments/run_forecast_v4_vm.sh; bash'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1090
source ~/ndt-venv/bin/activate
export PYTHONPATH="$PWD"
OUT="${OUT:-$HOME/v4_overnight}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
MODEL_DIR=data/models/forecasting/forecast-models-v4
DATA_DIR=data/forecasting/processed/forecast-dataset-v4-h3-provisional
echo "== start $(date -u +%FT%TZ) git $(git rev-parse --short HEAD) $(git branch --show-current)"

wait_idle() {  # at most 8 h of waiting in total
    local waited=0
    while pgrep -f "mininet:" >/dev/null; do
        (( waited == 0 )) && echo "   Mininet is up (someone measuring): waiting"
        sleep 60; waited=$((waited + 60))
        (( waited >= 28800 )) && { echo "!! waited 8 h for Mininet to stop: STOP"; exit 1; }
    done
}
step() {  # step <name> <command...>: log in $OUT/<name>.log, stop on a non-zero exit
    local name="$1"; shift
    wait_idle
    echo "== $name $(date -u +%T)"
    "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    tail -6 "$OUT/$name.log" | sed 's/^/   /'
    echo "   exit $rc"
    (( rc == 0 )) || { echo "!! $name failed: STOP (full log $OUT/$name.log)"; exit 1; }
}

# CI with the default profile (as GitHub runs it). Recorded, not blocking: training writes
# only gitignored/local files, and a lint error must not cost the night.
wait_idle
echo "== ci $(date -u +%T)"
env -u NDT_FORECAST_PROFILE make ci > "$OUT/ci.log" 2>&1
echo "   ci exit $? ($(grep -E '^(Ran|OK|FAILED)' "$OUT/ci.log" | tr '\n' ' '))"

export NDT_FORECAST_PROFILE=4c40s
[[ -f "$DATA_DIR/split_manifest.json" ]] || { echo "!! no $DATA_DIR/split_manifest.json: STOP"; exit 1; }
[[ -e "$MODEL_DIR" ]] && { echo "!! $MODEL_DIR exists, never overwrite: STOP"; exit 1; }
step versions python3 -c "from src.twin.forecasting import versions as v; \
assert v.MODEL_VERSION == 'forecast-models-v4', v.MODEL_VERSION; print(v.MODEL_VERSION, v.dataset_version())"
step train python3 scripts/experiments/train_forecast_baselines.py --no-mlflow \
    --run-name forecast-models-v4 --note "4C40S retrain: 25 forecast4c-* runs, seeds 9921-9925"
step margin python3 scripts/experiments/compute_forecast_margin.py --no-mlflow
step export python3 scripts/experiments/export_forecast_artifact.py

echo "== summary $(date -u +%FT%TZ)"
for f in model_selection skill_vs_persistence error_margin; do
    echo "--- $f.json"; cat "$MODEL_DIR/$f.json"
done
echo "--- files (sha256)"; (cd "$MODEL_DIR" && sha256sum ./* | sed 's#  \./#  #')
echo "--- dataset (sha256)"; (cd "$DATA_DIR" && sha256sum ./*.json | sed 's#  \./#  #')
echo "--- capacity v5_noprobe (to commit it: same sha in every copy, else ask Vy)"
CAPS=$(ls "$HOME"/*/data/benchmarks/controller_capacity_v5_noprobe.json 2>/dev/null)
for f in $CAPS; do sha256sum "$f"; done
[[ -n "$CAPS" ]] && cat "$(echo "$CAPS" | head -1)"
echo "== done $(date -u +%FT%TZ): all steps exit 0"
