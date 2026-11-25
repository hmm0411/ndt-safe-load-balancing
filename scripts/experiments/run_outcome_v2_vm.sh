#!/usr/bin/env bash
# post-migration-v2 on the VM, unattended (25/11): the Week 12 outcome pipeline unchanged, on
# Vy's probe-off 4C collection (850 attempts, seeds 9901-9905) with its declared split
# (evidence/week15/split_manifest_4c.json, arm probe_off: r01-r03 train, r04 validation,
# r05 test, test set locked before any model run).
#   dataset -> Analytical on validation -> Direct -> Hybrid -> select on validation, then
#   the test split ONCE (test_lock.json) -> margins m_U/m_R + coverage -> artifact -> evidence
# Same grids (configs/outcome/models.yaml), selection rule (evaluation.yaml) and margin rule
# (artifact rules) as v1. Waits while a Mininet network is up; stops at the first error;
# never overwrites a dataset, report or artifact folder.
#   RAW=~/w15-day34/data/experiment_runs/migration-outcome-4c-noprobe-v1 \
#     tmux new -d -s v2 'scripts/experiments/run_outcome_v2_vm.sh; bash'
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1090
source ~/ndt-venv/bin/activate
export PYTHONPATH="$PWD"
RAW="${RAW:?set RAW to the migration-outcome-4c-noprobe-v1 attempt root}"
OUT="${OUT:-$HOME/v2_outcome}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
CFG=configs/outcome/dataset_4c_v2.yaml
ART_CFG=configs/outcome/artifact_4c_v2.yaml
CAP=data/benchmarks/controller_capacity_v5_noprobe.json
SPLIT=evidence/week15/split_manifest_4c.json
DATA=data/outcome/processed/migration-outcome-processed-v2
REP=data/outcome/reports/4c-v2
ART=data/models/outcome/post-migration-v2
EVI=evidence/week15/outcome_v2
COMMON=(--dataset "$DATA" --config "$CFG" --direct-dir "$REP/direct" --hybrid-dir "$REP/hybrid")
echo "== start $(date -u +%FT%TZ) git $(git rev-parse --short HEAD) $(git branch --show-current)"
echo "   raw $RAW"

wait_idle() {
    local waited=0
    while pgrep -f "mininet:" >/dev/null; do
        (( waited == 0 )) && echo "   Mininet is up (someone measuring): waiting"
        sleep 60; waited=$((waited + 60))
        (( waited >= 28800 )) && { echo "!! waited 8 h for Mininet to stop: STOP"; exit 1; }
    done
}
step() {
    local name="$1"; shift
    wait_idle
    echo "== $name $(date -u +%T)"
    "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    tail -12 "$OUT/$name.log" | sed 's/^/   /'
    echo "   exit $rc"
    (( rc == 0 )) || { echo "!! $name failed: STOP (full log $OUT/$name.log)"; exit 1; }
}

for d in "$DATA" "$REP" "$ART" "$EVI"; do
    [[ -e "$d" ]] && { echo "!! $d exists, never overwrite: STOP"; exit 1; }
done
[[ -d "$RAW" && -f "$CAP" && -f "$SPLIT" ]] || { echo "!! missing raw root, $CAP or $SPLIT: STOP"; exit 1; }
step test_lock_check python3 -c "
import hashlib, json
arm = json.load(open('$SPLIT'))['arms']['probe_off']
test = '\n'.join(arm['splits']['test']['attempts'])
assert hashlib.sha256(test.encode()).hexdigest() == arm['test_set_sha256'], 'test set changed'
print('test set sha256 OK', arm['test_set_sha256'][:12], 'planned', arm['planned'],
      'on disk', arm['measured_on_disk'])"
step dataset python3 scripts/experiments/build_outcome_dataset.py --config "$CFG" \
    --raw-root "$RAW" --split "$SPLIT" --split-arm probe_off --capacity "$CAP" --out "$DATA"
step analytical python3 scripts/experiments/evaluate_outcome_analytical.py --dataset "$DATA" \
    --config "$CFG" --split validation --out "$REP/analytical_validation_metrics.csv"
step direct python3 scripts/experiments/train_outcome_models.py --stage direct "${COMMON[@]}" \
    --out "$REP/direct"
step hybrid python3 scripts/experiments/train_outcome_models.py --stage hybrid "${COMMON[@]}" \
    --out "$REP/hybrid"
step select_and_test python3 scripts/experiments/train_outcome_models.py \
    --stage select-and-test "${COMMON[@]}" --out "$REP/final"
step export python3 scripts/experiments/export_outcome_artifact.py "${COMMON[@]}" \
    --final-dir "$REP/final" --artifact-config "$ART_CFG" --capacity-file "$CAP" \
    --artifact-dir "$ART"
step check python3 scripts/experiments/export_outcome_artifact.py --check \
    --artifact-config "$ART_CFG" --capacity-file "$CAP" --artifact-dir "$ART"

# evidence: the processed-dataset reports and every report file (small CSV/JSON), with sha256
mkdir -p "$EVI"
cp "$DATA"/*.json "$EVI"/
cp "$REP"/analytical_validation_metrics.csv "$EVI"/
for s in direct hybrid final; do
    for f in "$REP/$s"/*.csv "$REP/$s"/*.json; do cp "$f" "$EVI/${s}_$(basename "$f")"; done
done
(cd "$EVI" && sha256sum ./* | sed 's#  \./#  #' > SHA256SUMS)
echo "== summary $(date -u +%FT%TZ)"
for f in final_model_selection.json final_uncertainty_margin.json final_test_lock.json; do
    echo "--- $f"; head -c 4000 "$EVI/$f"; echo
done
echo "--- final_test_metrics.csv"; cat "$EVI/final_test_metrics.csv"
echo "--- artifact (sha256)"; (cd "$ART" && find . -type f | sort | xargs sha256sum)
du -sh "$ART" "$EVI"
echo "== done $(date -u +%FT%TZ): all steps exit 0"
