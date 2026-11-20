#!/usr/bin/env bash
# Week 15 Day 7: raw 4C40S run directories out of the VM, one archive per sweep, with sha256.
# Run from the repo root (the worktree that holds data/experiment_runs).
#   scripts/experiments/backup_week15_runs.sh [OUT_DIR]      default ~/backup-week15
# Then download every part_* and SHA256SUMS (SSH-in-browser: Download file) and check
# the hashes on the local machine before deleting anything on the VM.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
OUT="${1:-$HOME/backup-week15}"
mkdir -p "$OUT"
for kind in probe-off-4c40s probe-sweep-4c40s capacity-4c40s-coarse capacity-4c40s-fine; do
    mapfile -t dirs < <(find data/experiment_runs data/experiment_runs_invalid -maxdepth 1 \
        -type d -name "${kind}-*" 2>/dev/null | sort)
    if [[ ${#dirs[@]} -eq 0 ]]; then echo "skip ${kind}: no run"; continue; fi
    tar -czf "$OUT/${kind}.tar.gz" "${dirs[@]}"
    echo "${kind}: ${#dirs[@]} run directories"
done
cp data/experiment_runs/week15_vm_*.log "$OUT/" 2>/dev/null || true
(cd "$OUT" && for f in *.tar.gz; do split -b 400M -d "$f" "${f}.part_"; done)
(cd "$OUT" && sha256sum *.tar.gz *.part_* week15_vm_*.log 2>/dev/null > SHA256SUMS)
ls -lh "$OUT"
echo "git: $(git rev-parse HEAD)"
