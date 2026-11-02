#!/usr/bin/env python3
"""Copy My's Week 12 outputs into evidence/week12/ (Day 7).

The outputs live in gitignored folders (data/outcome/processed, data/outcome/reports);
this puts the files the Week 12 Definition of Done names into evidence/week12/ under their
plan names, and writes outcome_evidence_manifest.json (source, destination, sha256).
Evidence is frozen: an existing file with other content is an error, never overwritten.

    python3 scripts/experiments/collect_outcome_evidence.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DATASET = "data/outcome/processed/migration-outcome-processed-v1"
REPORTS = "data/outcome/reports"
# evidence name -> source (relative to the repo root)
FILES = {
    "processed_dataset_report.json": f"{DATASET}/dataset_quality_report.json",
    "processed_dataset_manifest.json": f"{DATASET}/dataset_manifest.json",
    "feature_schema.json": f"{DATASET}/feature_schema.json",
    "target_schema.json": f"{DATASET}/target_schema.json",
    "processed_split_manifest.json": f"{DATASET}/split_manifest.json",
    "analytical_metrics.csv": f"{REPORTS}/analytical_validation_metrics.csv",
    "direct_ml_metrics.csv": f"{REPORTS}/direct/direct_ml_metrics.csv",
    "direct_hyperparameters.json": f"{REPORTS}/direct/hyperparameters.json",
    "direct_hyperparameter_search.csv": f"{REPORTS}/direct/hyperparameter_search.csv",
    "hybrid_metrics.csv": f"{REPORTS}/hybrid/hybrid_metrics.csv",
    "hybrid_hyperparameters.json": f"{REPORTS}/hybrid/hyperparameters.json",
    "hybrid_hyperparameter_search.csv": f"{REPORTS}/hybrid/hyperparameter_search.csv",
    "residual_analysis.csv": f"{REPORTS}/hybrid/residual_analysis.csv",
    "model_selection.json": f"{REPORTS}/final/model_selection.json",
    "test_lock.json": f"{REPORTS}/final/test_lock.json",
    "validation_metrics.csv": f"{REPORTS}/final/validation_metrics.csv",
    "test_metrics.csv": f"{REPORTS}/final/test_metrics.csv",
    "per_workload_metrics.csv": f"{REPORTS}/final/per_workload_metrics.csv",
    "model_comparison.csv": f"{REPORTS}/final/model_comparison.csv",
    "bootstrap_ci.csv": f"{REPORTS}/final/bootstrap_ci.csv",
    "saturation_sensitivity.csv": f"{REPORTS}/final/saturation_sensitivity.csv",
    "overload_cases.csv": f"{REPORTS}/final/overload_cases.csv",
    "ablation.csv": f"{REPORTS}/final/ablation.csv",
    "uncertainty_margin.json": f"{REPORTS}/final/uncertainty_margin.json",
    "coverage_metrics.json": f"{REPORTS}/final/coverage_metrics.json",
}
MANIFEST = "outcome_evidence_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("evidence/week12"))
    args = parser.parse_args()
    missing = [src for src in FILES.values() if not (args.root / src).is_file()]
    if missing:
        parser.error("missing outputs (run the earlier days first): " + ", ".join(missing))
    conflicts = []
    for name, src in FILES.items():
        dest = args.out / name
        if dest.exists() and sha256(dest) != sha256(args.root / src):
            conflicts.append(name)
    if conflicts:
        parser.error(f"evidence files already exist with other content: {conflicts}; "
                     "evidence is frozen, find out which run is right before replacing")
    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}
    copied = same = 0
    for name, src in FILES.items():
        dest = args.out / name
        if dest.exists():
            same += 1
        else:
            shutil.copy2(args.root / src, dest)
            copied += 1
        entries[name] = {"source": src, "sha256": sha256(dest)}
    (args.out / MANIFEST).write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": "copies of gitignored outputs; byte-identical to their sources",
        "files": entries}, indent=2) + "\n", encoding="utf-8")
    print(f"{len(FILES)} evidence files in {args.out}: {copied} copied, {same} already there "
          f"and identical; wrote {args.out / MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
