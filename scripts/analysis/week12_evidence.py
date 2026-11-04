#!/usr/bin/env python3
"""Week 12 evidence package (tuần 12, mục 46 và Definition of Done §60).

Does not recompute or invent anything. It checks that evidence/week12/ holds the files of
section 46, optionally runs `make ci` and `git diff --check` (ci.txt), and grades the
Week 12 Definition of Done into week12_dod.txt. Status: PASS / FAIL / MISSING.
MISSING = the file or tag that would prove the item is not there yet (never counted as PASS).

    python3 scripts/analysis/week12_evidence.py          # grade DoD from existing files
    python3 scripts/analysis/week12_evidence.py --ci     # also run make ci -> ci.txt
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.orchestrator.outcome_service import capacity_version  # noqa: E402
from src.twin.outcome.predictor import (  # noqa: E402
    ARTIFACT_MISSING,
    PostMigrationOutcomePrediction,
    artifact_problem,
)

OUT = Path("evidence/week12")
ARTIFACT = Path("data/models/outcome/post-migration-v1")
CAPACITY = Path("data/benchmarks/controller_capacity.json")
EXPECTED_ROWS = 300
RAW_TAG, CAPACITY_TAG, FINAL_TAG = "migration-outcome-raw-v2", "capacity-v4", "post-migration-outcome-v1"

REQUIRED_FILES = (
    "processed_dataset_report.json", "feature_schema.json", "target_schema.json",
    "analytical_metrics.csv", "direct_ml_metrics.csv", "hybrid_metrics.csv",
    "model_comparison.csv", "per_workload_metrics.csv",
    "residual_analysis.csv", "ablation.csv", "bootstrap_ci.csv",
    "saturation_sensitivity.csv", "overload_cases.csv",
    "uncertainty_margin.json", "coverage_metrics.json", "inference_latency.json",
    "forecast_scale_check.md", "runtime_smoke.json", "ci.txt", "week12_dod.txt",
)
Item = tuple[str, "bool | None", str]


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def csv_header(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle), [])
    except OSError:
        return []


def git_tag(tag: str) -> tuple[bool, bool]:
    """(tag exists, tag is an ancestor of origin/main)."""
    exists = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                            capture_output=True, text=True).returncode == 0
    if not exists:
        return False, False
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tag, "origin/main"],
                         capture_output=True, text=True)
    return True, anc.returncode == 0


def run_ci(out: Path) -> None:
    lines = [f"# make ci + git diff --check, {datetime.now(timezone.utc).isoformat()}"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    lines.append(f"git_commit: {commit.stdout.strip()}")
    for cmd in (["make", "ci"], ["git", "diff", "--check"]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines += [f"\n$ {' '.join(cmd)}", proc.stdout[-20000:], proc.stderr[-20000:],
                  f"exit_code: {proc.returncode}"]
    out.joinpath("ci.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def processed_rows(report: dict[str, Any]) -> tuple[bool | None, str]:
    if not report:
        return None, "processed_dataset_report.json not found"
    rows, one = report.get("rows"), report.get("one_row_per_attempt")
    return rows == EXPECTED_ROWS and one is True, f"rows={rows}, one_row_per_attempt={one}"


def disjoint_runs(manifest: dict[str, Any]) -> tuple[bool | None, str]:
    parts = [manifest.get(k) for k in ("train_runs", "validation_runs", "test_runs")]
    if not all(isinstance(p, list) and p for p in parts):
        return None, "training_manifest.json: train/validation/test runs are not lists"
    sets = [set(map(str, p)) for p in parts]  # type: ignore[arg-type]
    overlap = (sets[0] & sets[1]) | (sets[0] & sets[2]) | (sets[1] & sets[2])
    return not overlap, f"train={len(sets[0])} validation={len(sets[1])} test={len(sets[2])}, overlap={len(overlap)}"


def evaluate(
    out: Path, artifact: Path, capacity_file: Path,
    tags: Callable[[str], tuple[bool, bool]],
) -> list[Item]:
    items: list[Item] = []

    def add(name: str, status: bool | None, where: str) -> None:
        items.append((name, status, where))

    def exist(*paths: str | Path) -> bool:
        return all(Path(p).exists() for p in paths)

    def have(*names: str) -> bool:
        return all((out / n).is_file() for n in names)

    def proof(names: Sequence[str], criterion: bool) -> bool | None:
        """MISSING (None) while a proof file is absent; otherwise the criterion decides."""
        return criterion if have(*names) else None

    manifest = load_json(artifact / "training_manifest.json")
    schema = load_json(out / "feature_schema.json")
    target_schema = load_json(out / "target_schema.json")
    running = capacity_version(capacity_file) if capacity_file.is_file() else None

    raw_exists, raw_on_main = tags(RAW_TAG)
    dataset_manifest = out / "processed_dataset_manifest.json"
    if not raw_exists:
        add("Raw frozen", None, f"tag {RAW_TAG} does not exist yet")
    elif not dataset_manifest.is_file():
        add("Raw frozen", None, f"tag exists; {dataset_manifest} not found")
    else:
        recorded = load_json(dataset_manifest).get("source_dataset") == RAW_TAG
        add("Raw frozen", raw_on_main and recorded,
            f"tag {RAW_TAG} on main={raw_on_main}; processed_dataset_manifest.source_dataset={recorded}")
    ok, where = processed_rows(load_json(out / "processed_dataset_report.json"))
    add("Processed dataset, một attempt = một dòng", ok, where)
    forbidden = {str(x) for x in schema.get("forbidden") or []}
    prefixes = tuple(str(x) for x in schema.get("forbidden_prefixes") or [])
    features = [str(x) for x in schema.get("model_features") or []]
    clean = bool(features) and not any(f in forbidden or (prefixes and f.startswith(prefixes)) for f in features)
    add("No post leakage", (clean if schema else None) and exist("tests/unit/test_outcome_no_leakage.py"),
        f"test_outcome_no_leakage.py; feature_schema.json ({len(features)} model features, none forbidden)")
    ok, where = disjoint_runs(manifest)
    split = load_json(out / "split_manifest.json")
    clean_split = not split.get("unassigned") and not split.get("attempts_in_more_than_one_split")
    add("Run-level split", (ok and clean_split) if split and ok is not None else None,
        f"split_manifest.json (unassigned/duplicated empty={clean_split}); {where}")
    add("Target schema, feature schema có version",
        bool(schema.get("feature_schema")) and bool(target_schema.get("target_schema"))
        if schema and target_schema else None, "feature_schema.json, target_schema.json")
    add("Analytical baseline", proof(["analytical_metrics.csv"], exist("tests/unit/test_analytical_outcome.py")),
        "analytical_metrics.csv, test_analytical_outcome.py")
    models = [artifact / d for d in ("direct_model", "residual_models", "hybrid_model")]
    add("Direct, Residual, Hybrid",
        proof(["direct_ml_metrics.csv", "hybrid_metrics.csv"], all(m.is_dir() and any(m.iterdir()) for m in models)),
        "direct_ml_metrics.csv, hybrid_metrics.csv, models in the artifact")
    selection, lock = load_json(out / "model_selection.json"), load_json(out / "test_lock.json")
    add("Chọn trên validation, test một lần",
        (bool(selection.get("frozen_sha256")) and lock.get("status") == "completed") if selection and lock else None,
        f"model_selection.json frozen_sha256 set; test_lock.json status={lock.get('status')}")
    add("Metric theo workload, 4 target", proof(["per_workload_metrics.csv", "model_comparison.csv"], True),
        "per_workload_metrics.csv, model_comparison.csv (existence; contents are reviewed by hand)")
    margin = load_json(out / "uncertainty_margin.json")
    nonneg = all(isinstance(margin.get(k), (int, float)) and margin[k] >= 0 for k in ("m_U", "m_R"))
    add("m_U, m_R một phía, test coverage",
        (nonneg and margin.get("split") == "validation") if margin and have("coverage_metrics.json") else None,
        "uncertainty_margin.json (split=validation), coverage_metrics.json")
    latency = load_json(out / "inference_latency.json")
    cands = latency.get("candidates") or {}
    bench = (set(cands) >= {"1", "10"} and (latency.get("repeats") or 0) >= 1000
             and all("mean_ms" in cands[k] and "p95_ms" in cands[k] for k in ("1", "10"))) if latency else None
    add("Inference benchmark", bench, "inference_latency.json (1 and 10 candidates, >= 1000 repeats, mean and P95)")
    reason = artifact_problem(artifact, running)
    add("Artifact, model card đủ §46", None if reason == ARTIFACT_MISSING else reason is None,
        f"{artifact} artifact_problem={reason}; model_card.json={(artifact / 'model_card.json').is_file()}")
    smoke = load_json(out / "runtime_smoke.json").get("summary") or {}
    add("Runtime what-if predictor",
        (smoke.get("attempts") == 6 and smoke.get("ready") == 6 and exist("tests/unit/test_outcome_predictor.py"))
        if smoke else None, f"test_outcome_predictor.py, runtime_smoke.json summary={smoke or 'missing'}")
    add("Không tự migrate",
        exist("tests/unit/test_outcome_service.py")
        and "safe" not in {f.name for f in dataclasses.fields(PostMigrationOutcomePrediction)},
        "test_outcome_service.py (no transaction); PostMigrationOutcomePrediction has no field 'safe'")
    ci_path = out / "ci.txt"
    ci_text = ci_path.read_text(encoding="utf-8") if ci_path.is_file() else ""
    add("make ci", (ci_text.count("exit_code: 0") == 2) if ci_text else None, "ci.txt (run with --ci)")
    cap_exists, _ = tags(CAPACITY_TAG)
    cap_ok = bool(manifest) and running is not None and manifest.get("capacity_version") == running
    add("Tag capacity-v4; capacity_version khớp file đang chạy",
        cap_ok if cap_exists and manifest else None,
        f"tag {CAPACITY_TAG} exists={cap_exists}; artifact {manifest.get('capacity_version')} vs running {running}")
    add("Cờ bão hòa sinh từ script trong repo",
        proof(["attempt_saturation.csv"], exist("scripts/analysis/migration_saturation.py")),
        "scripts/analysis/migration_saturation.py, attempt_saturation.csv")
    text = (out / "forecast_scale_check.md").read_text(encoding="utf-8").lower() \
        if (out / "forecast_scale_check.md").is_file() else ""
    add("forecast_scale_check.md có kết quả và kết luận", ("## result" in text and "## reading" in text) if text else None,
        "forecast_scale_check.md (sections Result and Reading)")
    _, final_on_main = tags(FINAL_TAG)
    final_exists, _ = tags(FINAL_TAG)
    add(f"Tag {FINAL_TAG} trên main, sau PR dev -> main", final_on_main if final_exists else None,
        f"tag {FINAL_TAG} on main" if final_on_main else "PENDING (set after the DoD and the dev -> main merge)")
    return items


def write_dod(out: Path, items: Sequence[Item]) -> str:
    width = max(len(n) for n, _, _ in items)
    lines = [f"# Week 12 Definition of Done, generated {datetime.now(timezone.utc).isoformat()}",
             "# PASS = evidence found and criterion met; MISSING = not proven yet; FAIL = not met", ""]
    for name, status, where in items:
        label = "MISSING" if status is None else ("PASS" if status else "FAIL")
        lines.append(f"{name:<{width}}  {label:<7}  {where}")
    passed = sum(1 for _, s, _ in items if s)
    lines += ["", f"{passed}/{len(items)} PASS"]
    text = "\n".join(lines) + "\n"
    out.joinpath("week12_dod.txt").write_text(text, encoding="utf-8")
    return text


def check_files(out: Path) -> list[str]:
    problems = [f"missing file: {n}" for n in REQUIRED_FILES
                if n != "week12_dod.txt" and not (out / n).is_file()]
    figures = list((out / "figures").glob("fig_w12_*.png")) if (out / "figures").is_dir() else []
    if len(figures) < 8:
        problems.append(f"figures/fig_w12_*.png: {len(figures)} of 8")
    for path in sorted(out.glob("*.csv")):
        if b"\r" in path.read_bytes():
            problems.append(f"CRLF line endings: {path.name}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT)
    parser.add_argument("--capacity-file", type=Path, default=CAPACITY)
    parser.add_argument("--ci", action="store_true", help="run make ci and git diff --check -> ci.txt")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ci:
        run_ci(args.out)
    items = evaluate(args.out, args.artifact_dir, args.capacity_file, git_tag)
    print(write_dod(args.out, items), end="")
    problems = check_files(args.out)
    print("\nFile check (section 46):", "OK" if not problems else "")
    for problem in problems:
        print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
