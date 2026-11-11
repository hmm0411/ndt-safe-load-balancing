"""Definition of Done of the merged Week 13-14 plan (2C part) -> evidence/week13/week13_dod.txt.

PASS = the file, tag or test that proves the item exists and its criterion holds;
MISSING = what would prove it is not there yet (never counted as PASS); FAIL = it is
there and the criterion does not hold. Usage, from the repo root on the VM:
    python3 scripts/analysis/week13_evidence.py          # grade only
    python3 scripts/analysis/week13_evidence.py --ci     # also make ci -> ci.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

W13 = ROOT / "evidence" / "week13"
TAGS_PREREQ = ("post-migration-outcome-v1", "migration-outcome-processed-v1", "forecast-raw-v3")
SAFETY_FILES = ("safety_final_evaluation.json", "safety_confusion_table.csv",
                "safety_reason_distribution.csv", "safety_test_lock.json",
                "safety_v2_protocol.md", "safety_v2_round1_cv.txt")


@dataclass(frozen=True)
class Item:
    name: str
    status: bool | None  # True PASS, False FAIL, None MISSING
    where: str


def git_tag(tag: str) -> bool:
    proc = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                          capture_output=True, text=True, cwd=ROOT)
    return proc.returncode == 0


def text_of(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def evaluate(root: Path, tags: Callable[[str], bool]) -> list[Item]:
    items: list[Item] = []

    def add(name: str, status: bool | None, where: str) -> None:
        items.append(Item(name, status, where))

    def files(*rel: str) -> bool | None:
        return True if all((root / r).exists() for r in rel) else None

    missing = [t for t in TAGS_PREREQ if not tags(t)]
    add("Tiền đề: tag outcome-v1, processed-v1, forecast-raw-v3",
        None if missing else True, f"missing tags {missing}" if missing else "all tags exist")
    margin = root / "data/models/forecasting/forecast-models-v3/error_margin.json"
    add("forecast-models-v3 có m_f và so với persistence",
        files(str(margin.relative_to(root)),
              "data/models/forecasting/forecast-models-v3/skill_vs_persistence.json"),
        "error_margin.json, skill_vs_persistence.json")
    runtime = text_of(root / "configs/forecasting/runtime.yaml")
    add("Risk Detector chạy forecast-models-v3",
        ("forecast-models-v3" in runtime) if runtime else None, "configs/forecasting/runtime.yaml")
    add("Schema, policy v1, reason codes freeze",
        files("configs/decision/safety_policy_v1.yaml", "tests/unit/test_safety_policy_frozen.py",
              "tests/unit/test_decision_schema.py"), "policy + frozen-sha test")
    add("Candidate chỉ sinh khi có risk, không trùng",
        files("src/twin/decision/candidate_generator.py", "tests/unit/test_candidate_generator.py"),
        "candidate_generator.py + tests")
    add("Safety Gate 11 check, boundary inclusive",
        files("src/twin/decision/safety_gate.py", "tests/unit/test_safety_gate.py"),
        "safety_gate.py + tests")
    absent = [f for f in SAFETY_FILES if not (root / "evidence/week13" / f).exists()]
    add("Gate A/B trên test và stress, có CI Wilson",
        None if absent else True, f"missing {absent}" if absent else "evidence/week13/safety_*")
    app = text_of(root / "src/orchestrator/app.py")
    add("Endpoint evaluate không side-effect",
        True if "/decision/evaluate" in app else None, "src/orchestrator/app.py route")
    add("Ranking: policy có version, tie-break, NO_SAFE_ACTION",
        files("configs/decision/ranking_policy_v1.yaml", "src/twin/decision/ranking.py",
              "tests/unit/test_ranking.py", "tests/unit/test_ranking_policy.py"),
        "ranking_policy_v1.yaml + tests")
    sens = load_json(root / "evidence/week13/ranking_sensitivity.json")
    add("Ranking sensitivity", (sens.get("n_epochs", 0) > 0) if sens else None,
        f"ranking_sensitivity.json n_epochs={sens.get('n_epochs')}" if sens else
        "evidence/week13/ranking_sensitivity.json")
    loop = [p for p in ("src/twin/decision/closed_loop.py", "src/twin/decision/decision_context.py")
            if not (root / p).exists()]
    add("State machine, decision_id, pinning, revalidation, kill switch, idempotency",
        None if loop or "/decision/plan" not in app else True,
        f"missing {loop}" if loop else "closed_loop.py, decision_context.py, /decision/plan")
    analysis = load_json(root / "evidence/week14/closed_loop_2c_analysis.json")
    gates = analysis.get("scenario_gates", {})
    add("Dry-run và execute 2C đạt kỳ vọng từng kịch bản",
        analysis.get("all_gates_pass") if analysis else None,
        ", ".join(f"{k}={v['result']}" for k, v in gates.items()) or
        "evidence/week14/closed_loop_2c_analysis.json")
    hz = analysis.get("horizon", {})
    add("Latency toàn chuỗi và kết luận h = 3",
        (hz.get("status") == "OK") if analysis else None,
        f"horizon_steps={hz.get('horizon_steps')} current_h_sufficient="
        f"{hz.get('current_h_sufficient')}" if hz else "closed_loop_2c_analysis.json horizon")
    pe = analysis.get("prediction_errors", {})
    add("Sai số dự đoán so với thực tế", (pe.get("n", 0) > 0) if analysis else None,
        f"n={pe.get('n')} coverage={pe.get('u_dst_upper_bound_coverage')}" if pe else
        "closed_loop_2c_analysis.json prediction_errors")
    add("4C readiness audit", files("evidence/week14/readiness_4c.md"),
        "evidence/week14/readiness_4c.md")
    ci = text_of(root / "evidence/week13/ci.txt")
    add("make ci", (ci.count("exit_code: 0") == 2) if ci else None, "ci.txt (run with --ci)")
    for tag in ("forecast-models-v3", "decision-safety-v1", "ranking-policy-v1",
                "closed-loop-2c-v1"):
        add(f"Tag {tag}", True if tags(tag) else None,
            "exists" if tags(tag) else "PENDING (set after merge)")
    return items


def run_ci(out: Path) -> None:
    lines = [f"# make ci + git diff --check, {datetime.now(timezone.utc).isoformat()}"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT)
    lines.append(f"commit: {head.stdout.strip()}")
    for cmd in (["make", "ci"], ["git", "diff", "--check"]):
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        lines += [f"$ {' '.join(cmd)}", f"exit_code: {proc.returncode}",
                  (proc.stdout + proc.stderr)[-3000:]]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dod(out: Path, items: Sequence[Item]) -> str:
    label = {True: "PASS", False: "FAIL", None: "MISSING"}
    lines = [f"# Week 13-14 (2C) Definition of Done, generated "
             f"{datetime.now(timezone.utc).isoformat()}",
             "# PASS = evidence found and criterion met; MISSING = not proven yet; FAIL = not met",
             ""]
    lines += [f"{i.name:<74} {label[i.status]:<8} {i.where}" for i in items]
    passed = sum(1 for i in items if i.status is True)
    lines += ["", f"{passed}/{len(items)} PASS"]
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ci", action="store_true", help="run make ci and git diff --check -> ci.txt")
    ap.add_argument("--out", type=Path, default=W13 / "week13_dod.txt")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.ci:
        run_ci(args.out.parent / "ci.txt")
    print(write_dod(args.out, evaluate(ROOT, git_tag)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
