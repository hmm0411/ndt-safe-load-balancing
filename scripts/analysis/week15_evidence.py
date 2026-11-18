"""Definition of Done of Week 15 (plan 16-22/11, 4C) -> evidence/week15/week15_dod.txt.

PASS = the file, tag or result that proves the item exists and its criterion holds;
MISSING = not proven yet (never counted as PASS); FAIL = there and the criterion does not
hold. Items follow the plan Day by Day. Usage, from the repo root on the VM:
    python3 scripts/analysis/week15_evidence.py          # grade only
    python3 scripts/analysis/week15_evidence.py --ci     # also make ci -> evidence/week15/ci.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.week13_evidence import Item, git_tag, run_ci, text_of  # noqa: E402

W15 = ROOT / "evidence" / "week15"
CONTROLLERS = ("c1", "c2", "c3", "c4")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def evaluate(root: Path, tags: Callable[[str], bool]) -> list[Item]:
    items: list[Item] = []

    def add(name: str, status: bool | None, where: str) -> None:
        items.append(Item(name, status, where))

    def files(*rel: str) -> bool | None:
        return True if all((root / r).exists() for r in rel) else None

    def missing(*rel: str) -> str:
        absent = [r for r in rel if not (root / r).exists()]
        return f"missing {absent}" if absent else ", ".join(rel)

    # Before the week
    rel: tuple[str, ...] = ("scripts/experiments/derive_controller_capacity.py",
           "tests/unit/test_derive_controller_capacity.py")
    add("Trước tuần: hằng số capacity khớp capacity-v4", files(*rel), missing(*rel))

    # Day 1
    rel = ("configs/controllers_4c40s.yaml", "scripts/start_c3.sh", "scripts/start_c4.sh",
           "src/experiments/topologies/runtime_4c40s.py", "evidence/week15/day1_4c40s.md")
    add("Day 1: hạ tầng 4C40S và collection latency 4C", files(*rel), missing(*rel))
    rel = ("src/experiments/workloads/controller_paths.py", "tests/unit/test_controller_paths.py")
    add("Day 1: runner forecast theo config controller", files(*rel), missing(*rel))
    add("Day 1: nháp evaluation protocol", files("docs/evaluation-protocol-v1.md"),
        missing("docs/evaluation-protocol-v1.md"))

    # Day 2
    rel = ("src/experiments/workloads/setup_probe.py", "tests/unit/test_setup_probe.py",
           "evidence/week15/day2_setup_probe.md")
    add("Day 2: probe một-switch, 3 field trong twin", files(*rel), missing(*rel))
    rel = ("evidence/week15/probe_acceptance.md", "configs/experiments/probe_acceptance_v1.yaml")
    add("Day 2: tiêu chí probe ghi trước khi đo", files(*rel), missing(*rel))
    rel = ("src/experiments/final_evaluation/metrics.py",
           "tests/unit/test_final_evaluation_metrics.py")
    add("Day 2: chỉ số episode (OE, DOR, ...)", files(*rel), missing(*rel))

    # Day 3: the criterion is that the probe was scored; Huong 2 is a valid outcome.
    validation = load_json(root / "evidence/week15/probe_validation.json")
    add("Day 3: probe đã chấm theo P1-P5",
        True if validation.get("decision") else None,
        f"decision={validation.get('decision')} all_pass={validation.get('all_pass')}"
        if validation else "evidence/week15/probe_validation.json")

    # Day 4
    rules = "configs/experiments/capacity_v5_rules.yaml"
    capacity = load_json(root / "data/benchmarks/controller_capacity_v5.json")
    per = capacity.get("controllers", {})
    if capacity:
        complete = all((per.get(c) or {}).get("c_safe_pps") for c in CONTROLLERS)
        safe = ", ".join(f"{c}={per[c]['c_safe_pps']:.1f}" for c in CONTROLLERS if c in per)
        add("Day 4: capacity-v5 có c_safe cho c1-c4 (luật ghi trước)",
            complete and (root / rules).exists(), f"c_safe_pps {safe}")
    else:
        add("Day 4: capacity-v5 có c_safe cho c1-c4 (luật ghi trước)", None,
            "data/benchmarks/controller_capacity_v5.json")
    add("Day 4: bảng run 4C40S", files("data/benchmarks/capacity_runs_4c40s.csv"),
        missing("data/benchmarks/capacity_runs_4c40s.csv"))
    add("Day 4: tag capacity-v5", True if tags("capacity-v5") else None,
        "exists" if tags("capacity-v5") else "PENDING (set after merge)")

    # Day 5
    transfer = load_json(root / "evidence/week15/transfer_4c.json")
    decision = transfer.get("decision", {})
    add("Day 5: thử chuyển giao 2C sang 4C có quyết định",
        True if decision else None,
        f"forecast_reuse={decision.get('forecast_reuse')} outcome_reuse="
        f"{decision.get('outcome_reuse')}" if decision else
        missing("configs/experiments/transfer_4c_v1.yaml", "evidence/week15/transfer_4c.json"))

    # Day 6
    rel = ("configs/experiments/migration_outcome_4c_v1.yaml",)
    add("Day 6: config thu outcome 4C (5 mức tải đích)", files(*rel), missing(*rel))
    targets = text_of(root / "src/twin/outcome/targets.py")
    add("Day 6: imbalance toàn hệ trong targets.py",
        ("others_u_post" in targets) if targets else None, "src/twin/outcome/targets.py")

    # Day 7
    add("Day 7: sha256 của bản sao lưu dữ liệu thô",
        files("evidence/week15/backup_SHA256SUMS"), missing("evidence/week15/backup_SHA256SUMS"))
    ci = text_of(root / "evidence/week15/ci.txt")
    add("Day 7: make ci", (ci.count("exit_code: 0") == 2) if ci else None,
        "evidence/week15/ci.txt (run with --ci)")
    return items


def write_dod(out: Path, items: Sequence[Item]) -> str:
    label = {True: "PASS", False: "FAIL", None: "MISSING"}
    lines = [f"# Week 15 (4C) Definition of Done, generated {datetime.now(timezone.utc).isoformat()}",
             "# PASS = evidence found and criterion met; MISSING = not proven yet; FAIL = not met",
             ""]
    lines += [f"{i.name:<58} {label[i.status]:<8} {i.where}" for i in items]
    passed = sum(1 for i in items if i.status is True)
    lines += ["", f"{passed}/{len(items)} PASS"]
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ci", action="store_true", help="run make ci and git diff --check -> ci.txt")
    ap.add_argument("--out", type=Path, default=W15 / "week15_dod.txt")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.ci:
        run_ci(args.out.parent / "ci.txt")
    print(write_dod(args.out, evaluate(ROOT, git_tag)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
