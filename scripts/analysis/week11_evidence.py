#!/usr/bin/env python3
"""Week 11 evidence package (tuần 11, mục 46 và Definition of Done).

Does not recompute or invent anything. It checks that evidence/week11/ holds the files of
section 46, optionally runs `make ci` and `git diff --check` (ci.txt), and grades the
Week 11 Definition of Done into week11_dod.txt.

    python scripts/analysis/week11_evidence.py              # grade DoD from existing files
    python scripts/analysis/week11_evidence.py --ci         # also run make ci -> ci.txt
    python scripts/analysis/week11_evidence.py --confirm "Item name=where the evidence is"

An item that no file in the repo can prove (Week 10 carry-over, capacity check, off-VM backup,
tag) is MISSING until it is confirmed on the command line, and the confirmation text is printed
in week11_dod.txt. Status: PASS / FAIL / MISSING.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

OUT = Path("evidence/week11")
SCHEMA_VERSION = "migration-outcome-v1"

REQUIRED_FILES = (
    "dataset_card.json",
    "migration_outcome_manifest.json",
    "dataset_quality_report.json",
    "split_manifest.json",
    "attempt_summary.csv",
    "committed_summary.csv",
    "rollback_summary.csv",
    "workload_distribution.csv",
    "migration_latency.json",
    "smoke_success.json",
    "smoke_rollback.json",
    "ci.txt",
    "week11_dod.txt",
)

# Items nothing in the repo can prove; they need --confirm "<name>=<evidence>".
MANUAL_ITEMS = (
    "Nhánh dev có lại trên GitHub, đã bật bảo vệ",
    "Tag forecast-v1 nằm trên main có đủ code Tuần 10",
    "MLflow dựng lại, alias champion đã đặt",
    "R2, R5 trên VM mới đã chạy; DoD Tuần 10 vẫn 36/36",
    "Lưu trữ raw bất biến",
    "Capacity đã kiểm trên VM mới",
    "Dữ liệu thô đã sao lưu ngoài VM, có sha256",
)
TAG = "migration-outcome-raw-v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = read_json(path)
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


def run_ci(out: Path) -> None:
    lines = [f"# make ci + git diff --check, {datetime.now(timezone.utc).isoformat()}"]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    lines.append(f"git_commit: {commit.stdout.strip()}")
    for cmd in (["make", "ci"], ["git", "diff", "--check"]):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines += [f"\n$ {' '.join(cmd)}", proc.stdout[-20000:], proc.stderr[-20000:],
                  f"exit_code: {proc.returncode}"]
    out.joinpath("ci.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tag_on_main(tag: str) -> bool | None:
    """True if the tag is an ancestor of origin/main, None if it does not exist yet."""
    exists = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
                            capture_output=True, text=True)
    if exists.returncode != 0:
        return None
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", tag, "origin/main"],
                         capture_output=True, text=True)
    return anc.returncode == 0


def split_is_by_repeat(split: Mapping[str, Any]) -> bool:
    splits = split.get("splits") or {}
    seen: set[str] = set()
    for body in splits.values():
        repeats = set(body.get("repeats") or [])
        if not repeats or repeats & seen:
            return False
        seen |= repeats
        for attempt_id in body.get("attempts") or []:
            if attempt_id.rsplit("-", 1)[-1] not in repeats:
                return False
    return bool(splits)


def evaluate(out: Path, confirms: Mapping[str, str], tag_state: bool | None
             ) -> list[tuple[str, bool | None, str]]:
    quality = load_json(out / "dataset_quality_report.json")
    manifest = load_json(out / "migration_outcome_manifest.json")
    split = load_json(out / "split_manifest.json")
    ok_smoke = load_json(out / "smoke_success.json")
    bad_smoke = load_json(out / "smoke_rollback.json")
    pilot = load_json(out / "pilot_report.json")
    attempts = csv_header(out / "attempt_summary.csv")
    committed_rows = csv_rows(out / "committed_summary.csv")
    ci_text = out.joinpath("ci.txt").read_text(encoding="utf-8") if (out / "ci.txt").is_file() else ""
    q = quality.get

    def has(*names: str) -> bool:
        return all(n in attempts for n in names)

    def exist(*paths: str) -> bool:
        return all(Path(p).exists() for p in paths)

    def window_ok(sample: Mapping[str, Any]) -> bool:
        return sample.get("count") == 5 and len(sample.get("quality_valid") or []) == 5 \
            and all(sample.get("quality_valid") or [False])

    total = q("total_attempts")
    directions = q("direction_counts") or {}
    levels = q("destination_load_counts") or {}
    clean = (q("invalid_attempts") or []) == [] and q("quarantined") == 0 \
        and q("duplicate_transaction_count") == 0
    committed_n = q("committed_count")
    items: list[tuple[str, bool | None, str]] = []

    def add(name: str, status: bool | None, where: str) -> None:
        items.append((name, status, where))

    def manual(name: str) -> None:
        if name in confirms:
            add(name, True, f"confirmed: {confirms[name]}")
        else:
            add(name, None, "needs --confirm (not provable from files in the repo)")

    for name in MANUAL_ITEMS[:4]:
        manual(name)
    add("Schema State-Action-Outcome",
        exist("src/experiments/migration/schema.py", "tests/unit/test_migration_outcome_schema.py"),
        "src/experiments/migration/schema.py")
    add("Phiên bản schema", q("schema_version") == SCHEMA_VERSION == manifest.get("schema_version"),
        "dataset_quality_report.json, migration_outcome_manifest.json")
    add("Lưu Quality Score", has("before_quality_min", "after_quality_min"), "attempt_summary.csv")
    add("Lưu Consistency Score", has("before_consistency_min", "after_consistency_min"),
        "attempt_summary.csv")
    add("Trạng thái controller nguồn và đích",
        has("src_u_before_mean", "src_u_after_mean", "dst_u_before_mean", "dst_u_after_mean"),
        "attempt_summary.csv")
    add("Thông tin switch được migrate", has("switch_id", "switch_share_before_mean", "target_hot"),
        "attempt_summary.csv")
    add("Hành động migration",
        {"action_type", "source_controller", "target_controller", "switch_id", "failure_injected"}
        <= set(ok_smoke.get("action") or {}), "smoke_success.json")
    add("Lịch sử giao dịch",
        "history" in csv_header(out / "rollback_summary.csv")
        and bool((bad_smoke.get("transaction") or {}).get("body")), "rollback_summary.csv, smoke_rollback.json")
    add("Xử lý COMMITTED", bool(committed_n) and len(committed_rows) == committed_n,
        f"committed_summary.csv ({len(committed_rows)} rows, committed_count={committed_n})")
    add("Xử lý RESTORED",
        (bad_smoke.get("validation") or {}).get("transaction_status") == "RESTORED"
        and exist("tests/unit/test_migration_restored_outcome.py"), "smoke_rollback.json")
    add("Giữ giao dịch lỗi",
        (bad_smoke.get("validation") or {}).get("valid") is True
        and (bad_smoke.get("validation") or {}).get("training_eligible") is False,
        "smoke_rollback.json (valid, not training_eligible)")
    add("Cửa sổ trước (5 snapshot hợp lệ)",
        window_ok(ok_smoke.get("before_samples") or {}) and q("valid_pre_count") == total,
        f"smoke_success.json, valid_pre_count={q('valid_pre_count')}/{total}")
    add("Cửa sổ sau (5 snapshot hợp lệ)",
        window_ok(ok_smoke.get("after_samples") or {}) and q("valid_post_count") == total
        and q("post_window_incomplete_count") == 0,
        f"smoke_success.json, valid_post_count={q('valid_post_count')}/{total}")
    add("Metadata workload", has("workload", "phase", "destination_load_level", "repeat"),
        "attempt_summary.csv")
    add("Cả hai hướng migration", set(directions) == {"c1->c2", "c2->c1"}
        and all(v > 0 for v in directions.values()), f"direction_counts={directions}")
    add("Đa dạng mức tải đích", len(levels) >= 2 and all(v > 0 for v in levels.values()),
        f"destination_load_counts={levels}")
    add("Runner tự động", exist("src/experiments/migration/runner.py",
                                "tests/unit/test_migration_runner.py"),
        "src/experiments/migration/runner.py")
    add("Pilot PASS", bool((pilot.get("gate") or {}).get("passed")), "pilot_report.json (gate.passed)")
    add("Thu chính thức",
        total is not None and total == q("planned_attempts") == q("valid_count")
        and q("missing_attempts") == 0 and q("freeze_consistent") is True,
        f"{q('valid_count')}/{q('planned_attempts')} valid, freeze_consistent={q('freeze_consistent')}")
    manual(MANUAL_ITEMS[4])
    add("Validator cho dataset", exist("src/experiments/migration/validator.py",
                                       "tests/unit/test_migration_outcome_validator.py"),
        "src/experiments/migration/validator.py")
    add("Manifest dataset", bool(manifest.get("attempts")) and len(manifest["attempts"]) == total,
        "migration_outcome_manifest.json")
    add("Quality report", bool(quality), "dataset_quality_report.json")
    add("Chia tập theo repeat", split_is_by_repeat(split), "split_manifest.json")
    add("Thống kê dataset", (out / "dataset_statistics.csv").is_file()
        and (out / "workload_distribution.csv").is_file(), "dataset_statistics.csv")
    add("Không chia ngẫu nhiên theo dòng", split_is_by_repeat(split) and "never by row" in str(split.get("rule")),
        "split_manifest.json (rule + disjoint repeats)")
    add("Không tự bù dữ liệu raw", clean and q("code_bug_attempts") == 0 and q("unplanned_attempts") == 0,
        "dataset_quality_report.json (no invalid/quarantined/duplicate/unplanned)")
    add("make ci và git diff --check sạch", (ci_text.count("exit_code: 0") == 2) if ci_text else None,
        "ci.txt (run with --ci)")
    manual(MANUAL_ITEMS[5])
    add("Attempt dùng lỗi giả lập có failure_injected = true",
        (bad_smoke.get("action") or {}).get("failure_injected") is True, "smoke_rollback.json")
    add(MANUAL_ITEMS[6], MANUAL_ITEMS[6] in confirms or None,
        f"confirmed: {confirms[MANUAL_ITEMS[6]]}" if MANUAL_ITEMS[6] in confirms
        else "needs --confirm (gs://kltn-ndt-backup-2026/week11/ + sha256)")
    add("Có nhóm đối chứng NO_OP", bool(q("no_op_count")) and bool(csv_rows(out / "noop_vs_migrate.csv")),
        f"noop_vs_migrate.csv, no_op_count={q('no_op_count')}")
    tag_label = ("PENDING (tag is set after the DoD and the dev -> main merge)" if tag_state is None
                 else f"tag {TAG} on main" if tag_state else f"tag {TAG} exists but not on main")
    add(f"Tag {TAG} trên main, sau PR dev -> main", tag_state, tag_label)
    return items


def write_dod(out: Path, items: Sequence[tuple[str, bool | None, str]]) -> str:
    width = max(len(n) for n, _, _ in items)
    lines = [f"# Week 11 Definition of Done, generated {datetime.now(timezone.utc).isoformat()}",
             "# PASS = evidence found and criterion met; MISSING = not proven yet; FAIL = not met", ""]
    for name, status, where in items:
        label = "MISSING" if status is None else ("PASS" if status else "FAIL")
        lines.append(f"{name:<{width}}  {label:<7}  {where}")
    passed = sum(1 for _, s, _ in items if s)
    lines += ["", f"{passed}/{len(items)} PASS"]
    text = "\n".join(lines) + "\n"
    out.joinpath("week11_dod.txt").write_text(text, encoding="utf-8")
    return text


def check_files(out: Path) -> list[str]:
    problems = [f"missing file: {n}" for n in REQUIRED_FILES
                if n != "week11_dod.txt" and not (out / n).is_file()]
    if not (out / "figures").is_dir() or not list((out / "figures").glob("*.png")):
        problems.append("missing figures/")
    for path in sorted(out.glob("*.csv")):
        if b"\r" in path.read_bytes():
            problems.append(f"CRLF line endings: {path.name}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ci", action="store_true", help="run make ci and git diff --check -> ci.txt")
    parser.add_argument("--confirm", action="append", default=[], metavar="ITEM=EVIDENCE",
                        help="mark a manual DoD item as done (repeatable)")
    args = parser.parse_args()
    confirms = {}
    for item in args.confirm:
        name, sep, proof = item.partition("=")
        if not sep or not proof.strip():
            parser.error(f"--confirm needs ITEM=EVIDENCE, got {item!r}")
        if name.strip() not in MANUAL_ITEMS:
            parser.error(f"unknown manual item {name.strip()!r}; choose one of {list(MANUAL_ITEMS)}")
        confirms[name.strip()] = proof.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ci:
        run_ci(args.out)
    print(write_dod(args.out, evaluate(args.out, confirms, tag_on_main(TAG))), end="")
    problems = check_files(args.out)
    print("\nFile check (section 46):", "OK" if not problems else "")
    for problem in problems:
        print(f"  - {problem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
