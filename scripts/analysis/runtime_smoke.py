#!/usr/bin/env python3
"""Week 12 runtime smoke summary (evidence/week12/runtime_smoke.json).

Reads the attempts written by `run_migration_outcome.py --predict-url ...` and puts, for each
attempt: the prediction made BEFORE the migration (prediction.json), the value measured AFTER it
(mean over after_samples.jsonl, same definition as src/twin/outcome/targets.py), the error, whether
the measured value lies inside the margin, and the CPU p95 of the controllers during the attempt.

This is a smoke check of the runtime path, NOT an accuracy result: it has 6 attempts.
A prediction that is missing, failed or NOT_READY is written as such, with no numbers and
within_margin = null (never false, never 0).

    python3 scripts/analysis/runtime_smoke.py --root data/experiment_runs/migration-outcome-smoke-w12
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.migration.pilot import percentile  # noqa: E402
from src.experiments.migration.schema import MigrationAction, StateSnapshotView  # noqa: E402
from src.twin.outcome.targets import targets_from_window  # noqa: E402

DEFAULT_ROOT = "data/experiment_runs/migration-outcome-smoke-w12"
DEFAULT_OUT = "evidence/week12/runtime_smoke.json"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_window(path: Path) -> list[StateSnapshotView]:
    return [
        StateSnapshotView.from_twin_state(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def attempt_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def cpu_p95(windows: Sequence[Sequence[StateSnapshotView]], controllers: Sequence[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for cid in controllers:
        values = [s.controller(cid).process_cpu_percent for window in windows for s in window]
        out[cid] = percentile(values, 95)
    return out


def _compare(pred: float | None, observed: float, margin: float | None) -> dict[str, Any]:
    if pred is None:
        return {"predicted": None, "measured": observed, "error": None, "abs_error": None,
                "margin": margin, "within_margin": None}
    error = pred - observed
    inside = None if margin is None else abs(error) <= margin
    return {"predicted": pred, "measured": observed, "error": error, "abs_error": abs(error),
            "margin": margin, "within_margin": inside}


def summarize_attempt(attempt: Path) -> dict[str, Any]:
    action = MigrationAction.from_dict(_read_json(attempt / "action.json"))
    source, target = action.source_controller, action.target_controller
    before = _read_window(attempt / "before_samples.jsonl")
    after = _read_window(attempt / "after_samples.jsonl")
    row: dict[str, Any] = {
        "attempt_id": attempt.name, "switch_id": action.switch_id,
        "source_controller": source, "target_controller": target,
        "cpu_p95_percent": cpu_p95([before, after], [source, target]),
    }
    path = attempt / "prediction.json"
    if not path.exists():
        return {**row, "prediction_status": "NO_PREDICTION_FILE", "reason": "prediction.json not found"}
    record = _read_json(path)
    row["http_status"] = record.get("http_status")
    row["http_ms"] = record.get("http_ms")
    response = record.get("response")
    if "error" in record or not isinstance(response, dict):
        return {**row, "prediction_status": "REQUEST_FAILED", "reason": record.get("error", "no response body")}
    row["prediction_status"] = response.get("status")
    row["reason"] = response.get("reason")
    row["inference_time_ms"] = response.get("inference_time_ms")
    if response.get("status") != "READY":
        return row
    observed = targets_from_window(after, source, target)
    m_u = _number(response.get("dst_utilization_margin"))
    m_r = _number(response.get("response_margin_ms"))
    row["dst_utilization"] = _compare(_number(response.get("predicted_dst_utilization")), observed.u_dst_post, m_u)
    row["src_utilization"] = _compare(_number(response.get("predicted_src_utilization")), observed.u_src_post, m_u)
    row["worst_response_p95_ms"] = _compare(
        _number(response.get("predicted_worst_response_p95_ms")), observed.r95_post, m_r)
    return row


def build(root: Path) -> dict[str, Any]:
    rows = [summarize_attempt(p) for p in attempt_dirs(root)]
    ready = [r for r in rows if r.get("prediction_status") == "READY"]
    judged = [r["dst_utilization"]["within_margin"] for r in ready]
    return {
        "schema_version": "1.0",
        "purpose": "runtime smoke of POST /api/v1/twin/outcome/predict; not an accuracy result",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "margin_rule": "within_margin = |predicted - measured| <= margin; dst U uses m_U, "
                       "R95 uses m_R; src U shown with m_U for reference only",
        "summary": {
            "attempts": len(rows),
            "ready": len(ready),
            "not_ready_or_failed": len(rows) - len(ready),
            "dst_u_inside_margin": sum(1 for x in judged if x is True),
            "dst_u_outside_margin": sum(1 for x in judged if x is False),
        },
        "attempts": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    if not attempt_dirs(root):
        print(f"no attempts under {root}", file=sys.stderr)
        return 2
    result = build(root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    s = result["summary"]
    print(f"{s['attempts']} attempt(s): ready={s['ready']} not_ready_or_failed={s['not_ready_or_failed']} "
          f"dst_u_inside_margin={s['dst_u_inside_margin']} outside={s['dst_u_outside_margin']}")
    for r in result["attempts"]:
        print(f"  {r['attempt_id']}: {r['prediction_status']}"
              + (f" ({r.get('reason')})" if r.get("reason") else ""))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
