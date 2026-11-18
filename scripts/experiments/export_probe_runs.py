"""Export Day-3 probe runs (latency, timeouts, ownership, U) for the evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from src.experiments.capacity.aggregator import (
    in_window,
    parse_dt,
    percentile,
    read_jsonl,
)

GROUP_PREFIXES = ("probe-off-4c40s-", "probe-sweep-4c40s-")
BACKGROUND_FPS = 30.0
COLUMNS = [
    "run_id",
    "group",
    "repeat",
    "target_controller",
    "target_rate_fps",
    "controller_id",
    "is_target",
    "offered_rate_fps",
    "controller_samples",
    "processed_rate_mean",
    "processed_rate_p95",
    "handler_response_p95_ms_mean",
    "host_flow_setup_p95_ms",
    "probe_setup_p95_ms",
    "probe_samples",
    "probe_timeouts",
    "owned_switches_min",
    "owned_switches_max",
    "owned_switches_mean",
    "ownership_versions",
    "snapshot_valid_ratio",
    "capacity_pps_provisional",
    "u_provisional_mean",
    "u_provisional_p95",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_capacity(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(name): float(item["c_safe_pps"])
        for name, item in data["controllers"].items()
    }


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def p95_or_none(values: list[float]) -> float | None:
    return percentile(values, 0.95) if values else None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def is_valid(run_dir: Path) -> bool:
    return read_json(run_dir / "validation.json").get("valid") is True


def group_of(run_id: str) -> str:
    return "probe_off" if run_id.startswith("probe-off-") else "probe_sweep"


def repeat_of(run_id: str) -> int:
    return int(run_id.rsplit("-r", 1)[1])


def controller_stats(
    run_dir: Path, controller_id: str, start: Any, end: Any
) -> dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(run_dir / "controllers.jsonl")
        if row.get("controller_id") == controller_id
        and in_window(row, "observed_at", start, end)
    ]
    processed = [float(row["processed_packet_in_rate"]) for row in rows]
    handler = [float(row["response_p95_ms"]) for row in rows]
    return {
        "samples": len(rows),
        "processed_mean": mean_or_none(processed),
        "processed_p95": p95_or_none(processed),
        "handler_mean": mean_or_none(handler),
    }


def ownership_stats(
    snapshots: list[dict[str, Any]], controller_id: str
) -> dict[str, Any]:
    counts = [
        sum(
            1
            for item in snap.get("ownership", [])
            if item.get("owner_controller_id") == controller_id
        )
        for snap in snapshots
    ]
    versions = {snap.get("ownership_version") for snap in snapshots}
    return {
        "min": min(counts) if counts else None,
        "max": max(counts) if counts else None,
        "mean": mean_or_none([float(count) for count in counts]),
        "versions": len(versions) if snapshots else None,
    }


def utilization(rate: float | None, capacity: float) -> float | None:
    return None if rate is None else rate / capacity


def build_rows(run_dir: Path, capacity: dict[str, float]) -> list[dict[str, Any]]:
    meta = read_json(run_dir / "metadata.json")
    start = parse_dt(meta["measurement_started_at"])
    end = parse_dt(meta["measurement_ended_at"])
    target = str(meta["target_controller"])
    rate = float(meta["target_new_flow_rate"])
    summary = read_json(run_dir / "summary.json")
    probe = read_json(run_dir / "setup_probe_summary.json")
    snapshots = [
        row
        for row in read_jsonl(run_dir / "snapshots.jsonl")
        if in_window(row, "created_at", start, end)
    ]
    valid = [row.get("quality", {}).get("valid") is True for row in snapshots]
    run_id = str(meta["run_id"])
    rows = []
    for cid in sorted(capacity):
        stats = controller_stats(run_dir, cid, start, end)
        own = ownership_stats(snapshots, cid)
        item = probe.get(cid, {})
        is_target = cid == target
        rows.append(
            {
                "run_id": run_id,
                "group": group_of(run_id),
                "repeat": repeat_of(run_id),
                "target_controller": target,
                "target_rate_fps": rate,
                "controller_id": cid,
                "is_target": is_target,
                "offered_rate_fps": rate if is_target else BACKGROUND_FPS,
                "controller_samples": stats["samples"],
                "processed_rate_mean": stats["processed_mean"],
                "processed_rate_p95": stats["processed_p95"],
                "handler_response_p95_ms_mean": stats["handler_mean"],
                "host_flow_setup_p95_ms": (
                    summary.get("flow_setup_latency_p95_ms") if is_target else None
                ),
                "probe_setup_p95_ms": item.get("setup_p95_ms"),
                "probe_samples": item.get("samples"),
                "probe_timeouts": item.get("timeouts"),
                "owned_switches_min": own["min"],
                "owned_switches_max": own["max"],
                "owned_switches_mean": own["mean"],
                "ownership_versions": own["versions"],
                "snapshot_valid_ratio": (
                    sum(valid) / len(valid) if valid else None
                ),
                "capacity_pps_provisional": capacity[cid],
                "u_provisional_mean": utilization(
                    stats["processed_mean"], capacity[cid]
                ),
                "u_provisional_p95": utilization(
                    stats["processed_p95"], capacity[cid]
                ),
            }
        )
    return rows


def discover(root: Path) -> tuple[list[Path], list[str]]:
    kept: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith(GROUP_PREFIXES):
            continue
        (kept if is_valid(path) else skipped).append(path)
    return kept, [path.name for path in skipped]


def manifest_entry(run_dir: Path) -> dict[str, Any]:
    names = (
        "metadata.json",
        "summary.json",
        "setup_probe_summary.json",
        "controllers.jsonl",
        "snapshots.jsonl",
        "validation.json",
    )
    files = {
        name: sha256_file(run_dir / name)
        for name in names
        if (run_dir / name).exists()
    }
    return {"run_id": run_dir.name, "sha256": files}


def write_outputs(
    rows: list[dict[str, Any]],
    runs: list[Path],
    skipped: list[str],
    capacity_path: Path,
    output: Path,
    manifest: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "csv": str(output),
        "csv_sha256": sha256_file(output),
        "rows": len(rows),
        "capacity_file": str(capacity_path),
        "capacity_sha256": sha256_file(capacity_path),
        "capacity_note": "provisional (2C-derived); recompute U with capacity-v5",
        "skipped_invalid_runs": skipped,
        "runs": [manifest_entry(run) for run in runs],
    }
    manifest.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("data/experiment_runs"))
    parser.add_argument("--capacity", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/week15/day3_probe_runs_4c40s.csv")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evidence/week15/day3_probe_manifest.json"),
    )
    parser.add_argument("--expect-runs", type=int, default=None)
    args = parser.parse_args()
    capacity = load_capacity(args.capacity)
    runs, skipped = discover(args.runs_root)
    if args.expect_runs is not None and len(runs) != args.expect_runs:
        print(f"expected {args.expect_runs} valid runs, found {len(runs)}")
        return 1
    rows = [row for run in runs for row in build_rows(run, capacity)]
    write_outputs(rows, runs, skipped, args.capacity, args.output, args.manifest)
    print(f"runs={len(runs)} skipped_invalid={len(skipped)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
