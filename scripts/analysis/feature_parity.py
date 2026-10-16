#!/usr/bin/env python3
"""M4 tier 2: replay recorded runs through the runtime forecaster and compare
its feature vectors with the processed dataset, row by row.

The raw forecast_samples.jsonl of each run was recorded from TwinState by the
data collection. Each recorded instant is rebuilt into a minimal TwinState and
fed to RuntimeForecaster; the vector it builds (on_features hook) must equal
the dataset row with the same (run_id, controller_id, observed_at).

    python scripts/analysis/feature_parity.py --model-version forecast-v1 --split test
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.twin.forecasting.continuity import timestamp_to_seconds
from src.twin.forecasting.loader import load_jsonl
from src.twin.forecasting.predictor import RuntimeForecaster
from src.twin.forecasting.splitter import assign_split
from src.twin.forecasting.versions import MODEL_VERSION, MODELS_ROOT, PROCESSED_ROOT, dataset_version

Key = tuple[str, str, int]  # run_id, controller_id, observed_at in ms

CONTROLLER_FIELDS = (
    "processed_packet_in_rate", "flow_mod_rate", "process_cpu_percent",
    "process_memory_rss_mb", "response_p95_ms", "managed_switch_count",
    "safe_capacity_pps", "utilization",
)


def key_of(run_id: str, controller_id: str, observed_at: Any) -> Key:
    return run_id, controller_id, int(round(timestamp_to_seconds(observed_at) * 1000))


def twin_states_from_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One minimal TwinState per recorded instant (all controllers together).

    The load share is carried by one synthetic switch owned by the controller,
    so the replay also goes through forecast_rows()."""
    by_instant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_instant[str(sample["observed_at"])].append(sample)
    states = []
    for observed_at in sorted(by_instant, key=timestamp_to_seconds):
        group = by_instant[observed_at]
        controllers, switches, ownership = [], [], []
        for s in group:
            cid = str(s["controller_id"])
            controllers.append({"controller_id": cid, **{f: s[f] for f in CONTROLLER_FIELDS}})
            sid = f"replay-{cid}"
            switches.append({"switch_id": sid, "controller_id": cid,
                             "control_load_share": s["max_switch_control_load_share"]})
            ownership.append({"switch_id": sid, "owner_controller_id": cid})
        states.append({
            "snapshot_id": str(group[0].get("snapshot_id") or observed_at),
            "created_at": observed_at,
            "controllers": controllers,
            "switches": switches,
            "ownership": ownership,
            "quality": {"valid": all(bool(s.get("snapshot_valid")) for s in group)},
        })
    return states


def replay_run(samples: list[dict[str, Any]], model_dir: Path) -> dict[Key, list[float]]:
    run_id = str(samples[0]["run_id"])
    vectors: dict[Key, list[float]] = {}

    def hook(cid: str, observed_at: str, x: list[float]) -> None:
        vectors[key_of(run_id, cid, observed_at)] = x

    forecaster = RuntimeForecaster(model_dir, on_features=hook)
    for state in twin_states_from_samples(samples):
        forecaster.update(state)
    return vectors


def compare(
    expected: dict[Key, dict[str, Any]],
    runtime: dict[Key, list[float]],
    feature_names: list[str],
    atol: float = 1e-9,
) -> dict[str, Any]:
    mismatched: list[tuple[Key, str]] = []
    matched = 0
    for key, row in expected.items():
        x = runtime.get(key)
        if x is None:
            continue
        want = np.asarray([float(row[n]) for n in feature_names])
        got = np.asarray(x, dtype=float)
        bad = ~np.isclose(got, want, rtol=0.0, atol=atol)
        if bad.any():
            mismatched.append((key, feature_names[int(np.argmax(bad))]))
        else:
            matched += 1
    return {
        "expected_rows": len(expected),
        "compared_rows": matched + len(mismatched),
        "matched_rows": matched,
        "mismatched_rows": len(mismatched),
        "first_mismatches": [(list(k), f) for k, f in mismatched[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-version", default=MODEL_VERSION)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", type=Path, default=Path("data/experiment_runs"))
    args = parser.parse_args()

    model_dir = MODELS_ROOT / args.model_version
    dataset = PROCESSED_ROOT / dataset_version() / "forecast_dataset.csv"
    with dataset.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if assign_split(r["run_id"]) == args.split]
    expected = {key_of(r["run_id"], r["controller_id"], r["observed_at"]): r for r in rows}
    feature_names = RuntimeForecaster(model_dir).feature_names

    runtime: dict[Key, list[float]] = {}
    for run_id in sorted({r["run_id"] for r in rows}):
        samples = load_jsonl(args.raw_root / run_id / "forecast_samples.jsonl")
        runtime.update(replay_run(samples, model_dir))

    result = compare(expected, runtime, feature_names)
    for name, value in result.items():
        print(f"{name}: {value}")
    coverage = result["compared_rows"] / max(result["expected_rows"], 1)
    print(f"compared/expected: {coverage:.3f}")
    ok = result["mismatched_rows"] == 0 and coverage >= 0.9
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
