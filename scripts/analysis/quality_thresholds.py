"""P95 of age_of_twin_ms and synchronization_jitter_ms over the raw forecast runs.

Used to replace the `good` placeholders in configs/quality.yaml (V1).
Only valid snapshots are counted, one value per snapshot (both controllers of a
snapshot carry the same twin quality).

    python scripts/analysis/quality_thresholds.py --raw-root data/experiment_runs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(0, math.ceil(q / 100.0 * len(ordered)) - 1)
    return ordered[rank]


def collect(raw_root: Path) -> dict[str, list[float]]:
    seen: set[tuple[str, str]] = set()
    out: dict[str, list[float]] = {"age_ms": [], "jitter_ms": []}
    for path in sorted(raw_root.glob("*/forecast_samples.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row.get("run_id")), str(row.get("snapshot_id")))
                if key in seen or not row.get("snapshot_valid", False):
                    continue
                seen.add(key)
                out["age_ms"].append(float(row["age_of_twin_ms"]))
                out["jitter_ms"].append(float(row["synchronization_jitter_ms"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/experiment_runs"))
    parser.add_argument("--output", type=Path, default=Path("results/week10/quality_thresholds.json"))
    args = parser.parse_args()

    values = collect(args.raw_root)
    if not values["age_ms"]:
        print(f"no valid samples under {args.raw_root}")
        return 1
    report = {
        name: {
            "n": len(series),
            "p50": percentile(series, 50),
            "p95": percentile(series, 95),
            "p99": percentile(series, 99),
            "max": max(series),
        }
        for name, series in values.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        "configs/quality.yaml ->"
        f" age_ms.good={report['age_ms']['p95']:.1f}"
        f" jitter_ms.good={report['jitter_ms']['p95']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
