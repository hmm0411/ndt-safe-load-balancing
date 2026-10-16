#!/usr/bin/env python3
"""Runtime forecast report (Week 10, V4): summary per run and Figure 9.

Input: the JSONL the orchestrator writes for every snapshot when forecasting is
enabled (configs/forecasting/runtime.yaml -> data/raw/forecast/<RUN_ID>.jsonl).

For each run:
  results/week10/runtime/<RUN_ID>_summary.json
      READY ratio, NOT_READY reasons, OOD fallback ratio, inference p50/p95,
      service_ms_p95 (whole forecast step per snapshot -> write_horizon.py),
      overload alerts, and, matched with the utilization observed h seconds
      later: RMSE, coverage of the upper bound, overload hits/misses.
  results/week10/runtime/fig9_<RUN_ID>.png
      per controller: U(t) observed, U_hat placed at t+h, upper bound,
      threshold, alerts and OOD fallback samples.
And results/week10/runtime/runtime_summary.csv with one row per run.

    python scripts/analysis/forecast_runtime_report.py --run-id r1-forecast r3-latency
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THRESHOLD = 1.0


def load_rows(path: Path) -> list[dict[str, Any]]:
    """One row per (snapshot, controller) prediction.

    service_time_ms (whole ForecastService.on_twin_state for the snapshot) is
    copied onto each row of that snapshot.
    """
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            created_at = datetime.fromisoformat(str(payload["created_at"]))
            for item in payload.get("controllers", []):
                rows.append({**item, "created_at": created_at,
                             "service_time_ms": payload.get("service_time_ms")})
    return rows


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100.0 * len(ordered)) - 1)]


def realized(rows: list[dict[str, Any]], tolerance_s: float = 0.5) -> list[tuple[dict, float]]:
    """Pair each READY prediction with the U observed h seconds later."""
    observed: dict[str, list[tuple[datetime, float]]] = {}
    for row in rows:
        if row.get("current_utilization") is not None:
            observed.setdefault(row["controller_id"], []).append(
                (row["created_at"], float(row["current_utilization"]))
            )
    pairs: list[tuple[dict, float]] = []
    for row in rows:
        if row.get("status") != "READY" or row.get("horizon_seconds") is None:
            continue
        target = row["created_at"] + timedelta(seconds=float(row["horizon_seconds"]))
        best = min(
            observed.get(row["controller_id"], []),
            key=lambda item: abs((item[0] - target).total_seconds()),
            default=None,
        )
        if best is not None and abs((best[0] - target).total_seconds()) <= tolerance_s:
            pairs.append((row, best[1]))
    return pairs


def active_window(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows of the workload itself.

    The orchestrator keeps writing (INVALID) snapshots before Mininet is up and
    after it stops, until it is restarted for the next run. Keep the snapshots
    from the first one that is not INVALID_SNAPSHOT (cold start included) to
    the last READY one; invalid snapshots inside the window (telemetry loss,
    migration) stay in.
    """
    order = sorted({r["created_at"] for r in rows})
    first = next(
        (t for t in order if any(r["created_at"] == t and r.get("reason") != "INVALID_SNAPSHOT"
                                 for r in rows)),
        None,
    )
    ready_times = [r["created_at"] for r in rows if r.get("status") == "READY"]
    if first is None or not ready_times:
        return rows
    last = max(ready_times)
    return [r for r in rows if first <= r["created_at"] <= last]


def summarize(run_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_snapshots = len({r["snapshot_id"] for r in rows})
    rows = active_window(rows)
    ready = [r for r in rows if r.get("status") == "READY"]
    # Model latency only: an OOD-fallback sample skips the model (persistence).
    times = [
        float(r["inference_time_ms"])
        for r in ready
        if r.get("inference_time_ms") is not None and not r.get("ood_fallback")
    ]
    pairs = realized(rows)
    errors = [float(r["predicted_utilization"]) - actual for r, actual in pairs]
    covered = [actual <= float(r["upper_utilization_bound"]) for r, actual in pairs]
    actual_over = [(bool(r["overload_risk"]), actual >= THRESHOLD) for r, actual in pairs]
    persistence = [float(r["current_utilization"]) - actual for r, actual in pairs]
    service: dict[str, float] = {}
    for r in rows:
        if r.get("service_time_ms") is not None:
            service[r["snapshot_id"]] = float(r["service_time_ms"])
    quality = [float(r["quality_score"]) for r in rows if r.get("quality_score") is not None]
    return {
        "run_id": run_id,
        "predictions": len(rows),
        "snapshots": len({r["snapshot_id"] for r in rows}),
        "snapshots_in_file": total_snapshots,
        "ready_ratio": len(ready) / len(rows) if rows else None,
        "not_ready_reasons": dict(Counter(r.get("reason") for r in rows if r.get("status") != "READY")),
        "ood_fallback_ratio": (sum(bool(r.get("ood_fallback")) for r in ready) / len(ready)) if ready else None,
        "ood_features": dict(Counter(f for r in ready for f in r.get("ood_features") or [])),
        "overload_alerts": sum(bool(r.get("overload_risk")) for r in ready),
        "inference_ms_p50": percentile(times, 50),
        "inference_ms_p95": percentile(times, 95),
        "inference_ms_max": max(times) if times else None,
        "service_ms_p50": percentile(list(service.values()), 50),
        "service_ms_p95": percentile(list(service.values()), 95),
        "quality_score_min": min(quality) if quality else None,
        "quality_score_p5": percentile(quality, 5),
        "matched_pairs": len(pairs),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else None,
        "rmse_persistence": (
            math.sqrt(sum(e * e for e in persistence) / len(persistence)) if persistence else None
        ),
        "coverage_upper_bound": (sum(covered) / len(covered)) if covered else None,
        "actual_overloads": sum(a for _, a in actual_over),
        "overload_hits": sum(p and a for p, a in actual_over),
        "overload_misses": sum(a and not p for p, a in actual_over),
        "false_alarms": sum(p and not a for p, a in actual_over),
    }


def plot(run_id: str, rows: list[dict[str, Any]], output: Path) -> None:
    controllers = sorted({r["controller_id"] for r in rows})
    if not controllers:
        return
    start = min(r["created_at"] for r in rows)
    fig, axes = plt.subplots(len(controllers), 1, figsize=(11, 3.2 * len(controllers)),
                             sharex=True, squeeze=False, constrained_layout=True)
    for axis, cid in zip(axes[:, 0], controllers):
        mine = sorted((r for r in rows if r["controller_id"] == cid), key=lambda r: r["created_at"])
        seen = [r for r in mine if r.get("current_utilization") is not None]
        ready = [r for r in mine if r.get("status") == "READY"]

        def sec(row: dict, shift: float = 0.0) -> float:
            return (row["created_at"] - start).total_seconds() + shift

        axis.plot([sec(r) for r in seen], [r["current_utilization"] for r in seen],
                  color="black", lw=1.2, label="U(t) observed")
        x_hat = [sec(r, float(r["horizon_seconds"])) for r in ready]
        axis.plot(x_hat, [r["predicted_utilization"] for r in ready],
                  color="tab:blue", lw=1.0, label="U_hat placed at t+h")
        axis.fill_between(x_hat, [r["predicted_utilization"] for r in ready],
                          [r["upper_utilization_bound"] for r in ready],
                          color="tab:blue", alpha=0.2, label="+ m_f")
        axis.axhline(THRESHOLD, color="tab:red", ls="--", lw=1.0, label="threshold 1.0")
        alerts = [r for r in ready if r.get("overload_risk")]
        axis.scatter([sec(r, float(r["horizon_seconds"])) for r in alerts],
                     [r["upper_utilization_bound"] for r in alerts],
                     color="tab:red", s=12, zorder=3, label="overload alert")
        ood = [r for r in ready if r.get("ood_fallback")]
        axis.scatter([sec(r, float(r["horizon_seconds"])) for r in ood],
                     [r["predicted_utilization"] for r in ood],
                     marker="x", color="tab:orange", s=18, zorder=3, label="OOD fallback")
        axis.set_ylabel(f"{cid}  U")
        axis.grid(alpha=0.3)
    axes[0, 0].set_title(f"Figure 9 - runtime forecast, {run_id}")
    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=3)
    axes[-1, 0].set_xlabel("seconds since first snapshot")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/forecast"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/week10/runtime"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for run_id in args.run_id:
        path = args.input_dir / f"{run_id}.jsonl"
        if not path.exists():
            print(f"missing {path}")
            return 1
        rows = load_rows(path)
        summary = summarize(run_id, rows)
        summaries.append(summary)
        (args.output_dir / f"{run_id}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        plot(run_id, rows, args.output_dir / f"fig9_{run_id}.png")
        print(json.dumps(summary, indent=2))

    flat = [k for k, v in summaries[0].items() if not isinstance(v, dict)]
    with (args.output_dir / "runtime_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
