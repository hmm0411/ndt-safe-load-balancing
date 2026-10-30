"""forecast-v1 on raw-v2 under the capacity-v4 scale (Week 12, Day 2).

raw-v2 was collected while the orchestrator ran forecast-v1 live on U computed
with capacity-v4 (933 pps); forecast_samples.jsonl keeps every prediction. The
target of a prediction made at t is the measured U of the same controller at
t + 3 s. The metric is the one of skill_vs_persistence.json: RMSE of the model
divided by RMSE of persistence (below 1 = better than persistence).
Samples with ood_fallback are persistence by construction and are reported apart.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import math
import os
from typing import Any, NamedTuple

HORIZON_S = 3.0
TOLERANCE_S = 0.6
U_BINS = ((0.0, 0.6), (0.6, 0.9), (0.9, 99.0))


class Row(NamedTuple):
    workload: str
    kind: str
    cur: float
    pred: float
    act: float
    ood: bool


def _ts(value: str) -> float:
    return dt.datetime.fromisoformat(value).timestamp()


def load(root: str) -> tuple[list[Row], collections.Counter[str], collections.Counter[str]]:
    status: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    rows: list[Row] = []
    for attempt in sorted(glob.glob(os.path.join(root, "*/"))):
        name = os.path.basename(os.path.dirname(attempt))
        workload = name.split("-")[1]
        kind = "NO_OP" if name.startswith("noop-") else "MIGRATE"
        path = os.path.join(attempt, "forecast_samples.jsonl")
        if not os.path.isfile(path):
            continue
        series: dict[str, list[tuple[float, float, float, bool]]] = collections.defaultdict(list)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                payload = record["payload"]
                for ctrl in payload.get("controllers", []):
                    status[str(ctrl.get("status"))] += 1
                    if ctrl.get("status") != "READY":
                        reasons[str(ctrl.get("reason"))] += 1
                        continue
                    series[ctrl["controller_id"]].append(
                        (
                            _ts(record["observed_at"]),
                            float(ctrl["current_utilization"]),
                            float(ctrl["predicted_utilization"]),
                            bool(ctrl.get("ood_fallback")),
                        )
                    )
        for samples in series.values():
            samples.sort()
            for t, cur, pred, ood in samples:
                near = [s for s in samples if abs(s[0] - (t + HORIZON_S)) <= TOLERANCE_S]
                if not near:
                    continue
                target = min(near, key=lambda s: abs(s[0] - (t + HORIZON_S)))
                rows.append(Row(workload, kind, cur, pred, target[1], ood))
    return rows, status, reasons


def stats(rows: list[Row]) -> dict[str, Any] | None:
    n = len(rows)
    if n == 0:
        return None
    rmse_m = math.sqrt(sum((r.pred - r.act) ** 2 for r in rows) / n)
    rmse_p = math.sqrt(sum((r.cur - r.act) ** 2 for r in rows) / n)
    return {
        "n": n,
        "rmse_model": rmse_m,
        "rmse_persistence": rmse_p,
        "rmse_ratio": rmse_m / rmse_p if rmse_p else float("nan"),
        "mae_ratio": (sum(abs(r.pred - r.act) for r in rows)
                      / max(sum(abs(r.cur - r.act) for r in rows), 1e-12)),
        "bias": sum(r.pred - r.act for r in rows) / n,
    }


def show(name: str, rows: list[Row], extra: str = "") -> dict[str, Any] | None:
    s = stats(rows)
    if s is None:
        print(f"{name:<22} n=0")
        return None
    print(
        f"{name:<22} n={s['n']:6d}  RMSE ratio {s['rmse_ratio']:.2f}  "
        f"MAE ratio {s['mae_ratio']:.2f}  bias {s['bias']:+.4f}{extra}"
    )
    return s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/experiment_runs/migration-outcome-v2")
    parser.add_argument(
        "--skill", default="data/models/forecasting/forecast-v1/skill_vs_persistence.json"
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    rows, status, reasons = load(args.root)
    if not status:
        raise SystemExit(f"no forecast_samples.jsonl found under {args.root}")
    skill = json.load(open(args.skill, encoding="utf-8"))
    original: dict[str, float] = {}
    for split in ("test", "unseen"):
        for wl, models in skill.get(split, {}).items():
            original[wl.replace("-", "_")] = float(models["xgboost"])

    print("status:", dict(status))
    print("NOT_READY reasons:", dict(reasons))
    model_only = [r for r in rows if not r.ood]
    ood_only = [r for r in rows if r.ood]
    print(f"READY with a measured target: {len(rows)} (ood_fallback {len(ood_only)})")
    result: dict[str, Any] = {"status": dict(status), "reasons": dict(reasons)}
    print("--- all READY | model really predicting | ood_fallback (= persistence)")
    result["all"] = show("all", rows)
    result["model_only"] = show("model-only", model_only)
    result["ood_only"] = show("ood", ood_only)
    print("--- model-only by U (measured U at prediction time)")
    result["by_u"] = {}
    for lo, hi in U_BINS:
        subset = [r for r in model_only if lo <= r.cur < hi]
        result["by_u"][f"{lo}-{hi}"] = show(f"U {lo:.1f}-{min(hi, 9.9):.1f}", subset)
    print("--- model-only by workload (original = XGBoost RMSE ratio of forecast-v1 test)")
    result["by_workload"] = {}
    for wl in sorted({r.workload for r in model_only}):
        subset = [r for r in model_only if r.workload == wl]
        ref = original.get(wl)
        extra = f"  original {ref:.2f}" if ref is not None else "  original n/a"
        result["by_workload"][wl] = show(wl, subset, extra)
    print("--- model-only by kind")
    result["by_kind"] = {}
    for kind in ("MIGRATE", "NO_OP"):
        result["by_kind"][kind] = show(kind, [r for r in model_only if r.kind == kind])
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2, sort_keys=True)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()
