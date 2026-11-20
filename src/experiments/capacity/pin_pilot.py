"""Score the CPU-pinning pilot (configs/experiments/cpu_pin_pilot_v1.yaml).

Reads the run directories of one experiment: metadata.json (target rate), validation.json
(valid), setup_probe_summary.json (per-controller setup P95). For every rate it averages each
controller's P95 over the valid repeats, divides by the same controller's mean at the lowest
rate, and applies the rule declared before the runs. Usage, from the repo root:
    python3 -m src.experiments.capacity.pin_pilot [--runs data/experiment_runs]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path("configs/experiments/cpu_pin_pilot_v1.yaml")
CONTROLLERS = ("c1", "c2", "c3", "c4")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def collect(run_root: Path, experiment: str) -> dict[float, list[dict[str, float]]]:
    """rate -> one {controller: setup P95 ms} per valid run."""
    by_rate: dict[float, list[dict[str, float]]] = defaultdict(list)
    for run_dir in sorted(run_root.glob(f"{experiment}-*")):
        meta, valid = _load(run_dir / "metadata.json"), _load(run_dir / "validation.json")
        probe = _load(run_dir / "setup_probe_summary.json")
        if not (meta and valid and probe and valid.get("valid")):
            continue
        p95 = {c: float(probe[c]["setup_p95_ms"]) for c in CONTROLLERS if probe.get(c)}
        if len(p95) == len(CONTROLLERS):
            by_rate[float(meta["target_new_flow_rate"])].append(p95)
    return dict(by_rate)


def score(by_rate: Mapping[float, Sequence[Mapping[str, float]]],
          rule: Mapping[str, Any]) -> dict[str, Any]:
    target = str(rule["target"])
    need = int(rule["repeats_required"])
    rates = sorted(by_rate)
    short = {f"{r:g}": len(by_rate[r]) for r in rates if len(by_rate[r]) < need}
    if len(rates) < 2 or short:
        return {"decision": "incomplete", "repeats_short": short, "rates": [f"{r:g}" for r in rates]}
    mean = {r: {c: statistics.mean(run[c] for run in by_rate[r]) for c in CONTROLLERS}
            for r in rates}
    base = rates[0]
    rows, ok = [], True
    for rate in rates[1:]:
        ratio = {c: mean[rate][c] / mean[base][c] for c in CONTROLLERS}
        others = {c: v for c, v in ratio.items() if c != target}
        within = max(others.values()) <= float(rule["other_max_ratio"])
        first = (not rule.get("target_must_rank_first", True)
                 or ratio[target] > max(others.values()))
        ok &= within and first
        rows.append({"rate": rate, "ratio": ratio, "max_other_ratio": max(others.values()),
                     "others_within_limit": within, "target_ranks_first": first,
                     "pass": within and first})
    return {"decision": rule["on_pass"] if ok else rule["on_fail"], "pass": ok,
            "baseline_rate": base, "mean_p95_ms": {f"{r:g}": mean[r] for r in rates},
            "rows": rows, "repeats": {f"{r:g}": len(by_rate[r]) for r in rates}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="score the CPU-pinning pilot")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--runs", type=Path, default=Path("data/experiment_runs"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    sweep = yaml.safe_load(Path(config["sweep_config"]).read_text(encoding="utf-8"))
    result = score(collect(args.runs, str(sweep["experiment"]["type"])), config["decision"])
    result["config"] = str(args.config)
    out = Path(config["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in result.get("rows", []):
        ratios = " ".join(f"{c}={v:.3f}" for c, v in row["ratio"].items())
        print(f"{row['rate']:g} fps: {ratios} -> {'PASS' if row['pass'] else 'FAIL'}")
    print(f"decision: {result['decision']}  (wrote {out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
