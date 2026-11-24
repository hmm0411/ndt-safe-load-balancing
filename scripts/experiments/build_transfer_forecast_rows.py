#!/usr/bin/env python3
"""Day 5: 4C transfer runs -> forecast rows JSONL with the frozen 2C model's predictions.

    python3 scripts/experiments/build_transfer_forecast_rows.py \
        --runs ~/w15-day34/data/experiment_runs/transfer_4c \
        --capacity data/benchmarks/controller_capacity_v5_noprobe.json
Then evaluate_transfer_4c.py --forecast-rows <out> --outcome-rows <...>.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.final_evaluation import transfer_rows as tr  # noqa: E402
from src.twin.forecasting.predictor import _load_model  # noqa: E402


def check_capacity(run_root: Path, capacity: Path) -> list[str]:
    """Run ids whose metadata names another capacity file than the one evaluated."""
    wrong = []
    for meta_path in sorted(run_root.glob(f"{tr.PREFIX}*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if Path(str(meta.get("capacity_artifact", ""))).name != capacity.name:
            wrong.append(f"{meta_path.parent.name}: {meta.get('capacity_artifact')}")
    return wrong


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs/experiments/transfer_4c_v1.yaml")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--capacity", type=Path, required=True,
                    help="capacity file the runs were made with (checked against metadata)")
    ap.add_argument("--out", type=Path, default=ROOT / "evidence/week15/transfer_4c_forecast_rows.jsonl")
    args = ap.parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    wrong = check_capacity(args.runs, args.capacity)
    if wrong:
        raise SystemExit("runs made with another capacity file:\n  " + "\n  ".join(wrong))
    model_dir = ROOT / config["forecast"]["models"]
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    names = json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8"))[
        "feature_names"]
    model, scaler = _load_model(model_dir, card)
    if model is None:
        raise SystemExit("the deployed artifact is persistence: nothing to transfer")
    rows = tr.feature_rows(tr.load_transfer_samples(args.runs), names, int(card["horizon_steps"]))
    out_rows = tr.predict_rows(rows, names, model, scaler, str(card["output"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    runs = sorted({r["run_id"] for r in out_rows})
    print(f"{len(out_rows)} rows from {len(runs)} runs ({card['model_name']} "
          f"{card['model_version']}, h={card['horizon_steps']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
