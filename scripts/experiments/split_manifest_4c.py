"""Declare the 4C train/validation/test split by attempt (repeat) and lock the test set."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.migration.config import (  # noqa: E402
    config_sha256, expand_attempts, load_outcome_config)

ARMS = {
    "probe_on": ("configs/experiments/migration_outcome_4c_v1.yaml",
                 "data/experiment_runs/migration-outcome-4c-v1"),
    "probe_off": ("configs/experiments/migration_outcome_4c_noprobe_v1.yaml",
                  "data/experiment_runs/migration-outcome-4c-noprobe-v1"),
}


def on_disk(run: Path) -> set[str]:
    names = {p.name for p in run.glob("*") if p.is_dir()}
    return names | {p.name for p in run.glob("*/*") if p.is_dir()}


def arm(cfg_path: str, run_dir: str) -> dict[str, object]:
    cfg = load_outcome_config(ROOT / cfg_path)
    plan = expand_attempts(cfg)
    disk = on_disk(ROOT / run_dir)
    splits = {}
    for name, repeats in cfg.split.items():
        rows = [a for a in plan if a.split == name]
        splits[name] = {"repeats": list(repeats), "count": len(rows),
                        "migrate": sum(a.action_type == "MIGRATE" for a in rows),
                        "no_op": sum(a.action_type == "NO_OP" for a in rows),
                        "attempts": sorted(a.attempt_id for a in rows)}
    owners: dict[str, int] = {}
    for s in splits.values():
        for a in s["attempts"]:  # type: ignore[attr-defined]
            owners[a] = owners.get(a, 0) + 1
    ids = {a.attempt_id for a in plan}
    test = "\n".join(splits["test"]["attempts"])  # type: ignore[index,arg-type]
    return {"config": cfg_path, "config_sha256": config_sha256(ROOT / cfg_path),
            "planned": len(plan), "measured_on_disk": len(ids & disk),
            "planned_not_on_disk": sorted(ids - disk)[:10], "splits": splits,
            "unassigned": [a.attempt_id for a in plan if not a.split],
            "attempts_in_more_than_one_split": sorted(a for a, n in owners.items() if n > 1),
            "test_set_sha256": hashlib.sha256(test.encode()).hexdigest()}


def main() -> None:
    out = {"rule": "split by repeat (scenario + repeat), never by row; NO_OP follows its repeat",
           "arms": {k: arm(*v) for k, v in ARMS.items()}}
    dest = ROOT / "evidence/week15/split_manifest_4c.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", dest)


if __name__ == "__main__":
    main()
