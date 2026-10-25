#!/usr/bin/env python3
"""Week 11 Data Card (plan section 37): evidence/week11/dataset_card.json.

    python scripts/analysis/migration_dataset_card.py [--evidence evidence/week11]

Reads the matrix config, the card config and the evidence files written by
scripts/analysis/migration_dataset_report.py; never edit the JSON by hand,
change configs/experiments/migration_dataset_card.yaml and run this again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.migration.config import load_outcome_config  # noqa: E402
from src.experiments.migration.dataset_card import (  # noqa: E402
    build_dataset_card,
    load_card_config,
)
from src.experiments.migration.runner import DEFAULT_DURATION_S  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path,
                        default=Path("configs/experiments/migration_outcome.yaml"))
    parser.add_argument("--card-config", type=Path,
                        default=Path("configs/experiments/migration_dataset_card.yaml"))
    parser.add_argument("--evidence", type=Path, default=Path("evidence/week11"))
    args = parser.parse_args()
    card = build_dataset_card(load_outcome_config(args.config), load_card_config(args.card_config),
                              args.evidence, DEFAULT_DURATION_S)
    out = args.evidence / "dataset_card.json"
    out.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    actual = card["actual_attempts"]
    print(f"wrote {out}: {actual['total_attempts']}/{card['planned_attempts']['total']} attempts, "
          f"valid {actual['valid_count']}, COMMITTED {actual['committed_count']}, "
          f"NO_OP {actual['no_op_count']}, overload {actual['destination_overload_after_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
