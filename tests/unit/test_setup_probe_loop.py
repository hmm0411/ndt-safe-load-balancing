import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.experiments.workloads.setup_probe_loop import PushLoop

T0 = datetime(2026, 11, 17, 10, 0, 0, tzinfo=timezone.utc).timestamp()


def iso(offset: float) -> str:
    return datetime.fromtimestamp(T0 + offset, timezone.utc).isoformat()


def row(offset: float, latency: float) -> str:
    data = {"observed_at": iso(offset), "success": True, "flow_setup_latency_ms": latency}
    return json.dumps(data)


MIGRATION = {
    "transaction_id": "tx-1",
    "switch_id": "s1",
    "target_controller": "c2",
    "state": "COMMITTED",
    "history": [{"state": "VERIFYING", "at": iso(5)}],
}


class PushLoopTest(unittest.TestCase):
    def run_cycle(
        self, rows: list[str], transactions: list[dict[str, Any]], since: float
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        posted: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "qos_s1.jsonl").write_text("\n".join(rows) + "\n")
            loop = PushLoop(Path(tmp), {"s1": "c1", "s2": "c2"}, since, 10.0, 1.0)
            summary = loop.cycle(transactions, posted.append, T0 + 10)
        return summary, posted

    def test_a_sample_after_the_migration_counts_for_the_new_owner(self) -> None:
        summary, posted = self.run_cycle(
            [row(1, 3.0), row(6, 9.0)], [MIGRATION], T0
        )
        self.assertEqual(summary["stats"], {"c1": [3.0, 1, 0], "c2": [9.0, 1, 0]})
        self.assertEqual(sorted(posted[0]["controllers"]), ["c1", "c2"])

    def test_no_sample_means_an_empty_post_not_zeros(self) -> None:
        summary, posted = self.run_cycle([row(1, 3.0)], [], T0 + 100)
        self.assertEqual(summary["stats"], {})
        self.assertEqual(posted[0], {"controllers": {}})
        self.assertEqual(summary["unknown_owner"], 1)


if __name__ == "__main__":
    unittest.main()
