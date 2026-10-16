"""V4 runtime report: pairing U_hat(t) with U(t+h) and the summary counts."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.analysis.forecast_runtime_report import load_rows, plot, summarize

T0 = datetime(2026, 10, 17, tzinfo=timezone.utc)


def prediction(second: int, u: float, *, ready: bool = True, ood: bool = False) -> dict:
    base = {"snapshot_id": f"s{second}", "controller_id": "c1", "current_utilization": u,
            "horizon_seconds": 3.0, "inference_time_ms": 0.5 + second / 100,
            "quality_score": 0.95, "ood_fallback": ood, "ood_features": ["x"] if ood else []}
    if not ready:
        return {**base, "status": "NOT_READY", "reason": "INSUFFICIENT_HISTORY",
                "inference_time_ms": None}
    upper = u + 0.1 + 0.05
    return {**base, "status": "READY", "reason": None, "predicted_utilization": u + 0.1,
            "upper_utilization_bound": upper, "overload_risk": upper >= 1.0}


def write_run(directory: Path) -> Path:
    path = directory / "run.jsonl"
    us = [0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 0.9]
    with path.open("w", encoding="utf-8") as handle:
        for second, u in enumerate(us):
            payload = {"status": "OK", "snapshot_id": f"s{second}",
                       "created_at": (T0 + timedelta(seconds=second)).isoformat(),
                       "service_time_ms": 1.0 + second,
                       "controllers": [prediction(second, u, ready=second >= 2, ood=second == 4)]}
            handle.write(json.dumps(payload) + "\n")
    return path


class RuntimeReportTest(unittest.TestCase):
    def test_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = load_rows(write_run(Path(tmp)))
            summary = summarize("run", rows)
            plot("run", rows, Path(tmp) / "fig.png")
            self.assertTrue((Path(tmp) / "fig.png").exists())
        self.assertEqual(summary["predictions"], 10)
        self.assertAlmostEqual(summary["ready_ratio"], 0.8)
        self.assertEqual(summary["not_ready_reasons"], {"INSUFFICIENT_HISTORY": 2})
        self.assertAlmostEqual(summary["ood_fallback_ratio"], 1 / 8)
        # READY at t=2..6 have an observation at t+3 (t=5..9)
        self.assertEqual(summary["matched_pairs"], 5)
        # t=4: upper 0.85 vs U(7)=1.1 -> miss; t=5: upper 0.95 vs 1.2 -> miss
        # t=6: upper 1.05 (alert) vs 0.9 -> false alarm
        self.assertEqual(summary["actual_overloads"], 2)
        self.assertEqual(summary["overload_misses"], 2)
        self.assertEqual(summary["false_alarms"], 1)
        self.assertEqual(summary["service_ms_p95"], 10.0)
        self.assertEqual(summary["overload_alerts"], 4)  # t=6,7,8,9


if __name__ == "__main__":
    unittest.main()
