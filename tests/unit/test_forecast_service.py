"""ForecastService (V3): orchestrator wiring of the runtime forecaster."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.forecast_service import ForecastService
from src.telemetry.writer import JsonlWriter
from src.twin.forecasting.fixtures import twin_state, write_persistence_artifact


class ForecastServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "models" / "forecast-test").mkdir(parents=True)
        write_persistence_artifact(self.root / "models" / "forecast-test")
        self.store = CurrentStateStore()
        self.config: dict[str, Any] = {
            "enabled": True,
            "model_dir": "models/forecast-test",
            "overload_threshold": 1.0,
            "output_dir": "out",
        }

    def service(self, **overrides: Any) -> ForecastService:
        return ForecastService.from_config({**self.config, **overrides}, self.root,
                                           self.store, JsonlWriter())

    def test_disabled(self) -> None:
        self.assertEqual(self.service(enabled=False).payload()["status"], "DISABLED")

    def test_missing_model_keeps_running_and_reports_it(self) -> None:
        service = self.service(model_dir="models/nope")
        body = service.payload()
        self.assertEqual(body["status"], "MODEL_NOT_LOADED")
        self.assertTrue(body["error"])
        service.on_twin_state(twin_state(0))  # no exception
        self.assertIsNone(self.store.forecast_dict())

    def test_not_ready_before_first_snapshot(self) -> None:
        self.assertEqual(self.service().payload()["status"], "NOT_READY")

    def test_ready_payload_and_jsonl(self) -> None:
        service = self.service()
        for s in range(11):
            service.on_twin_state(twin_state(float(s)), run_id="r-test")
        body = service.payload()
        self.assertEqual(body["status"], "OK")
        self.assertEqual(body["snapshot_id"], "snap-10.0")
        self.assertEqual(body["model_version"], "forecast-test")
        self.assertEqual([c["status"] for c in body["controllers"]], ["READY", "READY"])
        self.assertNotIn("target_controller", json.dumps(body))  # no migration fields
        self.assertGreaterEqual(body["service_time_ms"], 0.0)
        lines = (self.root / "out" / "r-test.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 11)
        self.assertEqual(json.loads(lines[0])["controllers"][0]["reason"], "INSUFFICIENT_HISTORY")

    def test_forecaster_error_is_contained(self) -> None:
        service = self.service()
        service.on_twin_state(twin_state(0))
        service.on_twin_state({"snapshot_id": "", "created_at": None})  # broken state
        body = service.payload()
        self.assertEqual(body["status"], "OK")  # last good payload is kept
        self.assertIn("last_error", body)


class OrchestratorRouteTest(unittest.TestCase):
    def test_forecast_route_is_registered(self) -> None:
        from src.orchestrator import app as orchestrator

        paths = {getattr(route, "path", "") for route in orchestrator.app.routes}
        self.assertIn("/api/v1/twin/forecast", paths)
        self.assertIsInstance(orchestrator.forecast_service, ForecastService)


if __name__ == "__main__":
    unittest.main()
