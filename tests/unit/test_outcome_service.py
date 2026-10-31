"""POST /api/v1/twin/outcome/predict and its service (Week 12, Day 4)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from test_outcome_predictor import CAP, M_R, M_U, _states, _write_artifact

from src.orchestrator import app as orchestrator
from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.outcome_service import OutcomeService, capacity_version
from src.twin.outcome import predictor as p

ROOT = Path(__file__).resolve().parents[2]


class _Holder:
    """What CurrentStateStore keeps: an object with to_dict()."""

    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return self.value


def _service(
    root: Path | None, state_index: int | None = 4, warm: int = 5
) -> tuple[OutcomeService, CurrentStateStore]:
    store = CurrentStateStore()
    predictor = p.PostMigrationOutcomePredictor(root, CAP)
    service = OutcomeService(predictor, store)
    for state in _states()[:warm]:
        service.on_twin_state(state)
    if state_index is not None:
        store.set_twin_state(_Holder(_states()[state_index]))
    return service, store


class CapacityVersionTest(unittest.TestCase):
    def test_format_matches_dataset_and_manifest(self) -> None:
        path = ROOT / "data" / "benchmarks" / "controller_capacity.json"
        self.assertEqual(
            capacity_version(path), "controller_capacity.json@sha256:952c64644b99"
        )


class OutcomeServiceTest(unittest.TestCase):
    def test_no_twin_state_is_not_ready_without_numbers(self) -> None:
        service, _ = _service(None, state_index=None, warm=0)
        body = service.predict("s1", "c1", "c2")
        self.assertEqual((body["status"], body["reason"]), (p.NOT_READY, p.INVALID_SNAPSHOT))
        self.assertIsNone(body["predicted_dst_utilization"])

    def test_missing_artifact_is_not_ready(self) -> None:
        service, _ = _service(None)
        body = service.predict("s1", "c1", "c2")
        self.assertEqual((body["status"], body["reason"]), (p.NOT_READY, p.ARTIFACT_MISSING))

    def test_ready_prediction_has_all_plan_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _service(_write_artifact(Path(tmp)))
            body = service.predict("s1", "c1", "c2")
            self.assertEqual((body["status"], body["reason"]), (p.READY, None))
            self.assertAlmostEqual(body["predicted_src_utilization"], 0.64, places=6)
            self.assertAlmostEqual(body["predicted_dst_utilization"], 0.46, places=6)
            self.assertEqual((body["dst_utilization_margin"], body["response_margin_ms"]), (M_U, M_R))
            self.assertNotIn("safe", body)
            json.dumps(body)  # must be JSON-serializable

    def test_history_comes_from_the_telemetry_loop_not_from_the_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _service(_write_artifact(Path(tmp)), warm=4)
            self.assertEqual(service.predictor.history.size(), 4)
            body = service.predict("s1", "c1", "c2")
            self.assertEqual(body["reason"], p.INSUFFICIENT_HISTORY)
            self.assertEqual(service.predictor.history.size(), 4)  # the request added nothing

    def test_older_state_in_a_request_does_not_clear_the_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, store = _service(_write_artifact(Path(tmp)))
            service.on_twin_state(_states(6)[5])  # telemetry is already one snapshot ahead
            store.set_twin_state(_Holder(_states()[4]))  # request read the older one
            service.predict("s1", "c1", "c2")
            self.assertEqual(service.predictor.history.size(), 6)  # 5 + the newer one
            self.assertTrue(service.predictor.history.ready())

    def test_malformed_action_raises(self) -> None:
        service, _ = _service(None)
        with self.assertRaises(ValueError):
            service.predict("s1", "c1", "c1")
        with self.assertRaises(ValueError):
            service.predict("", "c1", "c2")

    def test_switch_owned_by_other_controller(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service, _ = _service(_write_artifact(Path(tmp)))
            body = service.predict("s1", "c2", "c1")
            self.assertEqual(body["reason"], p.SWITCH_NOT_OWNED_BY_SOURCE)

    def test_bad_snapshot_in_history_feed_is_contained(self) -> None:
        service, _ = _service(None, warm=0)
        service.on_twin_state({"controllers": [], "snapshot_id": "x"})  # no created_at
        service.on_twin_state({"status": "NOT_READY"})
        self.assertEqual(service.predictor.history.size(), 0)

    def test_from_config_without_files_stays_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = CurrentStateStore()
            service = OutcomeService.from_config(
                {"artifact_dir": "no/such/dir", "capacity_file": "no/such.json"}, Path(tmp), store
            )
            self.assertIsNone(service.predictor.running_capacity_version)
            body = service.predict("s1", "c1", "c2")
            self.assertEqual(body["status"], p.NOT_READY)


class OutcomeRouteTest(unittest.TestCase):
    def test_route_is_registered_as_post(self) -> None:
        routes = [
            r for r in orchestrator.app.routes
            if getattr(r, "path", "") == "/api/v1/twin/outcome/predict"
        ]
        self.assertEqual(len(routes), 1)
        self.assertEqual(getattr(routes[0], "methods", None), {"POST"})

    def test_route_has_no_side_effects(self) -> None:
        transactions = list(orchestrator.transaction_manager.list_all())
        ownership = dict(orchestrator.ownership_manager.snapshot_mapping())
        version = orchestrator.ownership_manager.version
        locked = list(orchestrator.transaction_manager.locked_switches())
        body = orchestrator.twin_outcome_predict(
            orchestrator.OutcomePredictRequest(
                switch_id="s1", source_controller="c1", target_controller="c2"
            )
        )
        self.assertEqual(body["status"], p.NOT_READY)  # nothing is trained or running here
        self.assertEqual(list(orchestrator.transaction_manager.list_all()), transactions)
        self.assertEqual(dict(orchestrator.ownership_manager.snapshot_mapping()), ownership)
        self.assertEqual(orchestrator.ownership_manager.version, version)
        self.assertEqual(list(orchestrator.transaction_manager.locked_switches()), locked)

    def test_malformed_action_is_http_400(self) -> None:
        with self.assertRaises(HTTPException) as caught:
            orchestrator.twin_outcome_predict(
                orchestrator.OutcomePredictRequest(
                    switch_id="s1", source_controller="c1", target_controller="c1"
                )
            )
        self.assertEqual(caught.exception.status_code, 400)

    def test_feed_is_wired_into_build_snapshot(self) -> None:
        import inspect

        self.assertIn("outcome_service.on_twin_state", inspect.getsource(orchestrator.build_snapshot))


if __name__ == "__main__":
    unittest.main()
