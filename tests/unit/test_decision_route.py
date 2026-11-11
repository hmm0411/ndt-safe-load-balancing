from __future__ import annotations

import unittest

from src.orchestrator import app as orchestrator


class DecisionRouteTest(unittest.TestCase):
    def test_route_is_registered_as_post(self) -> None:
        routes = [
            r for r in orchestrator.app.routes
            if getattr(r, "path", "") == "/api/v1/twin/decision/evaluate"
        ]
        self.assertEqual(len(routes), 1)
        self.assertEqual(getattr(routes[0], "methods", None), {"POST"})

    def test_closed_loop_routes_are_registered_and_wired(self) -> None:
        paths = {getattr(r, "path", "") for r in orchestrator.app.routes}
        for name in ("plan", "execute", "kill-switch"):
            self.assertIn(f"/api/v1/twin/decision/{name}", paths)
        service = orchestrator.decision_service
        self.assertIs(service.executor, orchestrator._execute_migration)
        self.assertEqual(service.record_root,
                         orchestrator.ROOT / service.loop_config.record_dir)

    def test_plan_route_executes_nothing(self) -> None:
        transactions = list(orchestrator.transaction_manager.list_all())
        version = orchestrator.ownership_manager.version
        root = orchestrator.decision_service.record_root
        orchestrator.decision_service.record_root = None  # no files from a unit test
        try:
            body = orchestrator.twin_decision_plan(orchestrator.DecisionRequest())
        finally:
            orchestrator.decision_service.record_root = root
        self.assertIn(body["status"], {"NOT_READY", "NO_RISK", "INVALID_SNAPSHOT"})
        self.assertEqual(list(orchestrator.transaction_manager.list_all()), transactions)
        self.assertEqual(orchestrator.ownership_manager.version, version)

    def test_service_is_wired_to_the_live_components(self) -> None:
        service = orchestrator.decision_service
        self.assertIs(service.store, orchestrator.store)
        self.assertIs(service.predictor, orchestrator.outcome_service.predictor)
        self.assertIs(service.transactions, orchestrator.transaction_manager)

    def test_route_has_no_side_effects(self) -> None:
        transactions = list(orchestrator.transaction_manager.list_all())
        ownership = dict(orchestrator.ownership_manager.snapshot_mapping())
        version = orchestrator.ownership_manager.version
        locked = list(orchestrator.transaction_manager.locked_switches())
        body = orchestrator.twin_decision_evaluate()
        self.assertIn(body["status"], {"OK", "NOT_READY"})
        self.assertEqual(list(orchestrator.transaction_manager.list_all()), transactions)
        self.assertEqual(dict(orchestrator.ownership_manager.snapshot_mapping()), ownership)
        self.assertEqual(orchestrator.ownership_manager.version, version)
        self.assertEqual(list(orchestrator.transaction_manager.locked_switches()), locked)


if __name__ == "__main__":
    unittest.main()
