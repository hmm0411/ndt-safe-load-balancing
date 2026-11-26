"""Rollback when the destination dies mid-transaction (fault D6, 26/11)."""
from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import requests
from fastapi import HTTPException

from src.execution.migration import ControllerAPIError
from src.execution.verification import MigrationVerifier

DPID = 7


class FakeClient:
    def __init__(self, roles: dict[str, Any], connected: dict[str, bool]) -> None:
        self.roles, self.connected = roles, connected

    def _check(self, cid: str) -> None:
        value = self.roles[cid]
        if isinstance(value, Exception):
            raise value

    def query_role(self, cid: str, dpid: int, timeout: float) -> dict[str, Any]:
        self._check(cid)
        return {"reply_role": self.roles[cid]}

    def state(self, cid: str) -> dict[str, Any]:
        self._check(cid)
        return {"switches": [{"dpid": DPID, "connected": self.connected[cid]}]}


def verify(roles: dict[str, Any], connected: dict[str, bool] | None = None) -> Any:
    client = FakeClient(roles, connected or {"c1": True, "c3": True})
    return MigrationVerifier(client, 3.0).verify_rollback(DPID, "c1", "c3")


class VerifyRollbackTest(unittest.TestCase):
    def test_normal_rollback(self) -> None:
        result = verify({"c1": "MASTER", "c3": "SLAVE"})
        self.assertTrue(result.ok)
        self.assertEqual(result.warnings, [])

    def test_dead_or_hung_destination_is_a_warning(self) -> None:
        for exc in (requests.exceptions.ConnectionError("refused"),
                    requests.exceptions.ReadTimeout("hung")):
            result = verify({"c1": "MASTER", "c3": exc})
            self.assertTrue(result.ok, exc)
            self.assertEqual(result.target_role, "UNREACHABLE")
            self.assertTrue(result.warnings[0].startswith("other_controller_unreachable:"))

    def test_source_must_still_be_master_and_connected(self) -> None:
        dead = requests.exceptions.ConnectionError("refused")
        self.assertFalse(verify({"c1": "SLAVE", "c3": dead}).ok)
        self.assertFalse(verify({"c1": "MASTER", "c3": dead}, {"c1": False, "c3": False}).ok)

    def test_reachable_destination_still_checked(self) -> None:
        self.assertEqual(verify({"c1": "MASTER", "c3": "MASTER"}).reasons,
                         ["other_controller_not_SLAVE"])
        self.assertEqual(verify({"c1": "MASTER", "c3": "SLAVE"}, {"c1": True, "c3": False}).reasons,
                         ["other_controller_disconnected"])

    def test_switch_reply_timeout_is_not_swallowed(self) -> None:
        with self.assertRaises(ControllerAPIError):
            verify({"c1": "MASTER", "c3": ControllerAPIError("timeout waiting for reply")})


class RollbackFailureReleasesLockTest(unittest.TestCase):
    def test_failed_rollback_finishes_the_transaction(self) -> None:
        from src.orchestrator import app as orch

        switch = sorted(orch.switch_dpids)[0]
        source = orch.ownership_manager.get_owner(switch)
        target = next(c for c in orch.CONTROLLERS if c != source)
        with mock.patch.object(orch.migration_executor, "promote_target",
                               side_effect=RuntimeError("target died")), \
                mock.patch.object(orch.rollback_executor, "restore_source",
                                  side_effect=RuntimeError("source unreachable")):
            with self.assertRaises(HTTPException) as ctx:
                orch.migrate(orch.MigrationRequest(switch_id=switch, target_controller=target))
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn(switch, orch.transaction_manager.locked_switches())
        self.assertEqual(orch.ownership_manager.get_owner(switch), source)


if __name__ == "__main__":
    unittest.main()
