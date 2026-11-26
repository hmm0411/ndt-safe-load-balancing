"""Telemetry faults D1/D2: off by default, bounded, and the store forgets a dropped controller."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.orchestrator.current_state import CurrentStateStore
from src.orchestrator.fault_injection import FaultInjector


class Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FaultInjectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.inj = FaultInjector(True, ["c1", "c2"], clock=self.clock)

    def test_off_refuses_everything(self) -> None:
        off = FaultInjector(False, ["c1"], clock=self.clock)
        with self.assertRaises(PermissionError):
            off.inject("freeze_telemetry", "c1", 10)
        self.assertFalse(off.skip_poll("c1"))

    def test_invalid_requests(self) -> None:
        for kind, cid, seconds in (("reboot", "c1", 5), ("freeze_telemetry", "c9", 5),
                                   ("freeze_telemetry", "c1", 0), ("drop_telemetry", "c1", 601),
                                   ("drop_telemetry", "c1", float("nan"))):
            with self.assertRaises(ValueError):
                self.inj.inject(kind, cid, seconds)

    def test_freeze_skips_polls_until_it_expires(self) -> None:
        self.inj.inject("freeze_telemetry", "c2", 60)
        self.assertTrue(self.inj.skip_poll("c2"))
        self.assertFalse(self.inj.skip_poll("c1"))
        self.assertFalse(self.inj.take_drop("c2"))      # a freeze keeps the old sample
        self.clock.t += 59.9
        self.assertTrue(self.inj.skip_poll("c2"))
        self.clock.t += 0.2
        self.assertFalse(self.inj.skip_poll("c2"))
        self.assertEqual(self.inj.active(), [])

    def test_drop_forgets_once(self) -> None:
        self.inj.inject("drop_telemetry", "c1", 30)
        self.assertTrue(self.inj.take_drop("c1"))
        self.assertFalse(self.inj.take_drop("c1"))
        self.assertTrue(self.inj.skip_poll("c1"))
        self.assertEqual(self.inj.active()[0]["kind"], "drop_telemetry")


class ForgetControllerTest(unittest.TestCase):
    def test_forget_removes_only_that_controller(self) -> None:
        store = CurrentStateStore()
        sw = lambda s, c: SimpleNamespace(switch_id=s, controller_id=c)  # noqa: E731
        store.update_collection("c1", SimpleNamespace(controller_id="c1"),
                                [sw("s1", "c1"), sw("s2", "c1")], {"switches": {}})
        store.update_collection("c2", SimpleNamespace(controller_id="c2"),
                                [sw("s1", "c2")], {"switches": {}})
        store.forget_controller("c1")
        controllers, switches = store.values()
        self.assertEqual([c.controller_id for c in controllers], ["c2"])
        self.assertEqual([(s.switch_id, s.controller_id) for s in switches], [("s1", "c2")])


if __name__ == "__main__":
    unittest.main()
