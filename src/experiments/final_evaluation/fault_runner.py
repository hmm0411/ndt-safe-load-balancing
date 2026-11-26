"""Runs the actions of a D episode (faults.FaultSpec) against the live system.

Everything that touches the system is injected (HTTP, signals, clock), so the sequencing is
unit-tested without a VM. Every action and its result goes to ``log`` (the runner writes
them to <episode>/faults.jsonl, which faults.check_episode reads).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from src.experiments.final_evaluation.faults import FaultAction, FaultSpec
from src.twin.decision.candidate_generator import natural_key

Http = Callable[[str, str, Mapping[str, Any] | None], tuple[int, dict[str, Any]]]
Signal = Callable[[str, str], list[int]]
Log = Callable[[dict[str, Any]], None]


def http_json(base_url: str, timeout: float = 60.0) -> Http:
    """(status, body) of a JSON request; HTTP errors are returned, never raised."""
    def call(method: str, path: str, body: Mapping[str, Any] | None) -> tuple[int, dict[str, Any]]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return int(resp.status), dict(json.loads(resp.read().decode() or "{}"))
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode() or "{}")
            except ValueError:
                payload = {}
            return int(exc.code), payload if isinstance(payload, dict) else {"detail": payload}
    return call


def pick_switch(state: Mapping[str, Any], source: str) -> str:
    """First switch (natural order) owned by ``source`` and not locked by a transaction."""
    ownership = state.get("ownership") or {}
    locked = set(state.get("locked_switches") or [])
    owned = [s for s, owner in ownership.items()
             if (owner.get("owner") if isinstance(owner, Mapping) else owner) == source
             and s not in locked]
    if not owned:
        raise RuntimeError(f"no free switch of {source}")
    return min(owned, key=natural_key)


def owner_of(state: Mapping[str, Any], switch_id: str) -> Any:
    entry = (state.get("ownership") or {}).get(switch_id)
    return entry.get("owner") if isinstance(entry, Mapping) else entry


class FaultRunner:
    def __init__(self, spec: FaultSpec, http: Http, signal: Signal, log: Log,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, settle_s: float = 3.0) -> None:
        self.spec, self.http, self.signal, self.log = spec, http, signal, log
        self.clock, self.sleep, self.settle_s = clock, sleep, settle_s

    def run(self, workload_start: float) -> None:
        """Each action at its time after ``workload_start`` (a clock() value); in order."""
        for action in sorted(self.spec.actions, key=lambda a: a.at_s):
            wait = workload_start + action.at_s - self.clock()
            if wait > 0:
                self.sleep(wait)
            try:
                self._do(action, workload_start)
            except Exception as exc:  # logged, the episode goes on
                self.log({"t": self.clock() - workload_start, "action": "error",
                          "kind": action.kind, "error": str(exc)})

    def _event(self, start: float, **fields: Any) -> None:
        self.log({"t": round(self.clock() - start, 3), **fields})

    def _do(self, action: FaultAction, start: float) -> None:
        p = action.params
        if action.kind == "telemetry":
            code, body = self.http("POST", "/api/v1/faults", dict(p))
            self._event(start, action="telemetry", http_status=code, **dict(p), response=body)
        elif action.kind == "signal":
            pids = self.signal(str(p["controller_id"]), str(p["signal"]))
            self._event(start, action="signal", controller_id=p["controller_id"],
                        signal=p["signal"], pids=pids)
        elif action.kind == "migrate_frozen_target":
            self._migrate_frozen(start, str(p["source"]), str(p["target"]), str(p["then"]),
                                 float(p["after_s"]))
        elif action.kind == "migrate_pair":
            self._migrate_pair(start, str(p["source"]), [str(t) for t in p["targets"]])
        else:
            raise ValueError(f"unknown fault action {action.kind!r}")

    def _migrate(self, start: float, switch: str, source: str, target: str) -> None:
        code, body = self.http("POST", "/api/v1/migrations",
                               {"switch_id": switch, "target_controller": target})
        self._event(start, action="migrate", switch_id=switch, source=source, target=target,
                    http_status=code, status=body.get("status"),
                    detail=body.get("failure_reason") or body.get("detail"))

    def _post_check(self, start: float, switch: str) -> None:
        self.sleep(self.settle_s)
        code, state = self.http("GET", "/api/v1/state", None)
        self._event(start, action="post_check", switch_id=switch, http_status=code,
                    owner=owner_of(state, switch), locked_switches=state.get("locked_switches"))

    def _migrate_frozen(self, start: float, source: str, target: str, then: str,
                        after_s: float) -> None:
        _, state = self.http("GET", "/api/v1/state", None)
        switch = pick_switch(state, source)
        self._event(start, action="signal", controller_id=target, signal="STOP",
                    pids=self.signal(target, "STOP"))
        worker = threading.Thread(target=self._migrate, args=(start, switch, source, target))
        worker.start()
        self.sleep(after_s)
        self._event(start, action="signal", controller_id=target, signal=then,
                    pids=self.signal(target, then))
        worker.join()
        self._post_check(start, switch)

    def _migrate_pair(self, start: float, source: str, targets: list[str]) -> None:
        if len(targets) != 2:
            raise ValueError("migrate_pair needs two targets")
        _, state = self.http("GET", "/api/v1/state", None)
        switch = pick_switch(state, source)
        gate = threading.Barrier(2)

        def one(target: str) -> None:
            gate.wait()
            self._migrate(start, switch, source, target)

        workers = [threading.Thread(target=one, args=(t,)) for t in targets]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        self._post_check(start, switch)
