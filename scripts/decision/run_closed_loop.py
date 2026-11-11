"""Drive the closed loop against a running orchestrator (Week 13, Day 6; VM only).

Once per interval it POSTs /api/v1/twin/decision/plan (dry run) or /execute and counts
the statuses. The workload is started separately (see docs/closed-loop-2c.md): this
script only asks for decisions, it never starts Mininet and never needs sudo.

  python3 scripts/decision/run_closed_loop.py --scenario gradual --mode dry_run --duration 60
  python3 scripts/decision/run_closed_loop.py --scenario gradual --mode execute \
      --duration 90 --after-commit cooldown --wait-post 50
  python3 scripts/decision/run_closed_loop.py --scenario conflict --mode dry_run \
      --duration 6 --interval 0.1 --conflict-migration s1:c2
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from typing import Any

SCENARIOS = ("stable", "gradual", "hot_switch", "verification_failure", "no_safe_action",
             "dst_near_capacity", "invalid_snapshot", "cooldown", "conflict")
Post = Callable[[str, dict[str, Any]], dict[str, Any]]


def http_post(base_url: str, timeout: float = 30.0) -> Post:
    def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            base_url.rstrip("/") + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return dict(json.loads(response.read().decode()))
        except urllib.error.HTTPError as exc:
            return {"status": f"HTTP_{exc.code}", "detail": exc.read().decode()[:300]}
        except (urllib.error.URLError, TimeoutError) as exc:
            return {"status": "UNREACHABLE", "detail": str(exc)}
    return post


def run(post: Post, scenario: str, mode: str, duration: float, interval: float,
        simulate_failure: str = "none", after_commit: str | None = None,
        max_commits: int = 1, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print) -> dict[str, Any]:
    """Ask for one decision per interval; returns the status counts per scenario label."""
    path = "/api/v1/twin/decision/" + ("execute" if mode == "execute" else "plan")
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    label, commits, executed = scenario, 0, []
    deadline = clock() + duration
    while clock() < deadline:
        body: dict[str, Any] = {"scenario": label}
        if mode == "execute":
            body["simulate_failure"] = simulate_failure if commits < max_commits else "none"
        out = post(path, body)
        decision_id = str(out.get("decision_id"))
        if decision_id not in seen:  # one count per snapshot, replays are not new decisions
            seen.add(decision_id)
            status = str(out.get("status"))
            counts[f"{label}:{status}"] += 1
            if out.get("transaction"):
                executed.append({"decision_id": decision_id, "status": status,
                                 "selected": (out.get("selected") or {}).get("candidate_id")})
                log(f"[{label}] {status} {executed[-1]['selected']} ({decision_id})")
                commits += 1
                if after_commit:
                    label = after_commit
        sleep(interval)
    return {"scenario": scenario, "mode": mode, "decisions": len(seen),
            "counts": dict(sorted(counts.items())), "executed": executed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--mode", required=True, choices=("dry_run", "execute"))
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--simulate-failure", choices=("none", "flow_mod"), default="none")
    ap.add_argument("--after-commit", choices=SCENARIOS, default=None,
                    help="scenario label of the decisions after the first executed one")
    ap.add_argument("--conflict-migration", metavar="SWITCH:TARGET", default=None,
                    help="start this manual migration in the background (scenario conflict)")
    ap.add_argument("--wait-post", type=float, default=0.0,
                    help="seconds to wait at the end so the observed outcome is recorded")
    ap.add_argument("--orchestrator-url", default="http://127.0.0.1:9000")
    args = ap.parse_args(argv)
    post = http_post(args.orchestrator_url)
    if args.conflict_migration:
        switch, target = args.conflict_migration.split(":", 1)
        body = {"switch_id": switch, "target_controller": target, "simulate_failure": "none"}
        threading.Timer(1.0, lambda: print("[manual migration]",
                                           post("/api/v1/migrations", body).get("status"))).start()
    summary = run(post, args.scenario, args.mode, args.duration, args.interval,
                  args.simulate_failure, args.after_commit)
    if args.wait_post > 0:
        print(f"waiting {args.wait_post:.0f} s for the post-migration window")
        time.sleep(args.wait_post)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
