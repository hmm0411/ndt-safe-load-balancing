"""Controller saturation check of a migration-outcome collection (Week 12).

A controller is saturated in a window when the 95th percentile of its process CPU
over that window's snapshots reaches ``SATURATION_CPU_P95`` (95 %, the capacity
health rule). Saturated windows cap the processed Packet-In rate, so U measured
there is a lower bound of the offered load.

Three outputs, reproducing ``evidence/week12/`` of migration-outcome-raw-v2:

* ``attempt_saturation.csv``: per attempt, max CPU p95 over controllers before and
  after the action and the class healthy_both / sat_before_only / sat_after_only /
  sat_both;
* ``per_attempt_cpu.csv``: per attempt and controller, over the before, transition
  and after windows: samples, max U, CPU p95 and max;
* ``cpu_vs_u.txt``: CPU by U bin over every sample (is U = 1 where CPU saturates?).

Raw files are read as written by the runner; a line that is not JSON is skipped.
"""
from __future__ import annotations

import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SATURATION_CPU_P95 = 95.0
HIGH_U = 0.72
U_BIN_EDGES = (0.0, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9, 1.0, 9.0)
SAMPLE_FILES = (
    ("before", "before_samples.jsonl"),
    ("transition", "transition_samples.jsonl"),
    ("after", "after_samples.jsonl"),
)
SATURATION_COLUMNS = (
    "run_id", "kind", "cpu_p95_before_max", "cpu_p95_after_max", "sat_before", "sat_after", "class",
)
PER_ATTEMPT_COLUMNS = ("run_id", "controller", "n", "u_max", "cpu_p95", "cpu_max")


@dataclass(frozen=True)
class CpuSample:
    run_id: str
    phase: str
    controller: str
    utilization: float
    cpu: float


def percentile(values: list[float], p: float) -> float:
    """Linear interpolation between closest ranks (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def controller_records(obj: Any) -> Iterator[dict[str, Any]]:
    """Every dict (at any depth) that carries ``process_cpu_percent``."""
    if isinstance(obj, dict):
        if "process_cpu_percent" in obj:
            yield obj
        for value in obj.values():
            yield from controller_records(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from controller_records(value)


def _json_lines(path: Path) -> Iterator[Any]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def attempt_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    return sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)


def attempt_kind(run_id: str) -> str:
    if run_id.startswith("mig-"):
        return "MIGRATE"
    if run_id.startswith("noop-"):
        return "NO_OP"
    return "OTHER"


def cpu_p95_by_controller(path: Path) -> dict[str, float]:
    by_controller: dict[str, list[float]] = collections.defaultdict(list)
    for record in _json_lines(path):
        for rec in controller_records(record):
            if rec.get("snapshot_valid") is False or rec.get("process_cpu_percent") is None:
                continue
            by_controller[str(rec.get("controller_id", "?"))].append(float(rec["process_cpu_percent"]))
    return {cid: percentile(values, 95) for cid, values in by_controller.items() if values}


def saturation_class(sat_before: bool, sat_after: bool) -> str:
    if sat_before and sat_after:
        return "sat_both"
    if sat_before:
        return "sat_before_only"
    if sat_after:
        return "sat_after_only"
    return "healthy_both"


def attempt_saturation(root: Path) -> list[dict[str, Any]]:
    rows = []
    for attempt in attempt_dirs(root):
        before = cpu_p95_by_controller(attempt / "before_samples.jsonl")
        after = cpu_p95_by_controller(attempt / "after_samples.jsonl")
        sat_before = any(v >= SATURATION_CPU_P95 for v in before.values())
        sat_after = any(v >= SATURATION_CPU_P95 for v in after.values())
        rows.append({
            "run_id": attempt.name,
            "kind": attempt_kind(attempt.name),
            "cpu_p95_before_max": round(max(before.values()), 1) if before else "",
            "cpu_p95_after_max": round(max(after.values()), 1) if after else "",
            "sat_before": sat_before,
            "sat_after": sat_after,
            "class": saturation_class(sat_before, sat_after),
        })
    return rows


def cpu_samples(root: Path) -> tuple[int, list[CpuSample], list[str]]:
    """(attempt directories, usable samples, keys of the first usable record)."""
    attempts = attempt_dirs(root)
    samples: list[CpuSample] = []
    example_keys: list[str] = []
    for attempt in attempts:
        for phase, name in SAMPLE_FILES:
            for record in _json_lines(attempt / name):
                for rec in controller_records(record):
                    if rec.get("snapshot_valid") is False:
                        continue
                    cpu = rec.get("process_cpu_percent")
                    u = rec.get("utilization")
                    if u is None:
                        pps, cap = rec.get("processed_packet_in_rate"), rec.get("safe_capacity_pps")
                        if pps is not None and cap:
                            u = float(pps) / float(cap)
                    if u is None or cpu is None:
                        continue
                    if not example_keys:
                        example_keys = sorted(rec.keys())
                    controller = rec.get("controller_id") or rec.get("controller") or "?"
                    samples.append(CpuSample(attempt.name, phase, str(controller), float(u), float(cpu)))
    return len(attempts), samples, example_keys


def per_attempt_cpu(samples: list[CpuSample]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[CpuSample]] = collections.defaultdict(list)
    for s in samples:
        groups[(s.run_id, s.controller)].append(s)
    rows = []
    for (run_id, controller), group in sorted(groups.items()):
        cpus = [s.cpu for s in group]
        rows.append({
            "run_id": run_id,
            "controller": controller,
            "n": len(group),
            "u_max": max(s.utilization for s in group),
            "cpu_p95": percentile(cpus, 95),
            "cpu_max": max(cpus),
        })
    return rows


def cpu_vs_u_report(n_attempts: int, samples: list[CpuSample], example_keys: list[str],
                    per_attempt: list[dict[str, Any]]) -> list[str]:
    lines = [f"attempt dirs: {n_attempts}", f"example keys: {example_keys}",
             f"usable samples: {len(samples)}", "",
             "U bin        n     CPU median  CPU p95   share CPU>=95%"]
    for lo, hi in zip(U_BIN_EDGES[:-1], U_BIN_EDGES[1:]):
        cpus = [s.cpu for s in samples if lo <= s.utilization < hi]
        if not cpus:
            continue
        share = 100.0 * sum(1 for c in cpus if c >= SATURATION_CPU_P95) / len(cpus)
        lines.append(f"{lo:.2f}-{hi:.2f}  {len(cpus):6d}  {percentile(cpus, 50):8.1f}  "
                     f"{percentile(cpus, 95):8.1f}  {share:8.1f}%")
    phases = dict(collections.Counter(s.phase for s in samples))
    high = [r for r in per_attempt if r["u_max"] >= HIGH_U]
    saturated = [r for r in high if r["cpu_p95"] >= SATURATION_CPU_P95]
    lines += ["", f"samples per phase: {phases}", "",
              f"attempt-controller pairs: {len(per_attempt)}",
              f"with u_max >= {HIGH_U}: {len(high)}; of those CPU p95 >= 95%: {len(saturated)}"]
    return lines


def write_outputs(root: Path, out: Path, report_name: str = "cpu_vs_u.txt") -> dict[str, Any]:
    """Write the three files into ``out``; return the class counts per kind."""
    out.mkdir(parents=True, exist_ok=True)
    saturation = attempt_saturation(root)
    with (out / "attempt_saturation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SATURATION_COLUMNS)
        writer.writerows([row[c] for c in SATURATION_COLUMNS] for row in saturation)

    n_attempts, samples, keys = cpu_samples(root)
    if not samples:
        raise ValueError(f"no usable CPU/U samples under {root}")
    per_attempt = per_attempt_cpu(samples)
    with (out / "per_attempt_cpu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PER_ATTEMPT_COLUMNS)
        for r in per_attempt:
            writer.writerow([r["run_id"], r["controller"], r["n"], round(r["u_max"], 4),
                             round(r["cpu_p95"], 1), round(r["cpu_max"], 1)])
    lines = cpu_vs_u_report(n_attempts, samples, keys, per_attempt)
    lines.append(f"saved: {out / 'per_attempt_cpu.csv'}")
    (out / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = collections.Counter((r["kind"], r["class"]) for r in saturation)
    return {"attempts": len(saturation), "classes": {f"{k} {c}": n for (k, c), n in sorted(counts.items())}}
