"""Progress-report figures (16/11/2026): every number is read from files already in the repo.

Inputs (all on dev; a missing one is skipped and reported, never replaced by a constant):
  data/benchmarks/capacity_runs.csv                                 -> fig_01_capacity_latency
  data/models/forecasting/forecast-models-v3/skill_vs_persistence.json -> fig_02_forecast_skill
  data/models/forecasting/forecast-models-v3/coverage.json          \
  evidence/week12/coverage_metrics.json                              -> fig_03_margin_coverage
  data/models/outcome/post-migration-v1/test_metrics.json           -> fig_04_outcome_rmse
  evidence/week13/safety_final_evaluation.json                       -> fig_05_gate_stress
  evidence/week13/safety_v2_round1_cv.txt                            -> fig_06_gate_tradeoff
  evidence/week14/closed_loop_2c_analysis.json                       -> fig_07_closed_loop_migrations
                                                                        fig_08_lead_time
Writes PNG (200 dpi) and PDF of each figure plus figures_data.json (the plotted values and
their source files) to evidence/week15/progress_report/. Flat academic style, same rcParams
as plot_week13.py. Usage, from the repo root:
    python3 scripts/analysis/plot_progress_report.py [--root .] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("evidence/week15/progress_report")
ACCENT, GREY, DARK, RED = "#4C72B0", "#BBBBBB", "#555555", "#C44E52"
WORKLOADS = ("stable", "gradual", "burst", "oscillating", "hot-switch")
# display names of the candidates in safety_v2_round1_cv.txt (v1B = the frozen v1 policy)
CV_NAMES = {"v1B": "v1 (current)", "a": "a: retrained", "b1": "b1: wider margin",
            "b2": "b2: wider margin"}
TARGET_COVERAGE = 0.95  # alpha = 0.05 of every margin (CLAUDE.md section 6)
U_LIMIT = 1.0  # overload threshold U = 1
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 9, "savefig.dpi": 200, "savefig.bbox": "tight"})

SOURCES = {
    "capacity": "data/benchmarks/capacity_runs.csv",
    "skill": "data/models/forecasting/forecast-models-v3/skill_vs_persistence.json",
    "forecast_coverage": "data/models/forecasting/forecast-models-v3/coverage.json",
    "outcome_coverage": "evidence/week12/coverage_metrics.json",
    "outcome_test": "data/models/outcome/post-migration-v1/test_metrics.json",
    "gate_final": "evidence/week13/safety_final_evaluation.json",
    "gate_cv": "evidence/week13/safety_v2_round1_cv.txt",
    "closed_loop": "evidence/week14/closed_loop_2c_analysis.json",
}


def norm_workload(name: str) -> str:
    return name.replace("_", "-")


# ---------------------------------------------------------------- loaders (pure)

def capacity_curves(path: Path) -> list[dict[str, Any]]:
    """Mean over repeats per (controller, rate) of the valid capacity runs.

    setup_p95_ms is None when no QoS probe succeeded (qos_success_ratio == 0): the CSV
    then holds 0.0, which is not a latency.
    """
    groups: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["valid"] == "True":
                groups[(row["controller"], float(row["target_new_flow_rate"]))].append(row)
    out = []
    for (controller, rate), rows in sorted(groups.items()):
        ok = [r for r in rows if float(r["qos_success_ratio"] or 0) > 0]
        setup = [float(r["flow_setup_latency_p95_ms"]) for r in ok]
        resp = [float(r["response_p95_ms_mean"]) for r in rows]
        out.append({"controller": controller, "rate_fps": rate, "repeats": len(rows),
                    "setup_p95_ms": sum(setup) / len(setup) if setup else None,
                    "response_p95_ms": sum(resp) / len(resp)})
    return out


def forecast_skill(report: dict[str, Any], models: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for split in ("test", "unseen"):
        for workload, by_model in report.get(split, {}).items():
            for model in models:
                if model in by_model:
                    rows.append({"workload": norm_workload(workload), "split": split,
                                 "model": model, "rmse_ratio": float(by_model[model])})
    return rows


def margin_coverage(forecast: dict[str, Any], outcome: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    splits = forecast["splits"]
    for split in ("test", "unseen"):
        for workload, cell in splits[split]["with_gate"]["by_workload"].items():
            rows.append({"margin": "m_f", "workload": norm_workload(workload),
                         "coverage": float(cell["coverage"]), "n": int(cell["samples"])})
    m_u = outcome["by_model"]["hybrid_xgb"]["m_U"]
    for cell in m_u["test"]:
        if cell["group"] == "workload":
            rows.append({"margin": "m_U", "workload": norm_workload(cell["value"]),
                         "coverage": cell["covered"] / cell["n"], "n": int(cell["n"])})
    return rows


def outcome_rmse(report: dict[str, Any], targets: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"model": model, "target": target, "rmse": float(cells[target]["rmse"]),
             "n": int(cells[target]["n"])}
            for model, cells in report["metrics"].items() for target in targets
            if target in cells]


def gate_stress(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("stress/gate_a", "stress/gate_b"):
        m = report["results"][key]["metrics"]
        rows.append({"gate": key.split("/")[1], "unsafe": m["unsafe"],
                     "unsafe_accepted": m["unsafe_accepted"],
                     "unsafe_rejected": m["unsafe_rejected"],
                     "safe": m["records"] - m["unsafe"], "safe_rejected": m["safe_rejected"]})
    return rows


def parse_cv_table(text: str) -> list[dict[str, Any]]:
    """Rows of the `cand | unsafe_accepted | ...` tables; first complete row per candidate."""
    header: list[str] | None = None
    seen: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 2:
            header = None if not line.strip() else header
            continue
        if cells[0] == "cand":
            header = cells
            continue
        if header is None or len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        try:
            parsed: dict[str, Any] = {"name": row["cand"],
                      "unsafe_accepted": int(row["unsafe_accepted"]),
                      "unsafe_rejected": int(row["unsafe_rejected"]),
                      "false_rejections": int(row["false_rejections"]),
                      "safe_accepted": int(row["safe_accepted"])}
        except (KeyError, ValueError):
            continue  # a summary line with '-' in it
        seen.setdefault(str(parsed["name"]), parsed)
    return list(seen.values())


def closed_loop_migrations(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for m in report["prediction_errors"]["migrations"]:
        _, switch, direction = m["candidate_id"].split(":")
        src, dst = direction.split("->")
        rows.append({"scenario": m["scenario"], "switch": switch, "src": src, "dst": dst,
                     "u_dst_before": float(m["u_before"][dst]),
                     "predicted_dst": float(m["predicted_dst"]),
                     "dst_upper_bound": float(m["dst_upper_bound"]),
                     "actual_dst": float(m["actual_dst"])})
    return rows


def lead_time(report: dict[str, Any]) -> list[dict[str, Any]]:
    h = report["horizon"]
    return [{"part": "Telemetry collection", "p95_ms": float(h["collection_p95_ms"])},
            {"part": "Decision pipeline", "p95_ms": float(h["decision_p95_ms"])},
            {"part": "Committed migration", "p95_ms": float(h["migration_p95_ms"]),
             "n": int(h["n_committed"])}]


# ---------------------------------------------------------------- figures

def save(fig: Figure, out: Path, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        path = out / f"{stem}.{ext}"
        fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def fig_capacity(rows: list[dict[str, Any]]) -> Figure:
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    fig.subplots_adjust(wspace=0.32)
    for controller, alpha in (("c1", 1.0), ("c2", 0.45)):
        own = [r for r in rows if r["controller"] == controller]
        live = [r for r in own if r["setup_p95_ms"] is not None]
        left.plot([r["rate_fps"] for r in live], [r["setup_p95_ms"] for r in live],
                  marker="o", ms=3, color=ACCENT, alpha=alpha, label=controller)
        right.plot([r["rate_fps"] for r in own], [r["response_p95_ms"] for r in own],
                   marker="o", ms=3, color=ACCENT, alpha=alpha, label=controller)
        last = live[-1] if live else None
        if last is not None:
            left.annotate(f"{controller} {last['setup_p95_ms']:.0f} ms",
                          (last["rate_fps"], last["setup_p95_ms"]), xytext=(5, 0),
                          textcoords="offset points", va="center", fontsize=8)
    for ax in (left, right):
        for rate, text in ((90, "safe level 90"), (140, "knee 140")):
            ax.axvline(rate, color=DARK, ls="--", lw=0.8)
            if ax is left:
                ax.text(rate, ax.get_ylim()[1], text, ha="center", va="bottom", fontsize=7,
                        color=DARK)
        ax.set_xlabel("New-flow rate (flows/s)")
    left.set_ylabel("ms")
    left.set_title("Flow-setup latency P95, measured at the host", pad=14)
    right.set_title("Packet-In handling P95, measured inside the controller", pad=14)
    left.set_xlim(25, 178)
    right.set_xlim(25, 205)
    right.set_ylim(0, 1.0)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def fig_skill(rows: list[dict[str, Any]], models: tuple[str, ...]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    order = [w for w in WORKLOADS if any(r["workload"] == w for r in rows)]
    width = 0.8 / len(models)
    colors = {models[0]: ACCENT}
    for k, model in enumerate(models):
        vals = [next((r["rmse_ratio"] for r in rows
                      if r["workload"] == w and r["model"] == model), math.nan) for w in order]
        xs = [i - 0.4 + width * (k + 0.5) for i in range(len(order))]
        bars = ax.bar(xs, vals, width=width * 0.95, color=colors.get(model, GREY),
                      label=model.replace("_", " "))
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.axhline(1.0, color=DARK, ls="--", lw=0.8)
    ax.text(-0.45, 1.02, "persistence = 1.0", fontsize=7, color=DARK, va="bottom")
    if "hot-switch" in order:
        ax.axvline(order.index("hot-switch") - 0.5, color=GREY, ls=":", lw=0.8)
    ax.set_xticks(range(len(order)),
                  [f"{w}\n(unseen)" if w == "hot-switch" else w for w in order])
    ax.set_ylabel("RMSE / RMSE of persistence")
    ax.set_title("Load forecast, h = 3 s: RMSE ratio to persistence (lower is better)")
    ax.legend(frameon=False, fontsize=8, ncol=len(models))
    return fig


def fig_coverage(rows: list[dict[str, Any]]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    order = [w for w in WORKLOADS if any(r["workload"] == w for r in rows)]
    for margin, offset, marker, label in (("m_f", -0.12, "o", "m_f (load forecast)"),
                                          ("m_U", 0.12, "s", "m_U (post-migration)")):
        for i, w in enumerate(order):
            cell = next((r for r in rows if r["margin"] == margin and r["workload"] == w), None)
            if cell is None:
                continue
            below = cell["coverage"] < TARGET_COVERAGE
            ax.scatter(i + offset, cell["coverage"], marker=marker, s=30,
                       color=RED if below else GREY, zorder=3,
                       label=label if i == 0 else None)
            ax.annotate(f"{100 * cell['coverage']:.1f}", (i + offset, cell["coverage"]),
                        xytext=(-6 if margin == "m_f" else 6, 0), textcoords="offset points",
                        ha="right" if margin == "m_f" else "left", va="center", fontsize=7)
    ax.axhline(TARGET_COVERAGE, color=DARK, ls="--", lw=0.8)
    ax.text(len(order) - 0.5, TARGET_COVERAGE + 0.004, "target 95%", ha="right",
            fontsize=7, color=DARK)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0.7, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    n_u = sorted({r["n"] for r in rows if r["margin"] == "m_U"})
    below = sum(r["coverage"] < TARGET_COVERAGE for r in rows)
    ax.set_title(f"Margin coverage on the test set: {below}/{len(rows)} points below 95% "
                 f"(m_U: n = {', '.join(map(str, n_u))} per workload)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    return fig


def fig_outcome(rows: list[dict[str, Any]], labels: dict[str, str]) -> Figure:
    targets = list(dict.fromkeys(r["target"] for r in rows))
    models = list(dict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(1, len(targets), figsize=(7.0, 2.2), sharex=True)
    for ax, target in zip(axes, targets):
        vals = [next(r["rmse"] for r in rows if r["model"] == m and r["target"] == target)
                for m in models]
        bars = ax.barh(range(len(models)), vals,
                       color=[ACCENT if m == "hybrid_xgb" else GREY for m in models])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_yticks(range(len(models)), [m.replace("_", " ") for m in models])
        ax.invert_yaxis()
        ax.set_title(labels.get(target, target))
        ax.set_xlabel("RMSE (U units)")
    n = rows[0]["n"] if rows else 0
    fig.suptitle(f"Post-Migration Outcome Model on the test set (n = {n} migrations)", fontsize=9,
                 y=1.08)
    return fig


def fig_gate_stress(rows: list[dict[str, Any]]) -> Figure:
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    names = {"gate_a": "Gate A (no margin)", "gate_b": "Gate B (margin m_U)"}
    xs = range(len(rows))
    leak = ax.bar([i - 0.18 for i in xs], [r["unsafe_accepted"] for r in rows], width=0.36,
                  color=RED, label="unsafe accepted (leaked)")
    fr = ax.bar([i + 0.18 for i in xs], [r["safe_rejected"] for r in rows], width=0.36,
                color=GREY, label="safe rejected (false rejection)")
    ax.bar_label(leak, labels=[f"{r['unsafe_accepted']}/{r['unsafe']}" for r in rows],
                 fontsize=7, padding=2)
    ax.bar_label(fr, labels=[f"{r['safe_rejected']}/{r['safe']}" for r in rows],
                 fontsize=7, padding=2)
    ax.set_xticks(list(xs), [names.get(r["gate"], r["gate"]) for r in rows])
    ax.set_ylabel("migrations")
    ax.set_title("Safety Gate on safety-stress-v1 (destination pushed near the limit)")
    ax.legend(frameon=False, fontsize=8)
    return fig


def fig_tradeoff(rows: list[dict[str, Any]]) -> Figure:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ref = rows[0]
    unsafe = ref["unsafe_accepted"] + ref["unsafe_rejected"]
    safe = ref["false_rejections"] + ref["safe_accepted"]
    ax.add_patch(Rectangle((0, 0), ref["false_rejections"], ref["unsafe_accepted"],
                               color=ACCENT, alpha=0.08, lw=0))
    better = sum(r["false_rejections"] < ref["false_rejections"]
                 and r["unsafe_accepted"] < ref["unsafe_accepted"] for r in rows)
    ax.text(0.3, 0.3, f"better than {CV_NAMES.get(ref['name'], ref['name'])}\non both: {better}",
            fontsize=7, color=DARK)
    for r in rows:
        is_ref = r is ref
        ax.scatter(r["false_rejections"], r["unsafe_accepted"], s=40, zorder=3,
                   color=ACCENT if is_ref else GREY)
        right = r["false_rejections"] > 20
        name = CV_NAMES.get(r["name"], r["name"])
        ax.annotate(f"{name} ({r['unsafe_accepted']} leaked, {r['false_rejections']} false rej.)",
                    (r["false_rejections"], r["unsafe_accepted"]),
                    xytext=(-6 if right else 6, -14 if 10 < r["false_rejections"] < 20 else 6),
                    textcoords="offset points",
                    ha="right" if right else "left", fontsize=7)
    ax.set_xlim(0, max(r["false_rejections"] for r in rows) + 3)
    ax.set_ylim(0, max(r["unsafe_accepted"] for r in rows) + 2)
    ax.set_xlabel(f"Safe migrations falsely rejected (of {safe})")
    ax.set_ylabel(f"Unsafe migrations leaked (of {unsafe})")
    ax.set_title("Safety v2 round 1 (cross-validation): no candidate beats v1 on both")
    return fig


def fig_migrations(rows: list[dict[str, Any]]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for i, r in enumerate(rows):
        over = r["actual_dst"] > U_LIMIT
        ax.plot([r["predicted_dst"], r["dst_upper_bound"]], [i, i], color=GREY, lw=6,
                solid_capstyle="butt", label="predicted → upper bound" if i == 0 else None)
        ax.scatter(r["u_dst_before"], i, facecolors="none", edgecolors=DARK, s=28,
                   label="destination U before" if i == 0 else None)
        ax.scatter(r["actual_dst"], i, color=RED if over else "black", s=28, zorder=3,
                   label="actual after migration" if i == 0 else None)
        ax.annotate(f"{r['actual_dst']:.3f}", (max(r["actual_dst"], r["dst_upper_bound"]), i),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=7,
                    color=RED if over else "black")
    ax.axvline(U_LIMIT, color=DARK, ls="--", lw=0.8)
    ax.set_yticks(range(len(rows)),
                  [f"{r['scenario']} · {r['switch']} {r['src']}→{r['dst']}" for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.35, 1.12)
    ax.set_xlabel("Destination controller U")
    covered = sum(r["actual_dst"] <= r["dst_upper_bound"] for r in rows)
    ax.set_title(f"Closed loop 2C: {covered}/{len(rows)} migrations with actual U ≤ upper bound")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.25),
              ncol=3)
    return fig


def fig_lead_time(rows: list[dict[str, Any]], current_h: int, required_h: int) -> Figure:
    fig, ax = plt.subplots(figsize=(6.0, 1.9))
    start = 0.0
    for k, r in enumerate(rows):
        width = r["p95_ms"] / 1000
        ax.barh(0, width, left=start, height=0.5, color=ACCENT, alpha=0.35 + 0.3 * k)
        mid = start + width / 2
        if width > 0.3:
            ax.text(mid, 0.33, f"{r['part']}\n{width:.2f} s", ha="center", va="bottom",
                    fontsize=7)
        else:
            ax.annotate(f"{r['part']}: {width:.2f} s", (mid, -0.25), xytext=(0, -14),
                        textcoords="offset points", ha="left", fontsize=7,
                        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": DARK})
        start += width
    ax.text(start + 0.05, 0, f"total {start:.2f} s", va="center", fontsize=8, weight="bold")
    for h, text in ((current_h, f"current h = {current_h} s"),
                    (required_h, f"required h = {required_h} s")):
        ax.axvline(h, color=DARK, ls="--", lw=0.8)
        ax.text(h + 0.04, -0.62, text, ha="left", fontsize=7, color=DARK)
    ax.set_ylim(-0.75, 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, max(required_h, start) + 0.6)
    ax.set_xlabel("seconds (P95 of each stage, closed loop 2C)")
    ax.set_title("Time from data collection to completed migration vs forecast horizon")
    return fig


# ---------------------------------------------------------------- main

def git_commit(root: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=root)
    return proc.stdout.strip() or "unknown"


def load_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def build(root: Path) -> dict[str, tuple[Callable[[], Figure], Any, list[str]]]:
    """name -> (draw, plotted rows, source files); only figures whose inputs exist."""
    src = {k: root / v for k, v in SOURCES.items()}
    jobs: dict[str, tuple[Callable[[], Figure], Any, list[str]]] = {}

    def need(*keys: str) -> bool:
        missing = [SOURCES[k] for k in keys if not src[k].exists()]
        for m in missing:
            print(f"SKIP: {m} not found")
        return not missing

    if need("capacity"):
        cap = capacity_curves(src["capacity"])
        jobs["fig_01_capacity_latency"] = (lambda: fig_capacity(cap), cap, [SOURCES["capacity"]])
    if need("skill"):
        models = ("xgboost", "random_forest")
        skill = forecast_skill(load_json(src["skill"]), models)
        jobs["fig_02_forecast_skill"] = (lambda: fig_skill(skill, models), skill,
                                         [SOURCES["skill"]])
    if need("forecast_coverage", "outcome_coverage"):
        cov = margin_coverage(load_json(src["forecast_coverage"]),
                              load_json(src["outcome_coverage"]))
        jobs["fig_03_margin_coverage"] = (lambda: fig_coverage(cov), cov,
                                          [SOURCES["forecast_coverage"],
                                           SOURCES["outcome_coverage"]])
    if need("outcome_test"):
        labels = {"u_dst_post": "Destination controller load", "u_src_post": "Source controller load"}
        rmse = outcome_rmse(load_json(src["outcome_test"]), tuple(labels))
        jobs["fig_04_outcome_rmse"] = (lambda: fig_outcome(rmse, labels), rmse,
                                       [SOURCES["outcome_test"]])
    if need("gate_final"):
        stress = gate_stress(load_json(src["gate_final"]))
        jobs["fig_05_gate_stress"] = (lambda: fig_gate_stress(stress), stress,
                                      [SOURCES["gate_final"]])
    if need("gate_cv"):
        cv = parse_cv_table(src["gate_cv"].read_text(encoding="utf-8"))
        if cv:
            jobs["fig_06_gate_tradeoff"] = (lambda: fig_tradeoff(cv), cv, [SOURCES["gate_cv"]])
        else:
            print(f"SKIP: no table rows in {SOURCES['gate_cv']}")
    if need("closed_loop"):
        loop = load_json(src["closed_loop"])
        migrations = closed_loop_migrations(loop)
        lead = lead_time(loop)
        current_h, required_h = int(loop["horizon"]["current_h"]), int(
            loop["horizon"]["horizon_steps"])
        jobs["fig_07_closed_loop_migrations"] = (lambda: fig_migrations(migrations), migrations,
                                                 [SOURCES["closed_loop"]])
        jobs["fig_08_lead_time"] = (lambda: fig_lead_time(lead, current_h, required_h),
                                    {"parts": lead, "current_h": current_h,
                                     "required_h": required_h}, [SOURCES["closed_loop"]])
    return jobs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None, help=f"default: <root>/{OUT}")
    args = ap.parse_args(argv)
    out = args.out or args.root / OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"generated_from_commit": git_commit(args.root), "figures": {}}
    for name, (draw, data, sources) in build(args.root).items():
        files = save(draw(), out, name)
        manifest["figures"][name] = {"sources": sources, "files": [Path(f).name for f in files],
                                     "data": data}
        print(f"wrote {name} (.png, .pdf)")
    (out / "figures_data.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"wrote {out / 'figures_data.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
