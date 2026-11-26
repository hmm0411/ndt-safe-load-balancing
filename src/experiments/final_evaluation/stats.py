"""Statistics of the closed-loop campaign (docs/evaluation-protocol-v1.md, section 6).

Unit of analysis: the episode (one method, one workload, one seed). For one metric:

* main test: Wilcoxon signed-rank, two-sided, on the pairs (workload, seed) of Full and one
  baseline (5 workloads x 10 seeds = 50 pairs), one test per baseline, Holm over the baselines;
* effect size: matched-pairs rank-biserial r = (T+ - T-) / (T+ + T-), signed so that r > 0
  means Full is better (lower or higher, as the metric says);
* per workload, descriptive only: median [min - max] of each method and of the paired
  difference, and "k/n seeds won" (no p-value: 10 pairs cannot carry the main claim).

Standard library only (no scipy on the orchestrator venv). Exact null distribution when there
are no ties and n <= EXACT_MAX_N, otherwise the normal approximation with tie correction and
continuity correction. A pair with a missing value (None) is dropped and counted, never 0.
Zero differences: ``zero_method`` "wilcox" (dropped, the default) or "pratt" (ranked, then
their ranks left out of T+ and T-), declared in the protocol before the campaign.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

EXACT_MAX_N = 50
ZERO_METHODS = ("wilcox", "pratt")


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based) of ``values``; ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _exact_two_sided(n: int, t_plus: float) -> float:
    """P-value of T+ under H0, ranks 1..n without ties (counts of subset sums)."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    t = int(round(t_plus))
    all_ = 2 ** n
    lower = sum(counts[: t + 1]) / all_
    upper = sum(counts[t:]) / all_
    return min(1.0, 2.0 * min(lower, upper))


@dataclass(frozen=True)
class WilcoxonResult:
    n_pairs: int            # pairs with both values
    n_used: int             # pairs that enter the ranks (non-zero differences for wilcox)
    n_zero: int
    t_plus: float
    t_minus: float
    p_value: float | None   # None when no difference can be ranked
    method: str             # exact / normal
    rank_biserial: float | None


def wilcoxon_signed_rank(differences: Sequence[float],
                         zero_method: str = "wilcox") -> WilcoxonResult:
    """Two-sided test of median difference 0; ``differences`` = improvement of Full per pair
    (positive = Full better), so T+ collects the pairs Full wins."""
    if zero_method not in ZERO_METHODS:
        raise ValueError(f"zero_method must be one of {ZERO_METHODS}")
    d = [float(x) for x in differences]
    if any(not math.isfinite(x) for x in d):
        raise ValueError("differences must be finite")
    zeros = sum(1 for x in d if x == 0.0)
    ranked = [x for x in d if x != 0.0] if zero_method == "wilcox" else d
    ranks = _ranks([abs(x) for x in ranked])
    t_plus = sum(r for r, x in zip(ranks, ranked) if x > 0)
    t_minus = sum(r for r, x in zip(ranks, ranked) if x < 0)
    nonzero = len(d) - zeros
    if nonzero == 0:
        return WilcoxonResult(len(d), 0, zeros, 0.0, 0.0, None, "none", None)
    rbc = (t_plus - t_minus) / (t_plus + t_minus)
    n = len(ranked)
    tied = len(set(abs(x) for x in ranked)) != n
    if zero_method == "wilcox" and not tied and n <= EXACT_MAX_N:
        return WilcoxonResult(len(d), n, zeros, t_plus, t_minus,
                              _exact_two_sided(n, t_plus), "exact", rbc)
    mean = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0
    if zero_method == "pratt":
        nz = zeros
        mean -= nz * (nz + 1) / 4.0
        var -= nz * (nz + 1) * (2 * nz + 1) / 24.0
    groups: dict[float, int] = {}
    for x in ranked:
        if x == 0.0:
            continue  # pratt: the zeros' ranks are already taken out of the variance above
        groups[abs(x)] = groups.get(abs(x), 0) + 1
    var -= sum(t ** 3 - t for t in groups.values()) / 48.0
    if var <= 0:
        return WilcoxonResult(len(d), n, zeros, t_plus, t_minus, None, "normal", rbc)
    z = (abs(t_plus - mean) - 0.5) / math.sqrt(var)
    p = math.erfc(max(z, 0.0) / math.sqrt(2.0))
    return WilcoxonResult(len(d), n, zeros, t_plus, t_minus, min(1.0, p), "normal", rbc)


def holm(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm-adjusted p-values (step-down, monotone); None stays None and is not counted."""
    named = sorted(((k, p) for k, p in p_values.items() if p is not None), key=lambda kv: kv[1])
    m = len(named)
    out: dict[str, float | None] = {k: None for k, p in p_values.items() if p is None}
    running = 0.0
    for i, (k, p) in enumerate(named):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


@dataclass(frozen=True)
class Episode:
    method: str
    workload: str
    seed: int
    value: float | None


def _improvement(full: float, other: float, lower_is_better: bool) -> float:
    return other - full if lower_is_better else full - other


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": statistics.median(values), "min": min(values),
            "max": max(values)}


def compare(episodes: Iterable[Episode], full: str, baselines: Sequence[str],
            lower_is_better: bool, zero_method: str = "wilcox",
            alpha: float = 0.05) -> dict[str, Any]:
    """Full vs each baseline on one metric: main test, Holm, effect size, per workload."""
    table: dict[tuple[str, str, int], float | None] = {}
    for e in episodes:
        key = (e.method, e.workload, int(e.seed))
        if key in table:
            raise ValueError(f"two episodes for {key}")
        table[key] = e.value
    methods = {m for m, _, _ in table}
    missing = [m for m in (full, *baselines) if m not in methods]
    if missing:
        raise ValueError(f"no episode for {missing}")
    results: dict[str, Any] = {}
    for base in baselines:
        cells = sorted({(w, s) for m, w, s in table if m in (full, base)})
        pairs, dropped = [], []
        for w, s in cells:
            va, vb = table.get((full, w, s)), table.get((base, w, s))
            if va is None or vb is None:
                dropped.append(f"{w}/{s}")
            else:
                pairs.append((w, s, float(va), float(vb)))
        diffs = [_improvement(a, b, lower_is_better) for _, _, a, b in pairs]
        test = wilcoxon_signed_rank(diffs, zero_method)
        per_workload = {}
        for w in sorted({p[0] for p in pairs}):
            sub = [p for p in pairs if p[0] == w]
            imp = [_improvement(a, b, lower_is_better) for _, _, a, b in sub]
            per_workload[w] = {
                full: _describe([a for _, _, a, _ in sub]),
                base: _describe([b for _, _, _, b in sub]),
                "improvement": _describe(imp),
                "seeds_won": sum(1 for x in imp if x > 0),
                "seeds_tied": sum(1 for x in imp if x == 0),
                "seeds": len(imp),
            }
        results[base] = {"test": asdict(test), "dropped_pairs": dropped,
                         "improvement": _describe(diffs), "per_workload": per_workload}
    adjusted = holm({b: results[b]["test"]["p_value"] for b in baselines})
    for b in baselines:
        p = adjusted[b]
        results[b]["p_holm"] = p
        results[b]["significant_holm"] = None if p is None else p < alpha
    return {"full": full, "baselines": list(baselines), "lower_is_better": lower_is_better,
            "zero_method": zero_method, "alpha": alpha,
            "improvement_sign": "positive = Full better", "comparisons": results}
