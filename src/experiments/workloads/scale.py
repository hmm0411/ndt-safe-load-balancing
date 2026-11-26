"""Per-controller load scale of the closed-loop campaign (decided 26/11, My: c1 x 1.2).

The five forecast workloads keep U below 1 (gradual tops at 0.95), so the closed loop is
never triggered. The campaign keeps their shapes and multiplies the target utilization of
the controller that runs the workload: NDT_WORKLOAD_SCALE="c1=1.2". Unset = 1.0 for every
controller, i.e. every existing run is unchanged. Constant profiles of the other
controllers (c2-c4 at 0.45) are not scaled when their controller has no entry.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

ENV = "NDT_WORKLOAD_SCALE"


def parse_scales(text: str | None) -> dict[str, float]:
    """"c1=1.2,c3=1.1" -> {"c1": 1.2, "c3": 1.1}; empty -> {}."""
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        cid, sep, value = part.partition("=")
        cid = cid.strip()
        if not sep or not cid:
            raise ValueError(f"{ENV}: expected <controller>=<factor>, got {part!r}")
        try:
            factor = float(value)
        except ValueError as exc:
            raise ValueError(f"{ENV}: factor of {cid} is not a number: {value!r}") from exc
        if not math.isfinite(factor) or not 0.0 < factor <= 3.0:
            raise ValueError(f"{ENV}: factor of {cid} must be in (0, 3], got {factor}")
        if cid in out:
            raise ValueError(f"{ENV}: {cid} given twice")
        out[cid] = factor
    return out


def workload_scales(environ: Mapping[str, str] | None = None) -> dict[str, float]:
    return parse_scales((os.environ if environ is None else environ).get(ENV))


def scale_for(controller_id: str, environ: Mapping[str, str] | None = None) -> float:
    return workload_scales(environ).get(controller_id, 1.0)
