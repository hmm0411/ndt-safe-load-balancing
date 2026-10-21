from __future__ import annotations

from collections.abc import Iterable, Mapping

from .constant import ConstantProfile


def parse_profile_specs(specs: Iterable[str] | None) -> dict[str, str]:
    """["c1=stable", "c2=constant:0.45"] -> {"c1": "stable", "c2": "constant:0.45"}."""
    result: dict[str, str] = {}
    for spec in specs or []:
        controller_id, separator, value = spec.partition("=")
        controller_id = controller_id.strip()
        value = value.strip()
        if not separator or not controller_id or not value:
            raise ValueError(
                f"invalid --profile {spec!r}; expected <controller>=<profile>"
            )
        if controller_id in result:
            raise ValueError(f"duplicate --profile for controller {controller_id!r}")
        result[controller_id] = value
    return result


def stable_utilizations(
    profiles: Mapping[str, str],
    default_target: float,
    controller_ids: Iterable[str],
) -> dict[str, float]:
    """Target utilization per controller for the stable workload script."""
    ids = tuple(controller_ids)
    unknown = sorted(set(profiles) - set(ids))
    if unknown:
        raise ValueError(f"unknown controller in --profile: {unknown}")

    result: dict[str, float] = {}
    for controller_id in ids:
        spec = profiles.get(controller_id, "stable")
        if spec == "stable":
            result[controller_id] = default_target
        elif spec.startswith("constant:"):
            try:
                level = float(spec.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid constant level in {spec!r}") from exc
            result[controller_id] = ConstantProfile(level).target_at(0.0).target_utilization
        else:
            raise ValueError(
                f"stable workload supports 'stable' or 'constant:<U>', got {spec!r}"
            )
    return result


def controller_modes(
    profiles: Mapping[str, str],
    scenario: str,
    controller_ids: Iterable[str],
) -> dict[str, float | None]:
    """Per controller: None = runs the scenario, a float = constant utilization."""
    ids = tuple(controller_ids)
    unknown = sorted(set(profiles) - set(ids))
    if unknown:
        raise ValueError(f"unknown controller in --profile: {unknown}")

    result: dict[str, float | None] = {}
    for controller_id in ids:
        spec = profiles.get(controller_id, scenario)
        if spec == scenario:
            result[controller_id] = None
        elif spec.startswith("constant:"):
            try:
                level = float(spec.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid constant level in {spec!r}") from exc
            result[controller_id] = ConstantProfile(level).target_at(0.0).target_utilization
        else:
            raise ValueError(
                f"{scenario} workload supports {scenario!r} or 'constant:<U>', got {spec!r}"
            )
    return result
