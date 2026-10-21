from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any


def _field(item: Any, name: str) -> str:
    if isinstance(item, Mapping):
        return str(item[name])
    return str(getattr(item, name))


def select_switch(
    ownership: Iterable[Any],
    source: str,
    rng: random.Random,
    hot_switch_id: str | None = None,
) -> str:
    """Pick the switch to migrate from controller `source`.

    ownership: entries with `switch_id` and `owner_controller_id`
    (TwinState.ownership as dicts, or OwnershipState objects).
    hot_switch_id given -> that switch must belong to `source`.
    Otherwise pick uniformly at random (seeded rng) among source's switches.
    """
    owned = sorted(
        _field(item, "switch_id")
        for item in ownership
        if _field(item, "owner_controller_id") == source
    )
    if not owned:
        raise ValueError(f"controller {source!r} owns no switch")
    if hot_switch_id is not None:
        if hot_switch_id not in owned:
            raise ValueError(
                f"hot switch {hot_switch_id!r} is not owned by {source!r}"
            )
        return hot_switch_id
    return rng.choice(owned)
