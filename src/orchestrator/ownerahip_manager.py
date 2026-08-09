from threading import RLock
from src.schemas.snapshot import OwnershipState

class OwnershipManager:
    def __init__(self, initial_ownership: dict[str, str], controller_ids: list[str]):
        self._ownership = dict(initial_ownership)
        self._controller_ids = list(controller_ids)
        self._version = 1
        self._last_generation = {switch_id: 0 for switch_id in initial_ownership}
        self._lock = RLock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def get_owner(self, switch_id: str) -> str:
        with self._lock:
            if switch_id not in self._ownership:
                raise KeyError(f"Unknown switch: {switch_id}")
            return self._ownership[switch_id]

    def snapshot_mapping(self) -> dict[str, str]:
        with self._lock:
            return dict(self._ownership)

    def commit_migration(self, switch_id: str, target_controller: str, generation_id: int):
        with self._lock:
            self._ownership[switch_id] = target_controller
            self._last_generation[switch_id] = generation_id
            self._version += 1

    def mark_restored(self, switch_id: str, source_controller: str, generation_id: int):
        with self._lock:
            self._ownership[switch_id] = source_controller
            self._last_generation[switch_id] = generation_id
            self._version += 1

    def states(self) -> list[OwnershipState]:
        with self._lock:
            result = []
            for switch_id, owner in sorted(self._ownership.items()):
                other = next((c for c in self._controller_ids if c != owner), owner)
                result.append(OwnershipState(
                    switch_id=switch_id,
                    owner_controller_id=owner,
                    source_role="MASTER",
                    target_role="SLAVE" if other != owner else "MASTER",
                    generation_id=self._last_generation[switch_id],
                    ownership_version=self._version,
                ))
            return result
