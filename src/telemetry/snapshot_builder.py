from datetime import datetime, timezone
from uuid import uuid4
from src.schemas.snapshot import NetworkSnapshot

class SnapshotBuilder:
    def __init__(self, validator):
        self.validator = validator

    def build(self, topology_version, ownership_version, controllers, switches, ownership):
        created_at = datetime.now(timezone.utc)
        mapping = {x.switch_id: x.owner_controller_id for x in ownership}
        quality = self.validator.validate(
            controllers=controllers,
            switches=switches,
            ownership_mapping=mapping,
            now=created_at,
        )
        return NetworkSnapshot(
            snapshot_id=f"snap-{uuid4().hex[:12]}",
            created_at=created_at,
            topology_version=int(topology_version),
            ownership_version=int(ownership_version),
            controllers=list(controllers),
            switches=list(switches),
            ownership=list(ownership),
            quality=quality,
        )
