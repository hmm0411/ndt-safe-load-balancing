class RollbackExecutor:
    def __init__(self, client, role_timeout_seconds, barrier_timeout_seconds):
        self.client = client
        self.role_timeout_seconds = role_timeout_seconds
        self.barrier_timeout_seconds = barrier_timeout_seconds

    def restore_source(self, dpid, source_controller, generation_id):
        role = self.client.set_role(
            source_controller, dpid, "MASTER", generation_id, self.role_timeout_seconds
        )
        barrier = self.client.barrier(
            source_controller, dpid, self.barrier_timeout_seconds
        )
        return {"role_result": role, "barrier_result": barrier}
