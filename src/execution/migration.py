import time
import requests

class ControllerAPIError(RuntimeError):
    pass

class ControllerClient:
    def __init__(self, controller_urls: dict[str, str], request_timeout_seconds=3.0):
        self.controller_urls = controller_urls
        self.request_timeout_seconds = request_timeout_seconds

    def _base(self, cid):
        return self.controller_urls[cid].rstrip("/")

    def state(self, cid):
        r = requests.get(f"{self._base(cid)}/api/v1/state", timeout=self.request_timeout_seconds)
        r.raise_for_status()
        return r.json()

    def telemetry(self, cid):
        r = requests.get(f"{self._base(cid)}/api/v1/telemetry", timeout=self.request_timeout_seconds)
        r.raise_for_status()
        return r.json()

    def install_test_flow(self, cid, dpid):
        base = self._base(cid)
        r = requests.post(f"{base}/api/v1/switches/{dpid}/flow-test", timeout=self.request_timeout_seconds,)
        r.raise_for_status()
        return r.json()

    def _wait(self, url, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        latest = {}
        while time.monotonic() < deadline:
            r = requests.get(url, timeout=self.request_timeout_seconds)
            r.raise_for_status()
            latest = r.json()
            if latest.get("status") in {"REPLIED", "ERROR"}:
                return latest
            time.sleep(0.05)
        raise ControllerAPIError(f"timeout waiting for {url}; last={latest}")

    def set_role(self, cid, dpid, role, generation_id, timeout_seconds):
        base = self._base(cid)
        r = requests.post(
            f"{base}/api/v1/switches/{dpid}/role",
            json={"role": role, "generation_id": int(generation_id)},
            timeout=self.request_timeout_seconds,
        )
        r.raise_for_status()
        xid = int(r.json()["xid"])
        result = self._wait(f"{base}/api/v1/role-requests/{xid}", timeout_seconds)
        if result.get("status") != "REPLIED":
            raise ControllerAPIError(str(result))
        return result

    def query_role(self, cid, dpid, timeout_seconds):
        base = self._base(cid)
        r = requests.post(
            f"{base}/api/v1/switches/{dpid}/role/query",
            timeout=self.request_timeout_seconds,
        )
        r.raise_for_status()
        xid = int(r.json()["xid"])
        return self._wait(f"{base}/api/v1/role-requests/{xid}", timeout_seconds)

    def barrier(self, cid, dpid, timeout_seconds):
        base = self._base(cid)
        r = requests.post(
            f"{base}/api/v1/switches/{dpid}/barrier",
            timeout=self.request_timeout_seconds,
        )
        r.raise_for_status()
        xid = int(r.json()["xid"])
        return self._wait(f"{base}/api/v1/barrier-requests/{xid}", timeout_seconds)

class RoleMigrationExecutor:
    def __init__(self, client, role_timeout_seconds, barrier_timeout_seconds):
        self.client = client
        self.role_timeout_seconds = role_timeout_seconds
        self.barrier_timeout_seconds = barrier_timeout_seconds

    def promote_target(self, dpid, target_controller, generation_id):
        role = self.client.set_role(
            target_controller, dpid, "MASTER", generation_id, self.role_timeout_seconds
        )
        barrier = self.client.barrier(
            target_controller, dpid, self.barrier_timeout_seconds
        )
        return {"role_result": role, "barrier_result": barrier}
