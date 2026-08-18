import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    source_role: str
    target_role: str
    source_connected: bool
    target_connected: bool
    packet_in_verified: bool
    flow_mod_verified: bool
    reasons: list[str]

    def to_dict(self):
        return asdict(self)


class MigrationVerifier:
    def __init__(self, client, timeout_seconds):
        self.client = client
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _switch_state(state, dpid):
        for item in state.get("switches", []):
            if int(item.get("dpid", -1)) == int(dpid):
                return item
        return None

    @staticmethod
    def _switch_telemetry(payload, dpid):
        switch_id = f"s{dpid}"
        for item in payload.get("switches", []):
            if item.get("switch_id") == switch_id:
                return item
        return None

    def verify_roles(self, dpid, source_controller, target_controller):
        source = self.client.query_role(source_controller, dpid, self.timeout_seconds)
        target = self.client.query_role(target_controller, dpid, self.timeout_seconds)

        source_role = source.get("reply_role", "UNKNOWN")
        target_role = target.get("reply_role", "UNKNOWN")
        reasons = []

        if source_role != "SLAVE":
            reasons.append(f"source_expected_SLAVE_actual_{source_role}")
        if target_role != "MASTER":
            reasons.append(f"target_expected_MASTER_actual_{target_role}")
        return source_role, target_role, reasons

    def verify_connectivity(self, dpid, source_controller, target_controller):
        source_state = self.client.state(source_controller)
        target_state = self.client.state(target_controller)

        source_switch = self._switch_state(source_state, dpid)
        target_switch = self._switch_state(target_state, dpid)

        source_connected = bool(source_switch and source_switch.get("connected") is True)
        target_connected = bool(target_switch and target_switch.get("connected") is True)

        reasons = []

        if not source_connected:
            reasons.append("source_switch_disconnected")
        if not target_connected:
            reasons.append("target_switch_disconnected")
        return source_connected, target_connected, reasons

    def verify_packet_in(self, dpid, target_controller):
        before = self.client.telemetry(target_controller)
        before_switch = self._switch_telemetry(before, dpid)
        if before_switch is None:
            return False, "target_switch_telemetry_missing"
        before_total = int(before_switch.get("processed_packet_in_total", 0))
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            current = self.client.telemetry(target_controller)
            current_switch = self._switch_telemetry(current, dpid)
            if current_switch is not None:
                current_total = int(current_switch.get("processed_packet_in_total", 0))
                if current_total > before_total:
                    return True, None
            time.sleep(0.1)
        return False, "target_no_new_processed_packet_in"

    def verify_flow_mod(self, dpid, target_controller, simulate_failure=False):
        before = self.client.telemetry(target_controller)
        before_switch = self._switch_telemetry(before, dpid)

        if before_switch is None:
            return False, "target_switch_telemetry_missing"

        before_total = int(before_switch.get("flow_mod_total", 0))
        self.client.install_test_flow(target_controller, dpid)
        deadline = time.monotonic() + self.timeout_seconds

        while time.monotonic() < deadline:
            current = self.client.telemetry(target_controller)
            current_switch = self._switch_telemetry(current, dpid)
            if current_switch is not None:
                current_total = int(current_switch.get("flow_mod_total", 0))
                if current_total > before_total:
                    if simulate_failure:
                        return False, "simulated_flow_mod_failure"
                    return True, None
            time.sleep(0.1)
        return False, "target_no_new_flow_mod"

    def verify_migration(self, dpid, source_controller, target_controller, simulate_flow_mod_failure=False):
        source_role, target_role, role_reasons = self.verify_roles(dpid, source_controller, target_controller)
        source_connected, target_connected, connectivity_reasons = self.verify_connectivity(dpid, source_controller, target_controller)
        reasons = role_reasons + connectivity_reasons
        packet_in_verified, packet_error = self.verify_packet_in(dpid, target_controller)
        if not packet_in_verified: reasons.append(packet_error)
        flow_mod_verified, flow_error = self.verify_flow_mod(dpid, target_controller, simulate_failure=simulate_flow_mod_failure)
        if not flow_mod_verified: reasons.append(flow_error)

        return VerificationResult(
            ok=not reasons,
            source_role=source_role,
            target_role=target_role,
            source_connected=source_connected,
            target_connected=target_connected,
            packet_in_verified=packet_in_verified,
            flow_mod_verified=flow_mod_verified,
            reasons=reasons,
        )

    def verify_rollback(self, dpid, restored_controller, other_controller):
        restored = self.client.query_role(restored_controller, dpid, self.timeout_seconds)
        other = self.client.query_role(other_controller, dpid, self.timeout_seconds)
        restored_role = restored.get("reply_role", "UNKNOWN")
        other_role = other.get("reply_role", "UNKNOWN")
        restored_state = self.client.state(restored_controller)
        other_state = self.client.state(other_controller)
        restored_switch = self._switch_state(restored_state, dpid)
        other_switch = self._switch_state(other_state, dpid)
        restored_connected = bool(restored_switch and restored_switch.get("connected") is True)
        other_connected = bool(other_switch and other_switch.get("connected") is True)
        reasons = []
        if restored_role != "MASTER":
            reasons.append("restored_controller_not_MASTER")
        if other_role != "SLAVE":
            reasons.append("other_controller_not_SLAVE")
        if not restored_connected:
            reasons.append("restored_controller_disconnected")
        if not other_connected:
            reasons.append("other_controller_disconnected")
        return VerificationResult(ok=not reasons, source_role=restored_role, target_role=other_role, source_connected=restored_connected, target_connected=other_connected, packet_in_verified=False, flow_mod_verified=False, reasons=reasons)