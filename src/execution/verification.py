from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    source_role: str
    target_role: str
    reasons: list[str]

    def to_dict(self):
        return asdict(self)

class MigrationVerifier:
    def __init__(self, client, timeout_seconds):
        self.client = client
        self.timeout_seconds = timeout_seconds

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
        return VerificationResult(not reasons, source_role, target_role, reasons)
