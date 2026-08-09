from __future__ import annotations
from threading import RLock
from time import monotonic
from typing import Any

class RoleManager:
    def __init__(self, logger):
        self.logger = logger
        self._roles: dict[int, str] = {}
        self._requests: dict[int, dict[str, Any]] = {}
        self._lock = RLock()

    @staticmethod
    def _role_name(ofproto, role: int) -> str:
        mapping = {
            ofproto.OFPCR_ROLE_NOCHANGE: "NOCHANGE",
            ofproto.OFPCR_ROLE_EQUAL: "EQUAL",
            ofproto.OFPCR_ROLE_MASTER: "MASTER",
            ofproto.OFPCR_ROLE_SLAVE: "SLAVE",
        }
        return mapping.get(role, f"UNKNOWN({role})")

    @staticmethod
    def _role_value(ofproto, role_name: str) -> int:
        mapping = {
            "NOCHANGE": ofproto.OFPCR_ROLE_NOCHANGE,
            "EQUAL": ofproto.OFPCR_ROLE_EQUAL,
            "MASTER": ofproto.OFPCR_ROLE_MASTER,
            "SLAVE": ofproto.OFPCR_ROLE_SLAVE,
        }
        role_name = role_name.upper()
        if role_name not in mapping:
            raise ValueError(f"Unsupported role: {role_name}")
        return mapping[role_name]

    def register_datapath(self, datapath) -> None:
        with self._lock:
            self._roles.setdefault(datapath.id, "EQUAL")

    def get_cached_role(self, dpid: int) -> str:
        with self._lock:
            return self._roles.get(dpid, "UNKNOWN")

    def count_master_switches(self) -> int:
        with self._lock:
            return sum(role == "MASTER" for role in self._roles.values())

    def send_role_request(self, datapath, role_name: str, generation_id: int) -> int:
        request = datapath.ofproto_parser.OFPRoleRequest(
            datapath,
            role=self._role_value(datapath.ofproto, role_name),
            generation_id=int(generation_id),
        )
        datapath.set_xid(request)
        with self._lock:
            self._requests[request.xid] = {
                "xid": request.xid,
                "dpid": datapath.id,
                "requested_role": role_name.upper(),
                "generation_id": int(generation_id),
                "status": "PENDING",
                "sent_monotonic": monotonic(),
            }
        datapath.send_msg(request)
        self.logger.info(
            "ROLE_REQUEST dpid=%s role=%s generation_id=%s xid=%s",
            datapath.id, role_name.upper(), generation_id, request.xid
        )
        return request.xid

    def query_current_role(self, datapath) -> int:
        return self.send_role_request(datapath, "NOCHANGE", 0)

    def handle_role_reply(self, msg) -> None:
        role_name = self._role_name(msg.datapath.ofproto, msg.role)
        with self._lock:
            self._roles[msg.datapath.id] = role_name
            request = self._requests.get(msg.xid)
            if request:
                request.update({
                    "status": "REPLIED",
                    "reply_role": role_name,
                    "reply_generation_id": int(msg.generation_id),
                    "elapsed_ms": (monotonic() - request["sent_monotonic"]) * 1000.0,
                })
        self.logger.info(
            "ROLE_REPLY dpid=%s role=%s generation_id=%s xid=%s",
            msg.datapath.id, role_name, msg.generation_id, msg.xid
        )

    def get_request(self, xid: int) -> dict[str, Any] | None:
        with self._lock:
            item = self._requests.get(xid)
            return dict(item) if item else None
