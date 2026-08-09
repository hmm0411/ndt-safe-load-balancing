from __future__ import annotations
from threading import RLock
from time import monotonic
from typing import Any

class BarrierManager:
    def __init__(self, logger):
        self.logger = logger
        self._requests: dict[int, dict[str, Any]] = {}
        self._lock = RLock()

    def send(self, datapath) -> int:
        request = datapath.ofproto_parser.OFPBarrierRequest(datapath)
        datapath.set_xid(request)
        with self._lock:
            self._requests[request.xid] = {
                "xid": request.xid,
                "dpid": datapath.id,
                "status": "PENDING",
                "sent_monotonic": monotonic(),
            }
        datapath.send_msg(request)
        return request.xid

    def handle_reply(self, msg) -> None:
        with self._lock:
            item = self._requests.get(msg.xid)
            if item:
                item.update({
                    "status": "REPLIED",
                    "elapsed_ms": (monotonic() - item["sent_monotonic"]) * 1000.0,
                })

    def get_request(self, xid: int) -> dict[str, Any] | None:
        with self._lock:
            item = self._requests.get(xid)
            return dict(item) if item else None
