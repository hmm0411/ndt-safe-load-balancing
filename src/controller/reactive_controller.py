from __future__ import annotations
import json
import os
import time

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet, ipv4, udp
from ryu.ofproto import ofproto_v1_3
from webob import Response

from src.controller.barrier import BarrierManager
from src.controller.role_manager import RoleManager
from src.controller.telemetry_agent import TelemetryAgent

REST_INSTANCE = "ndt_controller"
BENCHMARK_UDP_PORT = 9000
COOKIE_TABLE_MISS = 0x0
COOKIE_REACTIVE = 0x10
COOKIE_BENCHMARK = 0x30

PRIORITY_TABLE_MISS = 0
PRIORITY_REACTIVE = 10
PRIORITY_BENCHMARK = 20

IDLE_TIMEOUT_BENCHMARK = 5
IDLE_TIMEOUT_REACTIVE = 30

def json_response(payload, status=200):
    return Response(
        status=status,
        content_type="application/json",
        body=json.dumps(payload).encode("utf-8"),
    )

class ReactiveController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller_id = os.getenv("CONTROLLER_ID", "unknown")
        self.mac_to_port = {}
        self.datapaths = {}
        self.role_manager = RoleManager(self.logger)
        self.barrier_manager = BarrierManager(self.logger)
        self.telemetry = TelemetryAgent(self.controller_id, self.logger)

        kwargs["wsgi"].register(NDTRestController, {REST_INSTANCE: self})
        self._telemetry_thread = hub.spawn(self._telemetry_loop)

    def _telemetry_loop(self):
        while True:
            try:
                self.telemetry.sample(self.role_manager.count_master_switches())
            except Exception:
                self.logger.exception("TELEMETRY_SAMPLE_FAILED")
            hub.sleep(1.0)

    def get_datapath(self, dpid: int):
        return self.datapaths.get(int(dpid))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
            self.role_manager.register_datapath(dp)
            self.telemetry.register_switch(dp.id)
            self.logger.info("CHANNEL_UP controller=%s dpid=%016x", self.controller_id, dp.id)
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)
            self.logger.info("CHANNEL_DOWN controller=%s dpid=%016x", self.controller_id, dp.id)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        self.role_manager.register_datapath(dp)
        self.telemetry.register_switch(dp.id)
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        self.add_flow(
            dp,
            priority=PRIORITY_TABLE_MISS,
            match=parser.OFPMatch(),
            actions=[parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)],
            cookie=COOKIE_TABLE_MISS,
        )

    def add_flow(self, dp, priority, match, actions, buffer_id=None, idle_timeout=IDLE_TIMEOUT_REACTIVE, cookie=COOKIE_TABLE_MISS):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        kwargs = {
            "datapath": dp,
            "priority": priority,
            "cookie": cookie,
            "match": match,
            "instructions": [
                parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)
            ],
            "idle_timeout": 0 if priority == PRIORITY_TABLE_MISS else idle_timeout,
        }
        if buffer_id is not None:
            kwargs["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kwargs))
        self.telemetry.record_flow_mod(dp.id)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        start_ns = time.perf_counter_ns()
        msg = ev.msg
        dp = msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        dpid = dp.id

        self.telemetry.record_packet_in(dpid)
        current_role = self.role_manager.get_cached_role(dpid)
        if current_role != "MASTER":
            return

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        self.telemetry.record_processed_packet_in(dpid)

        dst, src = eth.dst, eth.src
        in_port = msg.match["in_port"]

        # 1. Detect UDP benchmark traffic
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        udp_pkt = pkt.get_protocol(udp.udp)
        is_benchmark = (ip_pkt is not None and udp_pkt is not None and udp_pkt.dst_port == BENCHMARK_UDP_PORT)

        # 2. MAC learning
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        # 3. Install flow only when destination is known
        if out_port != ofp.OFPP_FLOOD:
            if is_benchmark:
                match = parser.OFPMatch(
                    in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                    ipv4_src=ip_pkt.src, ipv4_dst=ip_pkt.dst,
                    ip_proto=17, udp_src=udp_pkt.src_port, udp_dst=udp_pkt.dst_port)
                priority = PRIORITY_BENCHMARK
                cookie = COOKIE_BENCHMARK
                idle_timeout = IDLE_TIMEOUT_BENCHMARK
            else:
                match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
                priority = PRIORITY_REACTIVE
                cookie = COOKIE_REACTIVE
                idle_timeout = IDLE_TIMEOUT_REACTIVE

            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                self.add_flow(dp, priority, match, actions, buffer_id=msg.buffer_id, idle_timeout=idle_timeout, cookie=cookie)
                self.telemetry.record_response_time(
                    (time.perf_counter_ns() - start_ns) / 1_000_000.0
                )
                return
            self.add_flow(dp, priority, match, actions, idle_timeout=idle_timeout, cookie=cookie)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        ))
        self.telemetry.record_response_time(
            (time.perf_counter_ns() - start_ns) / 1_000_000.0
        )

    @set_ev_cls(ofp_event.EventOFPRoleReply, MAIN_DISPATCHER)
    def role_reply_handler(self, ev):
        self.role_manager.handle_role_reply(ev.msg)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        self.barrier_manager.handle_reply(ev.msg)

class NDTRestController(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[REST_INSTANCE]

    @route(REST_INSTANCE, "/api/v1/state", methods=["GET"])
    def state(self, req, **kwargs):
        return json_response({
            "controller_id": self.app.controller_id,
            "switches": [
                {
                    "switch_id": f"s{dpid}",
                    "dpid": dpid,
                    "connected": True,
                    "role": self.app.role_manager.get_cached_role(dpid),
                }
                for dpid in sorted(self.app.datapaths)
            ],
        })

    @route(REST_INSTANCE, "/api/v1/telemetry", methods=["GET"])
    def telemetry(self, req, **kwargs):
        return json_response(self.app.telemetry.latest())

    @route(REST_INSTANCE, "/api/v1/switches/{dpid}/role", methods=["POST"])
    def set_role(self, req, dpid, **kwargs):
        dp = self.app.get_datapath(int(dpid))
        if dp is None:
            return json_response({"error": "datapath_not_connected"}, 404)
        try:
            payload = json.loads(req.body or b"{}")
            xid = self.app.role_manager.send_role_request(
                dp, str(payload["role"]), int(payload["generation_id"])
            )
            return json_response({"xid": xid, "status": "PENDING"}, 202)
        except (KeyError, ValueError, TypeError) as exc:
            return json_response({"error": str(exc)}, 400)

    @route(REST_INSTANCE, "/api/v1/switches/{dpid}/role/query", methods=["POST"])
    def query_role(self, req, dpid, **kwargs):
        dp = self.app.get_datapath(int(dpid))
        if dp is None:
            return json_response({"error": "datapath_not_connected"}, 404)
        xid = self.app.role_manager.query_current_role(dp)
        return json_response({"xid": xid, "status": "PENDING"}, 202)

    @route(REST_INSTANCE, "/api/v1/role-requests/{xid}", methods=["GET"])
    def role_request(self, req, xid, **kwargs):
        item = self.app.role_manager.get_request(int(xid))
        return json_response(item, 200) if item else json_response({"error": "unknown_xid"}, 404)

    @route(REST_INSTANCE, "/api/v1/switches/{dpid}/barrier", methods=["POST"])
    def barrier(self, req, dpid, **kwargs):
        dp = self.app.get_datapath(int(dpid))
        if dp is None:
            return json_response({"error": "datapath_not_connected"}, 404)
        xid = self.app.barrier_manager.send(dp)
        return json_response({"xid": xid, "status": "PENDING"}, 202)

    @route(REST_INSTANCE, "/api/v1/barrier-requests/{xid}", methods=["GET"])
    def barrier_request(self, req, xid, **kwargs):
        item = self.app.barrier_manager.get_request(int(xid))
        return json_response(item, 200) if item else json_response({"error": "unknown_xid"}, 404)

    @route(REST_INSTANCE, "/api/v1/switches/{dpid}/flow-test", methods=["POST"])
    def flow_test(self, req, dpid, **kwargs):
        dp = self.app.get_datapath(int(dpid))

        if dp is None:
            return json_response({"error": "datapath_not_connected"}, 404)
        role = self.app.role_manager.get_cached_role(dp.id)
        if role != "MASTER":
            return json_response({"error": "controller_not_master", "role": role,}, 409,)

        parser = dp.ofproto_parser
        match = parser.OFPMatch(eth_type=0x88B5)
        self.app.add_flow(dp, priority=100, match=match, actions=[], idle_timeout=5,)

        return json_response({"status": "FLOW_MOD_SENT", "dpid": int(dpid), "role": role,})