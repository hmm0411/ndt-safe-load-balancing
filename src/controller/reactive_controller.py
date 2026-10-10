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
QOS_UDP_PORT = 9001
COOKIE_TABLE_MISS = 0x0
COOKIE_REACTIVE = 0x10
COOKIE_VERIFICATION = 0x20
COOKIE_BENCHMARK = 0x30
COOKIE_QOS = 0x40

PRIORITY_TABLE_MISS = 0
PRIORITY_REACTIVE = 10
PRIORITY_BENCHMARK = 20
PRIORITY_QOS = 20

IDLE_TIMEOUT_BENCHMARK = 5
IDLE_TIMEOUT_REACTIVE = 30
IDLE_TIMEOUT_QOS = 30

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

        # =========================================================
        # Controller identity
        # =========================================================

        self.controller_id = os.getenv(
            "CONTROLLER_ID",
            "unknown",
        )

        # =========================================================
        # Runtime SDN state
        # =========================================================

        # MAC learning table:
        #
        # {
        #     dpid: {
        #         mac_address: port
        #     }
        # }
        self.mac_to_port = {}

        # Active OpenFlow datapaths connected to this controller.
        #
        # {
        #     dpid: datapath
        # }
        self.datapaths = {}

        # =========================================================
        # Role / Barrier / Telemetry managers
        # =========================================================

        self.role_manager = RoleManager(
            self.logger
        )

        self.barrier_manager = BarrierManager(
            self.logger
        )

        self.telemetry = TelemetryAgent(
            self.controller_id,
            self.logger,
        )

        # =========================================================
        # Capacity benchmark configuration
        # =========================================================

        self.benchmark_enabled = (
            os.getenv(
                "NDT_BENCHMARK_ENABLED",
                "0",
            ).strip().lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
        )

        try:
            self.benchmark_udp_port = int(
                os.getenv(
                    "NDT_BENCHMARK_UDP_PORT",
                    str(BENCHMARK_UDP_PORT),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "NDT_BENCHMARK_UDP_PORT "
                "must be an integer"
            ) from exc

        if not (
            1
            <= self.benchmark_udp_port
            <= 65535
        ):
            raise ValueError(
                "NDT_BENCHMARK_UDP_PORT "
                "must be in [1, 65535]"
            )

        try:
            self.qos_udp_port = int(
                os.getenv(
                    "NDT_QOS_UDP_PORT",
                    str(QOS_UDP_PORT),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "NDT_QOS_UDP_PORT "
                "must be an integer"
            ) from exc

        if not (
            1
            <= self.qos_udp_port
            <= 65535
        ):
            raise ValueError(
                "NDT_QOS_UDP_PORT "
                "must be in [1, 65535]"
            )

        try:
            self.benchmark_idle_timeout = int(
                os.getenv(
                    "NDT_BENCHMARK_FLOW_IDLE_TIMEOUT",
                    str(IDLE_TIMEOUT_BENCHMARK),
                )
            )
        except ValueError as exc:
            raise ValueError(
                "NDT_BENCHMARK_FLOW_IDLE_TIMEOUT "
                "must be an integer"
            ) from exc

        if self.benchmark_idle_timeout <= 0:
            raise ValueError(
                "NDT_BENCHMARK_FLOW_IDLE_TIMEOUT "
                "must be > 0"
            )

        # =========================================================
        # REST API
        # =========================================================

        wsgi = kwargs["wsgi"]

        wsgi.register(
            NDTRestController,
            {
                REST_INSTANCE: self,
            },
        )

        # =========================================================
        # Telemetry sampling loop
        # =========================================================

        self._telemetry_thread = hub.spawn(
            self._telemetry_loop
        )

        # =========================================================
        # Startup logging
        # =========================================================

        self.logger.info(
            "CONTROLLER_INITIALIZED "
            "controller=%s "
            "benchmark_enabled=%s "
            "benchmark_udp_port=%s "
            "qos_udp_port=%s "
            "benchmark_idle_timeout=%ss "
            "reactive_idle_timeout=%ss "
            "qos_idle_timeout=%ss",
            self.controller_id,
            self.benchmark_enabled,
            self.benchmark_udp_port,
            self.qos_udp_port,
            self.benchmark_idle_timeout,
            IDLE_TIMEOUT_REACTIVE,
            IDLE_TIMEOUT_QOS,
        )

    def _telemetry_loop(self):
        while True:
            try:
                self.telemetry.sample(self.role_manager.count_master_switches())
            except Exception:
                self.logger.exception("TELEMETRY_SAMPLE_FAILED")
            hub.sleep(1.0)

    def get_datapath(self, dpid: int):
        return self.datapaths.get(int(dpid))

    @set_ev_cls(
        ofp_event.EventOFPStateChange,
        [MAIN_DISPATCHER, DEAD_DISPATCHER],
    )
    def state_change_handler(self, ev):
        dp = ev.datapath
        dpid = getattr(dp, "id", None)

        if ev.state == MAIN_DISPATCHER:
            if dpid is None:
                self.logger.warning(
                    "CHANNEL_UP_WITHOUT_DPID controller=%s",
                    self.controller_id,
                )
                return

            self.datapaths[dpid] = dp
            self.role_manager.register_datapath(dp)
            self.telemetry.register_switch(dpid)

            self.logger.info(
                "CHANNEL_UP controller=%s dpid=%016x",
                self.controller_id,
                dpid,
            )

        elif ev.state == DEAD_DISPATCHER:
            if dpid is not None:
                self.datapaths.pop(dpid, None)

            dpid_text = (
                f"{dpid:016x}"
                if dpid is not None
                else "unknown"
            )

            self.logger.info(
                "CHANNEL_DOWN controller=%s dpid=%s",
                self.controller_id,
                dpid_text,
            )

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

        is_benchmark = (
            ip_pkt is not None
            and udp_pkt is not None
            and udp_pkt.dst_port == BENCHMARK_UDP_PORT
        )

        is_qos = (
            self.benchmark_enabled
            and ip_pkt is not None
            and udp_pkt is not None
            and udp_pkt.dst_port
            == self.qos_udp_port
        )

        self.logger.info(
            "EXPERIMENT_CLASSIFY "
            "controller=%s dpid=%016x "
            "in_port=%s src=%s dst=%s "
            "eth_type=%s "
            "ip_src=%s ip_dst=%s ip_proto=%s "
            "udp_src=%s udp_dst=%s "
            "is_benchmark=%s "
            "is_qos=%s",
            self.controller_id,
            dpid,
            in_port,
            src,
            dst,
            hex(eth.ethertype),
            getattr(ip_pkt, "src", None),
            getattr(ip_pkt, "dst", None),
            getattr(ip_pkt, "proto", None),
            getattr(udp_pkt, "src_port", None),
            getattr(udp_pkt, "dst_port", None),
            is_benchmark,
            is_qos,
        )

        # 2. MAC learning
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        # 3. Install flow only when destination is known
        if out_port != ofp.OFPP_FLOOD:
            if is_benchmark or is_qos:
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_type=(
                        ether_types.ETH_TYPE_IP
                    ),
                    ipv4_src=ip_pkt.src,
                    ipv4_dst=ip_pkt.dst,
                    ip_proto=17,
                    udp_src=udp_pkt.src_port,
                    udp_dst=udp_pkt.dst_port,
                )

                if is_benchmark:
                    priority = (
                        PRIORITY_BENCHMARK
                    )

                    cookie = (
                        COOKIE_BENCHMARK
                    )

                    idle_timeout = (
                        self.benchmark_idle_timeout
                    )

                else:
                    priority = (
                        PRIORITY_QOS
                    )

                    cookie = (
                        COOKIE_QOS
                    )

                    idle_timeout = (
                        IDLE_TIMEOUT_QOS
                    )

            else:
                reactive_match = {
                    "in_port":
                        in_port,

                    "eth_src":
                        src,

                    "eth_dst":
                        dst,

                    "eth_type":
                        eth.ethertype,
                }

                if ip_pkt is not None:
                    reactive_match.update(
                        {
                            "ipv4_src":
                                ip_pkt.src,

                            "ipv4_dst":
                                ip_pkt.dst,

                            "ip_proto":
                                ip_pkt.proto,
                        }
                    )

                    if udp_pkt is not None:
                        reactive_match.update(
                            {
                                "udp_src":
                                    udp_pkt.src_port,

                                "udp_dst":
                                    udp_pkt.dst_port,
                            }
                        )

                match = parser.OFPMatch(
                    **reactive_match
                )

                priority = (
                    PRIORITY_REACTIVE
                )

                cookie = (
                    COOKIE_REACTIVE
                )

                idle_timeout = (
                    IDLE_TIMEOUT_REACTIVE
                )

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
        self.app.add_flow(dp, priority=100, match=match, actions=[], idle_timeout=5,cookie=COOKIE_VERIFICATION)

        return json_response({"status": "FLOW_MOD_SENT", "dpid": int(dpid), "role": role,})
