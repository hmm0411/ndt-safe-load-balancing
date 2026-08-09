#!/usr/bin/env python3
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo

class Smoke2C4STopo(Topo):
    def build(self):
        switches = []
        for i in range(1, 5):
            s = self.addSwitch(f"s{i}", dpid=f"{i:016x}", protocols="OpenFlow13")
            h = self.addHost(
                f"h{i}",
                ip=f"10.0.0.{i}/24",
                mac=f"00:00:00:00:00:{i:02x}",
            )
            self.addLink(h, s)
            switches.append(s)
        self.addLink(switches[0], switches[1])
        self.addLink(switches[1], switches[2])
        self.addLink(switches[2], switches[3])

def main():
    setLogLevel("info")
    net = Mininet(
        topo=Smoke2C4STopo(),
        controller=None,
        switch=OVSSwitch,
        link=TCLink,
        build=False,
    )
    c1 = net.addController("c1", controller=RemoteController, ip="127.0.0.1", port=6653)
    c2 = net.addController("c2", controller=RemoteController, ip="127.0.0.1", port=6654)
    net.build()
    for switch in net.switches:
        switch.start([c1, c2])

    print("*** Topology ready. DO NOT generate traffic yet.")
    print("*** Run: curl -X POST http://127.0.0.1:9000/api/v1/init-roles")
    try:
        CLI(net)
    finally:
        net.stop()

if __name__ == "__main__":
    main()
