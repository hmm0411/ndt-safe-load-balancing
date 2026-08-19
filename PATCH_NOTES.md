# Telemetry completion patch notes

Apply these files on top of the current repository; they are not a full repository export.

Mandatory companion edit in `src/controller/reactive_controller.py`:

1. In `state_change_handler`, when a datapath enters `MAIN_DISPATCHER`, call:

```python
self.telemetry.register_switch(dp.id)
```

2. In `switch_features_handler`, after `self.role_manager.register_datapath(dp)`, also call:

```python
self.telemetry.register_switch(dp.id)
```

Do not create any second Packet-In/Flow-Mod counter outside `TelemetryAgent`.

The existing instrumentation positions are otherwise correct:

- `record_packet_in(dpid)` before the role check;
- `record_processed_packet_in(dpid)` only after `current_role == "MASTER"` and the LLDP filter;
- `record_flow_mod(dpid)` immediately after sending the `OFPFlowMod`;
- `record_response_time(...)` after Flow-Mod/Packet-Out processing finishes.

Current-code defects addressed by this patch:

- `processed_pin_total` was read before assignment in `TelemetryAgent.sample()`;
- controller output omitted `processed_packet_in_rate`;
- switch output omitted `processed_packet_in_rate`;
- orchestrator constructors omitted both processed fields;
- `ingested_at` was created before the controller request loop instead of immediately after each response;
- failure of one controller aborted the entire collection pass;
- `consistent` checked only owner IDs and did not compare actual MASTER/SLAVE/connectivity state;
- telemetry settings were mixed into/malformed under `migration.yaml`, while `telemetry.yaml` was empty;
- README rollback example used boolean `true`, while the implementation expects the `flow_mod` failure mode;
- unit coverage was missing schema/builder/stale/missing/role-conflict cases.

Additional correctness hardening: `packet_in_handler` now processes/records a Packet-In only when the cached role is exactly `MASTER` (rather than merely “not SLAVE”), so `processed_packet_in_*` keeps the definition promised by the telemetry design even during `EQUAL`/`UNKNOWN` startup states.
