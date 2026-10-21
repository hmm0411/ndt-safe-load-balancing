# NDT-SDN Safe Load Balancing Dashboard

Static dashboard for the KLTN **A Network Digital Twin-Assisted Framework for Safe Load Balancing in Multi-Controller SDN**.

## Files

```text
dashboard/
├── index.html
└── README_DEPLOY.md
```

No React/Vite build is required.

## Current LIVE API contract

The current FastAPI Orchestrator exposes:

```text
GET  /health                 orchestrator + reachability of every controller
GET  /api/v1/state           ownership, transactions, switch locks, telemetry errors, latest snapshot
GET  /api/v1/twin/state      TwinState: telemetry, Hard Gate valid, Quality Score
GET  /api/v1/twin/forecast   U_hat(t+h), m_f, overload risk, OOD fallback per controller
GET  /docs                   FastAPI Swagger UI
POST /api/v1/init-roles      (not called by the dashboard)
POST /api/v1/migrations      (not called by the dashboard)
```

The dashboard only reads. What each tab uses:

| Tab | Source |
|---|---|
| Dual View, Controllers, Switches | `/api/v1/twin/state` |
| Digital Twin | `/api/v1/twin/state` (Hard Gate, Quality Score) + `/api/v1/state` latest snapshot (conflicts, missing fields, stale sources) |
| Forecast & Risk | `/api/v1/twin/forecast` (READY / NOT_READY reason, U_hat, U_hat + m_f, server-side `overload_risk`, OOD fallback); the chart and the online check are built in the browser from successive snapshots |
| Execution | `/api/v1/state` transactions and `locked_switches` |
| Services | links to the API, both Ryu REST ports and MLflow; runtime resources; controller sessions from `/health` |
| Events | changes between polls (snapshot validity, forecast reason, overload risk, OOD gate, controller reachability, locks) + transaction history |
| Safety Gate, Calibration, Benchmark | not implemented in the backend yet (Nội dung 3–4): empty in LIVE, sample data in Demo |

`/health` calls every controller over REST, so the dashboard polls it every 5 s in the background instead of on every refresh.

To inspect every UI module with simulated data (numbers mirror the real setup: c_safe 1433 pps, h = 3 s, forecast-v1, m_f 0.048 / 0.055):

```text
http://localhost:9000/dashboard/?demo=1
```

## Access (LIVE)

On the VM the orchestrator listens on `127.0.0.1:9000` and serves the dashboard itself at `/dashboard/` (same origin as the API, so no CORS is needed). From your machine:

```bash
gcloud compute ssh kltn-sdn --zone asia-southeast1-b -- -N \
  -L 9000:127.0.0.1:9000 -L 5000:127.0.0.1:5000 \
  -L 8081:127.0.0.1:8081 -L 8082:127.0.0.1:8082
```

Then open `http://localhost:9000/dashboard/`. The extra ports make the MLflow and controller links in the Services tab work through the same tunnel.

## Local test

```bash
cd dashboard
python -m http.server 8080
```

Demo:

```text
http://localhost:8080/?demo=1
```

A page served from another origin (for example `:8080`) cannot read the orchestrator API: the orchestrator does not enable CORS. Use `http://localhost:9000/dashboard/` for LIVE data.

## GitHub

Place this folder in the KLTN repository:

```text
ndt-safe-load-balancing/
└── dashboard/
    ├── index.html
    └── README_DEPLOY.md
```

Changes go through a task branch and a PR into `dev`, like any other task (CLAUDE.md §7).

The dashboard is published via Claude Artifacts. A published copy runs on another origin, so it can show the Demo mode; LIVE data needs the orchestrator over HTTPS with CORS (see below) or the SSH tunnel above.

## GCP VM: make the public IP static

Current VM information:

```text
VM:            kltn-sdn
Zone:          asia-southeast1-b
Region:        asia-southeast1
Current IP:    34.21.244.114
IP type:       Ephemeral
Orchestrator:  127.0.0.1:9000
```

Google Cloud supports either reserving a new static external IPv4 address or promoting an existing ephemeral external IP to a static address.

### Console (recommended for your VM)

```text
Google Cloud Console
→ Compute Engine
→ VM instances
→ kltn-sdn
→ Edit
→ Network interfaces
→ External IPv4 address
→ select/reserve a static external IPv4
→ Save
```

Verify after saving that the public address is marked **Static**, not **Ephemeral**.

Official documentation:
https://docs.cloud.google.com/compute/docs/ip-addresses/configure-static-external-ip-address

### gcloud: reserve a new regional static IPv4

Your VM is in `asia-southeast1-b`, so the region is `asia-southeast1`:

```bash
gcloud compute addresses create ndt-api-ip \
  --region=asia-southeast1

gcloud compute addresses describe ndt-api-ip \
  --region=asia-southeast1
```

Then assign the reserved address to `kltn-sdn`. When replacing an existing external IP, Google Cloud requires the existing external address to be removed before assigning the new one.

Official reservation documentation:
https://docs.cloud.google.com/vpc/docs/reserve-static-external-ip-address

For your KLTN, the Console route is the safest way to avoid mixing up the regional address and VM network interface.

## Static IP is not HTTPS

The production path should be:

```text
Claude Dashboard
  ↓ HTTPS
api.<your-domain>
  ↓
GCP static public IP
  ↓ :443
Nginx/Caddy or HTTPS Load Balancer
  ↓
127.0.0.1:9000
  ↓
FastAPI Orchestrator
```

Do not use `http://34.x.x.x:9000` as the dashboard backend: it is plain HTTP and `POST /api/v1/migrations` has no authentication.

## DNS

For a domain such as `example.com`, create:

```text
api.example.com  A  <GCP static public IPv4>
```

Configure HTTPS for the subdomain, then enter `https://api.example.com` as **Orchestrator API base URL** in the dashboard Settings. This also needs CORS for the dashboard origin (FastAPI `CORSMiddleware`, GET only) and authentication in front of the POST endpoints; neither exists in the orchestrator today.

## Public GitHub safety

Do not commit:

```text
.env
API keys
passwords
tokens
SSH private keys
GCP service-account JSON
```

A public domain or public IP is not itself a credential, but you do not need to hard-code the IP when a domain is available.
