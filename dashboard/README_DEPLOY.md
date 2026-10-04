# NDT-SDN Safe Load Balancing Dashboard

Static dashboard for the KLTN **A Network Digital Twin-Assisted Framework for Safe Load Balancing in Multi-Controller SDN**.

## Files

```text
dashboard/
├── index.html
├── vercel.json
└── README_DEPLOY.md
```

No React/Vite build is required.

## Current LIVE API contract

The current FastAPI Orchestrator exposes:

```text
GET  /health
GET  /api/v1/state
GET  /api/v1/twin/state
POST /api/v1/init-roles
POST /api/v1/migrations
```

The dashboard uses `/api/v1/twin/state` for live Twin/telemetry data and `/api/v1/state` for transaction/ownership/error information. It does not invent LIVE results for Predictor/Candidate/Safety/Calibration/Experiments APIs that are not currently exposed.

To inspect every UI module with simulated data:

```text
https://<your-vercel-domain>/?demo=1
```

## Local test

```bash
cd dashboard
python -m http.server 8080
```

Demo:

```text
http://localhost:8080/?demo=1
```

Direct backend test:

```text
http://localhost:8080/?api=http://127.0.0.1:9000
```

## GitHub

Place this folder in the KLTN repository:

```text
ndt-safe-load-balancing/
└── dashboard/
    ├── index.html
    ├── vercel.json
    └── README_DEPLOY.md
```

Then:

```bash
git add dashboard/
git commit -m "feat: finalize NDT dashboard"
git push origin main
```

## Vercel

1. Add New → Project.
2. Import the GitHub repository.
3. **Root Directory = `dashboard`**.
4. Framework Preset = **Other**.
5. Build Command = empty.
6. Output Directory = `.`.
7. Deploy.

## Vercel → GCP API proxy

`vercel.json` intentionally contains:

```text
https://YOUR-BACKEND-DOMAIN.example.com
```

Replace it after the GCP backend has a real HTTPS domain. Example:

```json
{
  "headers": [
    {
      "source": "/api/:path*",
      "headers": [
        {
          "key": "Cache-Control",
          "value": "no-store, max-age=0"
        }
      ]
    }
  ],
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://api.example.com/api/:path*"
    }
  ]
}
```

The frontend continues to call:

```javascript
fetch('/api/v1/twin/state')
```

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
Vercel
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

Do not use `http://34.x.x.x:9000` as the final Vercel backend.

## DNS

For a domain such as `example.com`, create:

```text
api.example.com  A  <GCP static public IPv4>
```

Configure HTTPS for the subdomain, then put `https://api.example.com` in `vercel.json`.

## Public GitHub safety

Do not commit:

```text
.env
API keys
passwords
tokens
SSH private keys
GCP service-account JSON
Vercel tokens
```

A public domain or public IP is not itself a credential, but you do not need to hard-code the IP when a domain is available.
