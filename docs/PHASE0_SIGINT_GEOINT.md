# MEMSHADOW Phase-0 SIGINT/GEOINT

**Turn Your Servers Into Intelligence Sensors**

MEMSHADOW Phase-0 enables **SIGINT-lite + GEOINT-lite** capabilities using only server logs and OSINT. No additional hardware or network taps required—just transform existing access logs, auth logs, and network flows into actionable intelligence.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Components](#components)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [API Reference](#api-reference)
8. [Query Examples](#query-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

### What You Get

From **servers you already own**, Phase-0 extracts:

* **Inbound HTTP(S) metadata** - Caddy/Nginx/Apache logs: `src_ip`, `path`, `UA`, `status`, bytes
* **Outbound connection metadata** - Periodic `ss`/`conntrack` snapshots: `local_ip:port → remote_ip:port`
* **Auth/access events** - SSH logins, sudo commands, app authentication
* **Service-specific logs** - Matrix, GitLab, APIs—all remote IPs and timestamps

Combined with **OSINT**:

* **IP → Geo/ASN** via GeoLite2 databases
* **AOIs (Areas of Interest)** - Define countries, cities, regions using GeoJSON
* **Point-in-polygon matching** - Automatically tag observations by geographic region

### Core Concepts

#### OBSERVATION
A single intelligence event (HTTP request, network flow, auth event, etc.) enriched with:
- Geographic location (via GeoIP)
- AOI memberships (via point-in-polygon)
- Risk scoring
- Semantic embeddings (for similarity search)

#### DEVICE
An endpoint (IP, MAC, hostname) observed in network traffic. Tracks first/last seen, risk score.

#### IDENTITY
A user account (email, username, OAuth identity) seen in auth events.

#### AOI (Area of Interest)
A geographic region defined by a GeoJSON polygon. Observations are matched against AOIs automatically.

---

## Architecture

```
┌─────────────────┐
│  Server Logs    │
│  - Caddy JSON   │
│  - Nginx        │
│  - Auth logs    │
│  - Netflows     │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ sigint-net-agent│  <-- Tails logs, creates OBSERVATIONs
└────────┬────────┘
         │
         v
┌─────────────────┐
│ MEMSHADOW API   │
│ /api/v1/sigint  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  geo-enricher   │  <-- Adds GeoIP + AOI data
└────────┬────────┘
         │
         v
┌─────────────────┐
│  PostgreSQL     │  <-- Stores observations + nodes
│  ChromaDB       │  <-- Vector search
└─────────────────┘
```

### Data Flow

1. **Log Event** → `sigint-net-agent` parses into OBSERVATION JSON
2. **OBSERVATION** → Posted to MEMSHADOW API `/api/v1/sigint/observations`
3. **geo-enricher** → Enriches with GeoIP location + AOI memberships
4. **Storage** → Saved to PostgreSQL with 2048d embedding for semantic search
5. **Query** → Retrieve by time, AOI, risk score, device, etc.

---

## Quick Start

### 1. Prerequisites

```bash
# Install Python dependencies
pip install geoip2 shapely aiohttp

# Download GeoLite2 databases (requires free MaxMind account)
# https://dev.maxmind.com/geoip/geolite2-free-geolocation-data

mkdir -p /var/lib/memshadow/geoip
# Place GeoLite2-City.mmdb and GeoLite2-ASN.mmdb in the directory above
```

### 2. Run Database Migrations

```bash
cd /home/user/MEMSHADOW

# Generate migration (if not exists)
alembic revision --autogenerate -m "Add Phase-0 SIGINT/GEOINT tables"

# Apply migration
alembic upgrade head
```

### 3. Configure Environment

Add to `.env`:

```bash
# Enable SIGINT/GEOINT subsystem
ENABLE_SIGINT_GEOINT=true

# GeoIP database paths
GEOIP_CITY_DB_PATH=/var/lib/memshadow/geoip/GeoLite2-City.mmdb
GEOIP_ASN_DB_PATH=/var/lib/memshadow/geoip/GeoLite2-ASN.mmdb

# Auto-load AOIs on startup
SIGINT_AUTO_LOAD_AOIS=true

# Observation retention (days)
SIGINT_OBSERVATION_RETENTION_DAYS=90
```

### 4. Deploy sigint-net-agent

On each server you want to monitor:

```bash
# Copy agent to server
scp agents/sigint_net_agent.py user@server:/usr/local/bin/

# Set environment variables
export MEMSHADOW_API_URL=https://memshadow.yourdomain.com
export MEMSHADOW_API_TOKEN=your_jwt_token_here

# Tail Caddy JSON logs
python3 /usr/local/bin/sigint_net_agent.py \
  --source caddy \
  --log-file /var/log/caddy/access.json

# Or capture network flows (requires root)
sudo python3 /usr/local/bin/sigint_net_agent.py \
  --source netflow \
  --interval 60
```

### 5. Create AOIs

```bash
curl -X POST https://memshadow.yourdomain.com/api/v1/sigint/aois \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Beirut",
    "category": "CITY",
    "description": "Beirut, Lebanon",
    "labels": {
      "country": "LB"
    },
    "geometry": {
      "type": "Polygon",
      "crs": "EPSG:4326",
      "coordinates": [
        [
          [35.45, 33.84],
          [35.62, 33.84],
          [35.62, 33.96],
          [35.45, 33.96],
          [35.45, 33.84]
        ]
      ]
    }
  }'
```

### 6. Query Observations

```bash
# Get recent observations from Beirut AOI
curl "https://memshadow.yourdomain.com/api/v1/sigint/observations?\
aoi_ids=aoi:beirut&limit=100" \
  -H "Authorization: Bearer $TOKEN"

# Get SSH failures in last 24 hours
curl "https://memshadow.yourdomain.com/api/v1/sigint/observations?\
channel=AUTH_EVENT&\
start_time=2025-12-01T00:00:00Z&\
min_risk_score=0.3" \
  -H "Authorization: Bearer $TOKEN"
```

---

## Components

### 1. sigint-net-agent

**Location:** `/agents/sigint_net_agent.py`

**Purpose:** Runs on each monitored server. Tails logs and sends OBSERVATIONs to MEMSHADOW.

**Supported Sources:**

| Source   | Description                    | Log Format       |
|----------|--------------------------------|------------------|
| `caddy`  | Caddy web server access logs   | JSON             |
| `nginx`  | Nginx access logs              | Common/Combined  |
| `auth`   | Linux auth logs (SSH, sudo)    | Syslog           |
| `netflow`| Network flows via `ss`         | Structured       |

**Usage:**

```bash
# Caddy
python3 sigint_net_agent.py --source caddy --log-file /var/log/caddy/access.json

# Nginx
python3 sigint_net_agent.py --source nginx --log-file /var/log/nginx/access.log

# Auth logs
python3 sigint_net_agent.py --source auth --log-file /var/log/auth.log

# Network flows (every 60s)
sudo python3 sigint_net_agent.py --source netflow --interval 60
```

**Environment Variables:**

- `MEMSHADOW_API_URL` - MEMSHADOW API endpoint (default: `http://localhost:8000`)
- `MEMSHADOW_API_TOKEN` - Bearer token for authentication
- `SIGINT_AGENT_HOSTNAME` - Override hostname (default: `socket.gethostname()`)

### 2. geo-enricher Service

**Location:** `/app/services/sigint/geo_enricher.py`

**Purpose:** Enriches observations with GeoIP data and matches them against AOIs.

**Features:**

- **GeoIP Lookup:** Translates IP → `{lat, lon, city, country, ASN, as_org}`
- **Point-in-Polygon:** Checks if observation location falls within any AOI
- **Caching:** Prepared geometries for fast point-in-polygon checks
- **Automatic Enrichment:** All observations are enriched on ingestion

**Dependencies:**

```bash
pip install geoip2 shapely
```

### 3. API Endpoints

**Location:** `/app/api/v1/sigint.py`

**Endpoints:**

| Method | Endpoint                          | Description                          |
|--------|-----------------------------------|--------------------------------------|
| POST   | `/api/v1/sigint/observations`     | Create new observation               |
| GET    | `/api/v1/sigint/observations`     | Query observations with filters      |
| GET    | `/api/v1/sigint/observations/{id}`| Get observation by ID                |
| GET    | `/api/v1/sigint/observations/stats/summary` | Get statistics     |
| POST   | `/api/v1/sigint/devices`          | Create device node                   |
| GET    | `/api/v1/sigint/devices`          | List devices                         |
| POST   | `/api/v1/sigint/aois`             | Create Area of Interest              |
| GET    | `/api/v1/sigint/aois`             | List AOIs                            |
| GET    | `/api/v1/sigint/aois/{id}/observations` | Get observations in AOI    |
| POST   | `/api/v1/sigint/enrich/geoip`     | Manually enrich IP with GeoIP        |
| GET    | `/api/v1/sigint/health`           | Health check                         |

---

## Configuration

### Environment Variables

Add to `.env`:

```bash
# ============================================================
# Phase-0 SIGINT/GEOINT Configuration
# ============================================================

# Enable SIGINT/GEOINT subsystem
ENABLE_SIGINT_GEOINT=true

# GeoIP Database Paths
# Download from: https://dev.maxmind.com/geoip/geolite2-free-geolocation-data
GEOIP_CITY_DB_PATH=/var/lib/memshadow/geoip/GeoLite2-City.mmdb
GEOIP_ASN_DB_PATH=/var/lib/memshadow/geoip/GeoLite2-ASN.mmdb

# Auto-load AOIs on startup
SIGINT_AUTO_LOAD_AOIS=true

# Default observation retention period (days)
SIGINT_OBSERVATION_RETENTION_DAYS=90
```

### GeoIP Databases

1. **Sign up for MaxMind account** (free): https://www.maxmind.com/en/geolite2/signup
2. **Download databases**:
   - GeoLite2-City.mmdb
   - GeoLite2-ASN.mmdb
3. **Place in `/var/lib/memshadow/geoip/`**

Update `.env` if using different paths.

---

## Deployment

### Systemd Service for sigint-net-agent

Create `/etc/systemd/system/sigint-net-agent-caddy.service`:

```ini
[Unit]
Description=MEMSHADOW SIGINT Network Agent (Caddy)
After=network.target

[Service]
Type=simple
User=memshadow
WorkingDirectory=/opt/memshadow
Environment="MEMSHADOW_API_URL=https://memshadow.yourdomain.com"
Environment="MEMSHADOW_API_TOKEN=your_token_here"
ExecStart=/usr/bin/python3 /usr/local/bin/sigint_net_agent.py \
  --source caddy \
  --log-file /var/log/caddy/access.json
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sigint-net-agent-caddy
sudo systemctl start sigint-net-agent-caddy
sudo systemctl status sigint-net-agent-caddy
```

### Docker Deployment

If running MEMSHADOW in Docker, mount GeoIP databases:

```yaml
# docker-compose.yml
services:
  memshadow:
    volumes:
      - ./geoip:/var/lib/memshadow/geoip:ro
```

Place `.mmdb` files in `./geoip/` directory.

---

## API Reference

### Create Observation

```bash
POST /api/v1/sigint/observations
```

**Request Body:**

```json
{
  "modality": "SIGINT",
  "source": "sigint-net-agent",
  "timestamp": "2025-12-02T03:22:01Z",
  "host": "web01",
  "sensor_id": "web01-caddy",
  "labels": {
    "channel": "HTTP_ACCESS",
    "service": "caddy"
  },
  "subjects": [
    {
      "type": "DEVICE",
      "id": "ip:203.0.113.10"
    }
  ],
  "payload": {
    "src_ip": "203.0.113.10",
    "dst_ip": "198.51.100.5",
    "dst_port": 443,
    "method": "GET",
    "path": "/_matrix/client/v3/sync",
    "status": 200,
    "bytes": 18342,
    "user_agent": "Mozilla/5.0 ..."
  },
  "signals": {
    "risk_score": 0.18,
    "confidence": 0.9
  }
}
```

**Response:** `201 Created`

```json
{
  "id": "a3f1b2c4-...",
  "node_id": "obs:2025-12-02T03:22:01Z:web01:HTTP_ACCESS:203.0.113.10",
  "user_id": "...",
  "modality": "SIGINT",
  "location": {
    "lat": 40.7128,
    "lon": -74.0060,
    "accuracy_m": 50000,
    "city": "New York",
    "country": "US",
    "asn": 15169,
    "as_org": "GOOGLE"
  },
  "aoi_memberships": ["aoi:usa-northeast"],
  ...
}
```

### Query Observations

```bash
GET /api/v1/sigint/observations?modality=SIGINT&start_time=2025-12-01T00:00:00Z&limit=100
```

**Query Parameters:**

| Parameter       | Type     | Description                               |
|-----------------|----------|-------------------------------------------|
| `modality`      | string   | Filter by modality (SIGINT, GEOINT, etc.) |
| `channel`       | string   | Filter by channel (HTTP_ACCESS, etc.)     |
| `source`        | string   | Filter by source                          |
| `host`          | string   | Filter by host                            |
| `start_time`    | datetime | Filter by start time (ISO 8601)           |
| `end_time`      | datetime | Filter by end time                        |
| `aoi_ids`       | string   | Comma-separated AOI node IDs              |
| `min_risk_score`| float    | Minimum risk score (0.0-1.0)              |
| `limit`         | int      | Max results (1-10000, default: 100)       |
| `offset`        | int      | Pagination offset (default: 0)            |

### Create AOI

```bash
POST /api/v1/sigint/aois
```

**Request Body:**

```json
{
  "name": "Beirut",
  "category": "CITY",
  "description": "Beirut, Lebanon",
  "labels": {
    "country": "LB"
  },
  "geometry": {
    "type": "Polygon",
    "crs": "EPSG:4326",
    "coordinates": [
      [
        [35.45, 33.84],
        [35.62, 33.84],
        [35.62, 33.96],
        [35.45, 33.96],
        [35.45, 33.84]
      ]
    ]
  }
}
```

**Response:** `201 Created`

---

## Query Examples

### 1. All observations from Beirut in last 24 hours

```bash
curl "https://memshadow.yourdomain.com/api/v1/sigint/observations?\
aoi_ids=aoi:beirut&\
start_time=$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%SZ)" \
  -H "Authorization: Bearer $TOKEN"
```

### 2. Failed SSH attempts (high risk)

```bash
curl "https://memshadow.yourdomain.com/api/v1/sigint/observations?\
channel=AUTH_EVENT&\
min_risk_score=0.3&\
limit=1000" \
  -H "Authorization: Bearer $TOKEN"
```

### 3. All devices observed in last week

```bash
curl "https://memshadow.yourdomain.com/api/v1/sigint/devices?limit=1000" \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Timeline of activity for specific IP

```bash
# First find device
DEVICE_ID="ip:203.0.113.10"

curl "https://memshadow.yourdomain.com/api/v1/sigint/observations?\
subject_ids=$DEVICE_ID&\
start_time=2025-12-01T00:00:00Z" \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Statistics summary

```bash
curl "https://memshadow.yourdomain.com/api/v1/sigint/observations/stats/summary?\
start_time=2025-12-01T00:00:00Z&\
end_time=2025-12-02T00:00:00Z" \
  -H "Authorization: Bearer $TOKEN"
```

**Example Response:**

```json
{
  "total_observations": 15234,
  "by_modality": {
    "SIGINT": 15000,
    "OSINT": 234
  },
  "by_channel": {
    "HTTP_ACCESS": 12000,
    "NETFLOW": 2500,
    "AUTH_EVENT": 734
  },
  "time_range": {
    "start": "2025-12-01T00:00:00Z",
    "end": "2025-12-02T00:00:00Z"
  }
}
```

---

## Troubleshooting

### GeoIP databases not found

```
ERROR: GeoIP City database not found: /var/lib/memshadow/geoip/GeoLite2-City.mmdb
```

**Solution:**

1. Download databases from MaxMind (requires free account)
2. Place `.mmdb` files in `/var/lib/memshadow/geoip/`
3. Update `GEOIP_CITY_DB_PATH` and `GEOIP_ASN_DB_PATH` in `.env`

### sigint-net-agent can't connect to MEMSHADOW API

```
ERROR: Failed to post observation: Connection refused
```

**Solution:**

1. Verify `MEMSHADOW_API_URL` is correct
2. Check firewall rules
3. Verify MEMSHADOW is running: `curl https://memshadow.yourdomain.com/health`
4. Check authentication token is valid

### No observations appearing

**Check agent logs:**

```bash
sudo journalctl -u sigint-net-agent-caddy -f
```

**Common issues:**

- Log file path incorrect
- Log file permissions (agent can't read)
- Log format doesn't match parser (e.g., using Nginx parser on Caddy logs)
- API authentication failing

### AOI matching not working

```
INFO: Loaded AOI: aoi:beirut (Beirut)
```

If you see this in logs but observations still have empty `aoi_memberships`:

1. **Check geometry format** - Must be valid GeoJSON
2. **Verify coordinates** - GeoJSON uses `[lon, lat]` not `[lat, lon]`
3. **Check CRS** - Default is `EPSG:4326` (WGS84)
4. **Test manually:**

```bash
curl -X POST https://memshadow.yourdomain.com/api/v1/sigint/enrich/geoip \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ip_address": "8.8.8.8"}'
```

---

## Next Steps

### Phase-1: Advanced Analytics

- **Clustering** - Geographic clustering of observations
- **Timeline analysis** - Activity patterns over time
- **Anomaly detection** - Unusual access patterns
- **Threat correlation** - Link observations to known threat actors

### Phase-2: Expanded Collection

- **DNS queries** - Capture outbound DNS lookups
- **TLS metadata** - SNI, certificate fingerprints
- **Custom protocols** - Extend parsers for proprietary logs

### Phase-3: Integration

- **MCP endpoints** - Memory Context Protocol for LLM integration
- **Alerting** - Real-time alerts for high-risk observations
- **Dashboards** - Grafana dashboards for visualization
- **Export** - STIX/TAXII export for threat intel sharing

---

## Support

For questions or issues:

- **GitHub Issues**: https://github.com/SWORDIntel/MEMSHADOW/issues
- **Documentation**: `/docs/`
- **API Docs**: https://memshadow.yourdomain.com/api/docs

---

**Built with MEMSHADOW v2.0 - Production-Grade Memory Persistence Platform**
