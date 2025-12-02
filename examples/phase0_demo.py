#!/usr/bin/env python3
"""
MEMSHADOW Phase-0 SIGINT/GEOINT Demo

Demonstrates:
1. Creating AOIs (Areas of Interest)
2. Simulating observations
3. Querying observations by AOI and time
4. Getting statistics
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any

import aiohttp


# Configuration
MEMSHADOW_API_URL = "http://localhost:8000"
API_TOKEN = ""  # Set your JWT token here


async def create_aoi(session: aiohttp.ClientSession, name: str, geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Create an Area of Interest."""
    url = f"{MEMSHADOW_API_URL}/api/v1/sigint/aois"

    aoi_data = {
        "name": name,
        "category": "CITY",
        "description": f"{name} area of interest",
        "labels": {},
        "geometry": geometry
    }

    async with session.post(url, json=aoi_data) as resp:
        if resp.status in [200, 201]:
            result = await resp.json()
            print(f"✓ Created AOI: {result['node_id']} ({name})")
            return result
        else:
            text = await resp.text()
            print(f"✗ Failed to create AOI {name}: {resp.status} - {text}")
            return {}


async def create_observation(session: aiohttp.ClientSession, obs_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create an observation."""
    url = f"{MEMSHADOW_API_URL}/api/v1/sigint/observations"

    async with session.post(url, json=obs_data) as resp:
        if resp.status in [200, 201]:
            result = await resp.json()
            print(f"✓ Created observation: {result['node_id']}")
            return result
        else:
            text = await resp.text()
            print(f"✗ Failed to create observation: {resp.status} - {text}")
            return {}


async def query_observations(
    session: aiohttp.ClientSession,
    **params
) -> list:
    """Query observations with filters."""
    url = f"{MEMSHADOW_API_URL}/api/v1/sigint/observations"

    async with session.get(url, params=params) as resp:
        if resp.status == 200:
            results = await resp.json()
            print(f"✓ Found {len(results)} observations")
            return results
        else:
            text = await resp.text()
            print(f"✗ Failed to query observations: {resp.status} - {text}")
            return []


async def get_stats(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Get observation statistics."""
    url = f"{MEMSHADOW_API_URL}/api/v1/sigint/observations/stats/summary"

    async with session.get(url) as resp:
        if resp.status == 200:
            stats = await resp.json()
            print(f"✓ Statistics retrieved")
            return stats
        else:
            text = await resp.text()
            print(f"✗ Failed to get stats: {resp.status} - {text}")
            return {}


async def main():
    """Run Phase-0 demo."""
    print("=" * 60)
    print("MEMSHADOW Phase-0 SIGINT/GEOINT Demo")
    print("=" * 60)
    print()

    # Set up session
    headers = {}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    async with aiohttp.ClientSession(headers=headers) as session:

        # Step 1: Create AOIs
        print("Step 1: Creating Areas of Interest (AOIs)")
        print("-" * 60)

        # Beirut, Lebanon
        beirut_geom = {
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
        await create_aoi(session, "Beirut", beirut_geom)

        # New York City
        nyc_geom = {
            "type": "Polygon",
            "crs": "EPSG:4326",
            "coordinates": [
                [
                    [-74.05, 40.68],
                    [-73.90, 40.68],
                    [-73.90, 40.82],
                    [-74.05, 40.82],
                    [-74.05, 40.68]
                ]
            ]
        }
        await create_aoi(session, "New York City", nyc_geom)

        print()
        await asyncio.sleep(1)

        # Step 2: Create sample observations
        print("Step 2: Creating Sample Observations")
        print("-" * 60)

        # Simulated HTTP access from NYC
        obs1 = {
            "modality": "SIGINT",
            "source": "demo-script",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": "web01",
            "sensor_id": "web01-demo",
            "labels": {
                "channel": "HTTP_ACCESS",
                "service": "demo"
            },
            "subjects": [
                {"type": "DEVICE", "id": "ip:8.8.8.8"}
            ],
            "payload": {
                "src_ip": "8.8.8.8",  # Google DNS (will geolocate to US)
                "dst_ip": "198.51.100.5",
                "dst_port": 443,
                "method": "GET",
                "path": "/api/v1/status",
                "status": 200,
                "bytes": 1024
            },
            "signals": {
                "risk_score": 0.05
            }
        }
        await create_observation(session, obs1)

        # Simulated failed SSH attempt
        obs2 = {
            "modality": "SIGINT",
            "source": "demo-script",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": "auth01",
            "sensor_id": "auth01-demo",
            "labels": {
                "channel": "AUTH_EVENT",
                "service": "sshd",
                "outcome": "failure"
            },
            "subjects": [
                {"type": "DEVICE", "id": "ip:203.0.113.100"}
            ],
            "payload": {
                "event_type": "ssh_login_failure",
                "method": "password",
                "username": "admin",
                "src_ip": "203.0.113.100"
            },
            "signals": {
                "risk_score": 0.6,
                "threat_indicators": ["failed_auth", "common_username"]
            }
        }
        await create_observation(session, obs2)

        print()
        await asyncio.sleep(1)

        # Step 3: Query observations
        print("Step 3: Querying Observations")
        print("-" * 60)

        # Query all observations in last hour
        one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        results = await query_observations(
            session,
            start_time=one_hour_ago,
            limit=100
        )

        if results:
            print("\nSample observation:")
            print(json.dumps(results[0], indent=2, default=str)[:500] + "...")

        print()
        await asyncio.sleep(1)

        # Step 4: Get statistics
        print("Step 4: Getting Statistics")
        print("-" * 60)

        stats = await get_stats(session)
        if stats:
            print(json.dumps(stats, indent=2, default=str))

        print()
        print("=" * 60)
        print("Demo Complete!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Deploy sigint-net-agent to your servers")
        print("2. Configure GeoIP databases")
        print("3. Create AOIs for regions you want to monitor")
        print("4. Query observations via API")
        print()
        print("See docs/PHASE0_SIGINT_GEOINT.md for full documentation.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted")
