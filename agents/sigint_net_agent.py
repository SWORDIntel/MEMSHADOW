#!/usr/bin/env python3
"""
MEMSHADOW SIGINT Network Agent

Tails server logs (Caddy, Nginx, auth logs, etc.) and converts them
into OBSERVATION nodes for MEMSHADOW's Phase-0 SIGINT/GEOINT capability.

Usage:
    # Tail Caddy JSON logs
    python sigint_net_agent.py --source caddy --log-file /var/log/caddy/access.json

    # Tail Nginx logs (common format)
    python sigint_net_agent.py --source nginx --log-file /var/log/nginx/access.log

    # Capture network flows (requires root)
    sudo python sigint_net_agent.py --source netflow --interval 60

    # Tail auth logs
    python sigint_net_agent.py --source auth --log-file /var/log/auth.log

Environment Variables:
    MEMSHADOW_API_URL: MEMSHADOW API endpoint (default: http://localhost:8000)
    MEMSHADOW_API_TOKEN: Bearer token for authentication
    SIGINT_AGENT_HOSTNAME: Override hostname (default: socket.gethostname())
"""

import argparse
import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

import aiohttp


# ============================================================================
# Configuration
# ============================================================================

MEMSHADOW_API_URL = os.getenv("MEMSHADOW_API_URL", "http://localhost:8000")
MEMSHADOW_API_TOKEN = os.getenv("MEMSHADOW_API_TOKEN", "")
SIGINT_AGENT_HOSTNAME = os.getenv("SIGINT_AGENT_HOSTNAME", socket.gethostname())

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("sigint-net-agent")


# ============================================================================
# Observation Builder
# ============================================================================

class ObservationBuilder:
    """Builds OBSERVATION nodes from log events."""

    @staticmethod
    def make_node_id(timestamp: str, host: str, channel: str, identifier: str) -> str:
        """Generate a unique node_id for an observation."""
        # Format: obs:TIMESTAMP:HOST:CHANNEL:IDENTIFIER
        return f"obs:{timestamp}:{host}:{channel}:{identifier}"

    @staticmethod
    def build_observation(
        modality: str,
        channel: str,
        timestamp: datetime,
        host: str,
        sensor_id: str,
        payload: Dict[str, Any],
        labels: Optional[Dict[str, str]] = None,
        subjects: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Build an OBSERVATION dict.

        Args:
            modality: SIGINT, GEOINT, OSINT, etc.
            channel: HTTP_ACCESS, NETFLOW, AUTH_EVENT, etc.
            timestamp: When the observation occurred
            host: Hostname where observation was made
            sensor_id: Sensor identifier
            payload: Raw event data
            labels: Optional labels
            subjects: Optional subject references

        Returns:
            Observation dict matching ObservationCreate schema
        """
        ts_iso = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp

        # Generate identifier from payload
        identifier = payload.get("src_ip", payload.get("username", "unknown"))

        obs = {
            "node_id": ObservationBuilder.make_node_id(ts_iso, host, channel, identifier),
            "modality": modality,
            "source": "sigint-net-agent",
            "timestamp": ts_iso,
            "host": host,
            "sensor_id": sensor_id,
            "labels": labels or {},
            "subjects": subjects or [],
            "payload": payload,
            "signals": {}
        }

        return obs


# ============================================================================
# Log Parsers
# ============================================================================

class CaddyLogParser:
    """Parses Caddy JSON access logs."""

    @staticmethod
    def parse_line(line: str, host: str) -> Optional[Dict[str, Any]]:
        """Parse a Caddy JSON log line into an observation."""
        try:
            event = json.loads(line)

            # Extract timestamp
            ts = event.get("ts")
            if ts:
                timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            # Extract request data
            request = event.get("request", {})
            remote_ip = request.get("remote_ip", request.get("client_ip", "unknown"))

            payload = {
                "src_ip": remote_ip,
                "dst_ip": event.get("server_ip", "unknown"),
                "dst_port": event.get("port", 443),
                "method": request.get("method"),
                "path": request.get("uri"),
                "status": event.get("status"),
                "bytes": event.get("size"),
                "user_agent": request.get("headers", {}).get("User-Agent", [""])[0] if isinstance(request.get("headers", {}).get("User-Agent"), list) else request.get("headers", {}).get("User-Agent", ""),
                "duration_ms": event.get("duration", 0) * 1000,
                "protocol": request.get("proto", "HTTP/1.1")
            }

            subjects = [
                {"type": "DEVICE", "id": f"ip:{remote_ip}"}
            ]

            labels = {
                "channel": "HTTP_ACCESS",
                "service": "caddy"
            }

            obs = ObservationBuilder.build_observation(
                modality="SIGINT",
                channel="HTTP_ACCESS",
                timestamp=timestamp,
                host=host,
                sensor_id=f"{host}-caddy",
                payload=payload,
                labels=labels,
                subjects=subjects
            )

            return obs

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse Caddy log line: {line[:100]}")
            return None
        except Exception as e:
            logger.error(f"Error parsing Caddy log: {e}")
            return None


class NginxLogParser:
    """Parses Nginx access logs (common/combined format)."""

    # Regex for Nginx combined log format
    NGINX_COMBINED_RE = re.compile(
        r'(?P<remote_ip>[\d\.]+) - (?P<remote_user>[\w-]+) \[(?P<time_local>[^\]]+)\] '
        r'"(?P<request>[^"]*)" (?P<status>\d+) (?P<body_bytes_sent>\d+) '
        r'"(?P<http_referer>[^"]*)" "(?P<http_user_agent>[^"]*)"'
    )

    @staticmethod
    def parse_line(line: str, host: str) -> Optional[Dict[str, Any]]:
        """Parse an Nginx access log line into an observation."""
        try:
            match = NginxLogParser.NGINX_COMBINED_RE.match(line)
            if not match:
                logger.debug(f"Nginx log line didn't match pattern: {line[:100]}")
                return None

            data = match.groupdict()

            # Parse timestamp (Nginx format: 02/Dec/2025:03:22:01 +0000)
            try:
                timestamp = datetime.strptime(data["time_local"], "%d/%b/%Y:%H:%M:%S %z")
            except ValueError:
                timestamp = datetime.now(timezone.utc)

            # Parse request line
            request_parts = data["request"].split(" ")
            method = request_parts[0] if len(request_parts) > 0 else "GET"
            path = request_parts[1] if len(request_parts) > 1 else "/"
            protocol = request_parts[2] if len(request_parts) > 2 else "HTTP/1.1"

            remote_ip = data["remote_ip"]

            payload = {
                "src_ip": remote_ip,
                "method": method,
                "path": path,
                "protocol": protocol,
                "status": int(data["status"]),
                "bytes": int(data["body_bytes_sent"]),
                "user_agent": data["http_user_agent"],
                "referer": data["http_referer"] if data["http_referer"] != "-" else None,
                "remote_user": data["remote_user"] if data["remote_user"] != "-" else None
            }

            subjects = [
                {"type": "DEVICE", "id": f"ip:{remote_ip}"}
            ]

            # Add identity subject if authenticated
            if payload["remote_user"]:
                subjects.append({"type": "IDENTITY", "id": f"identity:acct:{payload['remote_user']}"})

            labels = {
                "channel": "HTTP_ACCESS",
                "service": "nginx"
            }

            obs = ObservationBuilder.build_observation(
                modality="SIGINT",
                channel="HTTP_ACCESS",
                timestamp=timestamp,
                host=host,
                sensor_id=f"{host}-nginx",
                payload=payload,
                labels=labels,
                subjects=subjects
            )

            return obs

        except Exception as e:
            logger.error(f"Error parsing Nginx log: {e}")
            return None


class AuthLogParser:
    """Parses Linux auth logs for SSH/sudo events."""

    # Regex patterns for auth events
    SSH_SUCCESS_RE = re.compile(r'Accepted (?P<method>\w+) for (?P<user>\w+) from (?P<ip>[\d\.]+)')
    SSH_FAILURE_RE = re.compile(r'Failed (?P<method>\w+) for (?P<user>\w+) from (?P<ip>[\d\.]+)')
    SUDO_RE = re.compile(r'(?P<user>\w+) : TTY=(?P<tty>\S+) ; PWD=(?P<pwd>\S+) ; USER=(?P<target_user>\w+) ; COMMAND=(?P<command>.+)')

    @staticmethod
    def parse_line(line: str, host: str) -> Optional[Dict[str, Any]]:
        """Parse an auth log line into an observation."""
        try:
            # SSH success
            match = AuthLogParser.SSH_SUCCESS_RE.search(line)
            if match:
                data = match.groupdict()
                timestamp = datetime.now(timezone.utc)  # Auth logs don't always have full timestamps

                payload = {
                    "event_type": "ssh_login_success",
                    "method": data["method"],
                    "username": data["user"],
                    "src_ip": data["ip"]
                }

                subjects = [
                    {"type": "DEVICE", "id": f"ip:{data['ip']}"},
                    {"type": "IDENTITY", "id": f"identity:acct:{data['user']}"}
                ]

                labels = {
                    "channel": "AUTH_EVENT",
                    "service": "sshd",
                    "outcome": "success"
                }

                obs = ObservationBuilder.build_observation(
                    modality="SIGINT",
                    channel="AUTH_EVENT",
                    timestamp=timestamp,
                    host=host,
                    sensor_id=f"{host}-auth",
                    payload=payload,
                    labels=labels,
                    subjects=subjects
                )

                return obs

            # SSH failure
            match = AuthLogParser.SSH_FAILURE_RE.search(line)
            if match:
                data = match.groupdict()
                timestamp = datetime.now(timezone.utc)

                payload = {
                    "event_type": "ssh_login_failure",
                    "method": data["method"],
                    "username": data["user"],
                    "src_ip": data["ip"]
                }

                subjects = [
                    {"type": "DEVICE", "id": f"ip:{data['ip']}"}
                ]

                labels = {
                    "channel": "AUTH_EVENT",
                    "service": "sshd",
                    "outcome": "failure"
                }

                # Mark as higher risk
                signals = {
                    "risk_score": 0.3,
                    "threat_indicators": ["failed_auth"]
                }

                obs = ObservationBuilder.build_observation(
                    modality="SIGINT",
                    channel="AUTH_EVENT",
                    timestamp=timestamp,
                    host=host,
                    sensor_id=f"{host}-auth",
                    payload=payload,
                    labels=labels,
                    subjects=subjects
                )
                obs["signals"] = signals

                return obs

            # Sudo commands
            match = AuthLogParser.SUDO_RE.search(line)
            if match:
                data = match.groupdict()
                timestamp = datetime.now(timezone.utc)

                payload = {
                    "event_type": "sudo_command",
                    "user": data["user"],
                    "target_user": data["target_user"],
                    "command": data["command"],
                    "pwd": data["pwd"],
                    "tty": data["tty"]
                }

                subjects = [
                    {"type": "IDENTITY", "id": f"identity:acct:{data['user']}"}
                ]

                labels = {
                    "channel": "AUTH_EVENT",
                    "service": "sudo"
                }

                obs = ObservationBuilder.build_observation(
                    modality="SIGINT",
                    channel="AUTH_EVENT",
                    timestamp=timestamp,
                    host=host,
                    sensor_id=f"{host}-auth",
                    payload=payload,
                    labels=labels,
                    subjects=subjects
                )

                return obs

            return None

        except Exception as e:
            logger.error(f"Error parsing auth log: {e}")
            return None


class NetflowCapture:
    """Captures network flow data using ss/netstat."""

    @staticmethod
    async def capture_flows(host: str) -> List[Dict[str, Any]]:
        """
        Capture current network flows using 'ss -tn'.

        Returns:
            List of observations for each active connection
        """
        observations = []

        try:
            # Run ss command to get TCP connections
            result = subprocess.run(
                ["ss", "-tn", "state", "established"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                logger.error(f"ss command failed: {result.stderr}")
                return observations

            # Parse ss output
            # Format: State Recv-Q Send-Q Local Address:Port Peer Address:Port
            for line in result.stdout.splitlines()[1:]:  # Skip header
                parts = line.split()
                if len(parts) < 5:
                    continue

                state = parts[0]
                local_addr = parts[3]
                peer_addr = parts[4]

                # Parse addresses
                try:
                    local_ip, local_port = local_addr.rsplit(":", 1)
                    peer_ip, peer_port = peer_addr.rsplit(":", 1)
                except ValueError:
                    continue

                # Skip localhost connections
                if peer_ip.startswith("127.") or peer_ip == "::1":
                    continue

                timestamp = datetime.now(timezone.utc)

                payload = {
                    "local_ip": local_ip,
                    "local_port": int(local_port),
                    "remote_ip": peer_ip,
                    "remote_port": int(peer_port),
                    "state": state,
                    "direction": "outbound"
                }

                subjects = [
                    {"type": "DEVICE", "id": f"ip:{peer_ip}"}
                ]

                labels = {
                    "channel": "NETFLOW",
                    "service": "ss"
                }

                obs = ObservationBuilder.build_observation(
                    modality="SIGINT",
                    channel="NETFLOW",
                    timestamp=timestamp,
                    host=host,
                    sensor_id=f"{host}-netflow",
                    payload=payload,
                    labels=labels,
                    subjects=subjects
                )

                observations.append(obs)

        except Exception as e:
            logger.error(f"Error capturing netflows: {e}")

        return observations


# ============================================================================
# MEMSHADOW API Client
# ============================================================================

class MEMSHADOWClient:
    """Client for posting observations to MEMSHADOW API."""

    def __init__(self, api_url: str, api_token: str):
        """
        Initialize MEMSHADOW client.

        Args:
            api_url: MEMSHADOW API base URL
            api_token: Bearer token for authentication
        """
        self.api_url = api_url.rstrip("/")
        self.api_token = api_token
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def post_observation(self, observation: Dict[str, Any]) -> bool:
        """
        Post an observation to MEMSHADOW.

        Args:
            observation: Observation dict

        Returns:
            True if successful, False otherwise
        """
        if not self.session:
            logger.error("Client session not initialized")
            return False

        try:
            url = f"{self.api_url}/api/v1/sigint/observations"

            async with self.session.post(url, json=observation) as resp:
                if resp.status == 200 or resp.status == 201:
                    logger.debug(f"Posted observation: {observation['node_id']}")
                    return True
                else:
                    text = await resp.text()
                    logger.error(f"Failed to post observation: {resp.status} - {text}")
                    return False

        except Exception as e:
            logger.error(f"Error posting observation: {e}")
            return False


# ============================================================================
# Log Tailer
# ============================================================================

async def tail_file(file_path: str, parser, host: str, client: MEMSHADOWClient):
    """
    Tail a log file and send observations to MEMSHADOW.

    Args:
        file_path: Path to log file
        parser: Parser instance (CaddyLogParser, NginxLogParser, etc.)
        host: Hostname
        client: MEMSHADOW client
    """
    logger.info(f"Tailing log file: {file_path}")

    try:
        with open(file_path, "r") as f:
            # Seek to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()
                if not line:
                    await asyncio.sleep(0.5)
                    continue

                # Parse line
                obs = parser.parse_line(line.strip(), host)
                if obs:
                    await client.post_observation(obs)

    except FileNotFoundError:
        logger.error(f"Log file not found: {file_path}")
    except KeyboardInterrupt:
        logger.info("Stopped tailing log file")
    except Exception as e:
        logger.error(f"Error tailing file: {e}")


async def capture_netflows_periodic(interval: int, host: str, client: MEMSHADOWClient):
    """
    Periodically capture network flows and send to MEMSHADOW.

    Args:
        interval: Capture interval in seconds
        host: Hostname
        client: MEMSHADOW client
    """
    logger.info(f"Capturing network flows every {interval}s")

    try:
        while True:
            observations = await NetflowCapture.capture_flows(host)
            logger.info(f"Captured {len(observations)} network flows")

            for obs in observations:
                await client.post_observation(obs)

            await asyncio.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Stopped capturing network flows")
    except Exception as e:
        logger.error(f"Error capturing network flows: {e}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MEMSHADOW SIGINT Network Agent")

    parser.add_argument(
        "--source",
        required=True,
        choices=["caddy", "nginx", "auth", "netflow"],
        help="Log source type"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file (required for caddy, nginx, auth)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Capture interval in seconds (for netflow)"
    )
    parser.add_argument(
        "--api-url",
        default=MEMSHADOW_API_URL,
        help="MEMSHADOW API URL"
    )
    parser.add_argument(
        "--api-token",
        default=MEMSHADOW_API_TOKEN,
        help="MEMSHADOW API token"
    )
    parser.add_argument(
        "--hostname",
        default=SIGINT_AGENT_HOSTNAME,
        help="Override hostname"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.source in ["caddy", "nginx", "auth"] and not args.log_file:
        parser.error(f"--log-file is required for source: {args.source}")

    # Select parser
    parsers = {
        "caddy": CaddyLogParser(),
        "nginx": NginxLogParser(),
        "auth": AuthLogParser()
    }

    log_parser = parsers.get(args.source)

    # Run agent
    async with MEMSHADOWClient(args.api_url, args.api_token) as client:
        if args.source == "netflow":
            await capture_netflows_periodic(args.interval, args.hostname, client)
        else:
            await tail_file(args.log_file, log_parser, args.hostname, client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped")
        sys.exit(0)
