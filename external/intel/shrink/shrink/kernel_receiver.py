#!/usr/bin/env python3
"""
Userspace SHRINK kernel receiver for MEMSHADOW v2.

Listens on a Netlink socket for raw MEMSHADOW binary messages produced by the
SHRINK kernel module and forwards validated payloads to the Brain ingest API:

    POST /api/v1/ingest/shrink  (Content-Type: application/octet-stream)

The receiver performs lightweight validation (magic/version/msg_type/length)
before relaying traffic to keep the kernel path simple while ensuring the
Brain only sees well-formed messages.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests


LOGGER = logging.getLogger("shrink.kernel_receiver")

# Resolve the protocol library
LIB_PATH = (
    Path(__file__).resolve()
    .parents[4]
    .joinpath("libs", "memshadow-protocol", "python")
)
if LIB_PATH.exists() and str(LIB_PATH) not in sys.path:
    sys.path.insert(0, str(LIB_PATH))

try:
    from dsmil_protocol import (
        HEADER_SIZE,
        MemshadowHeader,
        MessageType,
        Priority,
    )
except ImportError as exc:  # pragma: no cover - only during misconfiguration
    raise SystemExit(
        "Unable to import dsmil_protocol. Make sure libs/memshadow-protocol/python "
        "is available on PYTHONPATH."
    ) from exc


PSYCH_MESSAGE_TYPES: Iterable[MessageType] = (
    MessageType.PSYCH_ASSESSMENT,
    MessageType.DARK_TRIAD_UPDATE,
    MessageType.RISK_UPDATE,
    MessageType.NEURO_UPDATE,
    MessageType.TMI_UPDATE,
    MessageType.COGNITIVE_UPDATE,
    MessageType.FULL_PSYCH,
    MessageType.PSYCH_THREAT_ALERT,
    MessageType.PSYCH_ANOMALY,
    MessageType.PSYCH_RISK_THRESHOLD,
)


@dataclass
class ReceiverConfig:
    """Runtime configuration for the SHRINK receiver."""

    brain_url: str = "http://127.0.0.1:8080/api/v1/ingest/shrink"
    connect_timeout: float = 2.5
    request_timeout: float = 5.0
    netlink_proto: int = socket.NETLINK_USERSOCK
    netlink_groups: int = 0
    recv_buffer_bytes: int = 262_144
    enable_retry_backoff: bool = True


class ShrinkKernelReceiver:
    """Bridge Netlink-delivered MEMSHADOW messages into the Brain HTTP API."""

    NLMSG_HDRLEN = struct.calcsize("IHHII")

    def __init__(self, config: Optional[ReceiverConfig] = None):
        self.config = config or ReceiverConfig()
        self._sock: Optional[socket.socket] = None
        self._running = threading.Event()
        self._session = requests.Session()

    def start(self) -> None:
        """Start the Netlink receive loop."""
        if self._running.is_set():
            return
        self._running.set()
        self._sock = socket.socket(
            socket.AF_NETLINK,
            socket.SOCK_RAW,
            self.config.netlink_proto,
        )
        # Increase buffer to avoid drop during bursts
        self._sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, self.config.recv_buffer_bytes
        )
        self._sock.bind((os.getpid(), self.config.netlink_groups))
        LOGGER.info(
            "SHRINK receiver listening on Netlink proto=%s groups=0x%X",
            self.config.netlink_proto,
            self.config.netlink_groups,
        )

        while self._running.is_set():
            try:
                data = self._sock.recv(self.config.recv_buffer_bytes)
                if not data:
                    continue
                self._handle_netlink_message(data)
            except OSError as exc:
                if self._running.is_set():
                    LOGGER.error("Netlink receive failed: %s", exc, exc_info=True)
                time.sleep(0.5)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Unhandled receiver error: %s", exc)
                time.sleep(0.25)

    def stop(self) -> None:
        """Stop the receiver loop."""
        self._running.clear()
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # ------------------------------------------------------------------ helpers
    def _handle_netlink_message(self, data: bytes) -> None:
        """Parse a Netlink frame and forward MEMSHADOW payloads."""
        if len(data) < self.NLMSG_HDRLEN + HEADER_SIZE:
            LOGGER.debug("Dropping short Netlink frame (%d bytes)", len(data))
            return

        length, msg_type, flags, seq, pid = struct.unpack(
            "IHHII", data[: self.NLMSG_HDRLEN]
        )
        payload = data[self.NLMSG_HDRLEN : length]
        if len(payload) < HEADER_SIZE:
            LOGGER.debug(
                "Dropping Netlink payload shorter than MEMSHADOW header (%d bytes)",
                len(payload),
            )
            return

        try:
            header = MemshadowHeader.unpack(payload[:HEADER_SIZE])
        except ValueError as exc:
            if "magic" in str(exc).lower():
                LOGGER.warning("Invalid MEMSHADOW magic from pid=%s: %s", pid, exc)
            else:
                LOGGER.warning("Failed to parse MEMSHADOW header: %s", exc)
            return

        if header.msg_type not in PSYCH_MESSAGE_TYPES:
            LOGGER.info(
                "Ignoring unsupported msg_type=%s priority=%s",
                header.msg_type.name,
                getattr(header.priority, "name", str(header.priority)),
            )
            return

        expected_payload = HEADER_SIZE + header.payload_len
        if len(payload) < expected_payload:
            LOGGER.warning(
                "Payload length mismatch (expected %d, got %d) msg_type=%s",
                header.payload_len,
                len(payload) - HEADER_SIZE,
                header.msg_type.name,
            )
            return

        message_bytes = payload[:expected_payload]
        self._forward_to_brain(message_bytes, header)

    def _forward_to_brain(self, message: bytes, header: MemshadowHeader) -> None:
        """Send a validated MEMSHADOW message to the Brain HTTP API."""
        try:
            resp = self._session.post(
                self.config.brain_url,
                data=message,
                headers={"Content-Type": "application/octet-stream"},
                timeout=(
                    self.config.connect_timeout,
                    self.config.request_timeout,
                ),
            )
            resp.raise_for_status()
            LOGGER.debug(
                "Forwarded %s (%d bytes) priority=%s",
                header.msg_type.name,
                len(message),
                header.priority.name if isinstance(header.priority, Priority) else header.priority,
            )
        except requests.RequestException as exc:
            LOGGER.error(
                "Failed to forward MEMSHADOW message to %s: %s",
                self.config.brain_url,
                exc,
            )
            if self.config.enable_retry_backoff:
                time.sleep(0.5)


# --------------------------------------------------------------------------- CLI
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHRINK MEMSHADOW receiver")
    parser.add_argument(
        "--brain-url",
        default=ReceiverConfig.brain_url,
        help="Brain ingest endpoint URL (default: %(default)s)",
    )
    parser.add_argument(
        "--netlink-proto",
        type=int,
        default=ReceiverConfig.netlink_proto,
        help="Netlink protocol number (default: %(default)s)",
    )
    parser.add_argument(
        "--netlink-groups",
        type=lambda x: int(x, 0),
        default=ReceiverConfig.netlink_groups,
        help="Netlink multicast groups bitmask (default: %(default)s)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    config = ReceiverConfig(
        brain_url=args.brain_url,
        netlink_proto=args.netlink_proto,
        netlink_groups=args.netlink_groups,
    )
    receiver = ShrinkKernelReceiver(config)
    try:
        receiver.start()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown
        LOGGER.info("SHRINK receiver interrupted, shutting down")
    finally:
        receiver.stop()


if __name__ == "__main__":  # pragma: no cover - manual execution only
    main()
