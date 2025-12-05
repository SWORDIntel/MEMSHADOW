#!/usr/bin/env python3
"""
MEMSHADOW Protocol Ingest Plugin for DSMIL Brain

Parses MEMSHADOW binary protocol messages (v2) and extracts:
- Psychological events from SHRINK kernel module
- Improvement announcements
- Memory tier sync data

This plugin handles the binary wire format defined in:
- libs/memshadow-protocol/c/include/dsmil_protocol.h
- libs/memshadow-protocol/python/dsmil_protocol.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

# Add libs path for protocol imports
libs_path = Path(__file__).parent.parent.parent.parent.parent / "libs" / "memshadow-protocol" / "python"
if libs_path.exists() and str(libs_path) not in sys.path:
    sys.path.insert(0, str(libs_path))

logger = logging.getLogger(__name__)

try:
    from dsmil_protocol import (
        MemshadowMessage, MemshadowHeader, PsychEvent,
        MessageType, Priority, HEADER_SIZE, PSYCH_EVENT_SIZE,
        detect_protocol_version
    )
    PROTOCOL_AVAILABLE = True
except ImportError:
    PROTOCOL_AVAILABLE = False
    HEADER_SIZE = 32
    PSYCH_EVENT_SIZE = 64
    logger.warning("MEMSHADOW protocol library not available")

try:
    from mrac_registry import update_register, update_heartbeat, update_command_ack
except Exception:
    def update_register(app_id: str, name: str, capabilities: Dict[str, Any]):  # type: ignore
        return
    def update_heartbeat(app_id: str, telemetry: Dict[str, Any]):  # type: ignore
        return
    def update_command_ack(app_id: str, command_id: str, status: str):  # type: ignore
        return

try:
    from ..ingest_framework import IngestPlugin, IngestResult
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    parent = Path(__file__).parent.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    from ingest_framework import IngestPlugin, IngestResult

logger = logging.getLogger(__name__)


class MemshadowIngestPlugin(IngestPlugin):
    """
    MEMSHADOW Protocol Ingest Plugin

    Parses binary MEMSHADOW protocol messages and extracts structured data
    for storage in memory tiers.
    """

    @property
    def name(self) -> str:
        return "memshadow_ingest"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Ingest MEMSHADOW binary protocol messages (SHRINK psych events, improvements, etc.)"

    @property
    def supported_types(self) -> List[str]:
        return ["memshadow", "binary", "psych_event", "improvement"]

    def __init__(self):
        if not PROTOCOL_AVAILABLE:
            logger.error("MEMSHADOW protocol library not available - plugin disabled")
        self._enabled = PROTOCOL_AVAILABLE

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin"""
        if not PROTOCOL_AVAILABLE:
            return False
        self._enabled = config.get("enabled", True)
        logger.info(f"MemshadowIngestPlugin initialized (enabled={self._enabled})")
        return True

    def ingest(self, source: Any, **kwargs) -> IngestResult:
        """
        Ingest MEMSHADOW protocol binary data

        Args:
            source: bytes containing MEMSHADOW protocol message(s)
            **kwargs: Additional options

        Returns:
            IngestResult with parsed data and metadata
        """
        if not self._enabled:
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=["Plugin disabled - protocol library not available"],
            )

        try:
            # Get bytes
            if isinstance(source, bytes):
                data = source
            elif isinstance(source, (str, Path)):
                with open(source, "rb") as f:
                    data = f.read()
            elif hasattr(source, "read"):
                data = source.read()
            else:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=[f"Unsupported source type: {type(source)}"],
                )

            if len(data) < HEADER_SIZE:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=[f"Data too short: {len(data)} < {HEADER_SIZE}"],
                )

            # Detect protocol version
            try:
                version = detect_protocol_version(data)
            except Exception as e:
                return IngestResult(
                    success=False,
                    plugin_name=self.name,
                    errors=[f"Failed to detect protocol version: {e}"],
                )

            # Parse message(s)
            parsed_messages = []
            offset = 0
            items_ingested = 0
            bytes_processed = 0

            while offset < len(data):
                try:
                    # Try to parse a message
                    remaining = data[offset:]
                    if len(remaining) < HEADER_SIZE:
                        break

                    # Parse header
                    header = MemshadowHeader.unpack(remaining)

                    # Check if we have full message
                    total_message_size = HEADER_SIZE + header.payload_len
                    if len(remaining) < total_message_size:
                        logger.warning(f"Incomplete message at offset {offset}, need {total_message_size} bytes")
                        break

                    # Parse full message
                    message = MemshadowMessage.unpack(remaining[:total_message_size])
                    parsed_messages.append(message)

                    # Extract structured data based on message type
                    extracted_data = self._extract_message_data(message)
                    if extracted_data:
                        items_ingested += len(extracted_data) if isinstance(extracted_data, list) else 1

                    bytes_processed += total_message_size
                    offset += total_message_size

                except Exception as e:
                    logger.error(f"Error parsing message at offset {offset}: {e}")
                    # Try to skip to next potential message start
                    # Look for magic number
                    magic_pos = remaining.find(b"MSHW", 1)
                    if magic_pos > 0:
                        offset += magic_pos
                    else:
                        break

            # Build metadata
            metadata = {
                "protocol_version": version,
                "messages_parsed": len(parsed_messages),
                "bytes_processed": bytes_processed,
                "message_types": [msg.header.msg_type for msg in parsed_messages],
            }

            # Extract all data from messages
            all_extracted_data = []
            for msg in parsed_messages:
                extracted = self._extract_message_data(msg)
                if extracted:
                    if isinstance(extracted, list):
                        all_extracted_data.extend(extracted)
                    else:
                        all_extracted_data.append(extracted)

            return IngestResult(
                success=True,
                plugin_name=self.name,
                data=all_extracted_data if all_extracted_data else parsed_messages,
                metadata=metadata,
                items_ingested=items_ingested,
                bytes_processed=bytes_processed,
            )

        except Exception as e:
            logger.error(f"Error ingesting MEMSHADOW data: {e}", exc_info=True)
            return IngestResult(
                success=False,
                plugin_name=self.name,
                errors=[str(e)],
            )

    def _extract_message_data(self, message: MemshadowMessage) -> Optional[List[Dict[str, Any]]]:
        """
        Extract structured data from a MEMSHADOW message

        Returns:
            List of extracted data dictionaries, or None
        """
        extracted = []

        # Handle psychological events
        if message.header.msg_type in (
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
        ):
            # Extract psych events
            for event in message.events:
                event_data = {
                    "type": "psych_event",
                    "message_type": message.header.msg_type,
                    "session_id": event.session_id,
                    "timestamp_ns": message.header.timestamp_ns,
                    "timestamp_offset_us": event.timestamp_offset_us,
                    "event_type": event.event_type,
                    "window_size": event.window_size,
                    "context_hash": event.context_hash,
                    "scores": {
                        "acute_stress": event.acute_stress,
                        "machiavellianism": event.machiavellianism,
                        "narcissism": event.narcissism,
                        "psychopathy": event.psychopathy,
                        "burnout_probability": event.burnout_probability,
                        "espionage_exposure": event.espionage_exposure,
                        "confidence": event.confidence,
                    },
                    "dark_triad_average": event.dark_triad_average,
                }
                extracted.append(event_data)

        # Handle improvement messages
        elif message.header.msg_type == MessageType.IMPROVEMENT_ANNOUNCE:
            # Parse improvement announcement
            try:
                from improvement_types import ImprovementAnnouncement
                announcement = ImprovementAnnouncement.unpack(message.raw_payload)
                extracted.append({
                    "type": "improvement_announcement",
                    "improvement_id": announcement.improvement_id,
                    "improvement_type": announcement.improvement_type,
                    "priority": announcement.priority,
                    "source_node_id": announcement.source_node_id,
                    "improvement_percentage": announcement.improvement_percentage,
                    "size_bytes": announcement.size_bytes,
                })
            except Exception as e:
                logger.debug(f"Could not parse improvement announcement: {e}")

        elif message.header.msg_type == MessageType.IMPROVEMENT_PAYLOAD:
            # Parse improvement package
            try:
                from improvement_types import ImprovementPackage
                package = ImprovementPackage.unpack(message.raw_payload)
                extracted.append({
                    "type": "improvement_package",
                    "improvement_id": package.improvement_id,
                    "improvement_type": package.improvement_type,
                    "priority": package.priority,
                    "source_node_id": package.source_node_id,
                    "improvement_percentage": package.improvement_percentage,
                    "baseline_metrics": {
                        "accuracy": package.baseline_metrics.accuracy,
                        "latency_ms": package.baseline_metrics.latency_ms,
                    },
                    "improved_metrics": {
                        "accuracy": package.improved_metrics.accuracy,
                        "latency_ms": package.improved_metrics.latency_ms,
                    },
                })
            except Exception as e:
                logger.debug(f"Could not parse improvement package: {e}")

        # Threat/Intel reports (payload is JSON)
        elif message.header.msg_type in (
            MessageType.THREAT_REPORT,
            MessageType.INTEL_REPORT,
            MessageType.BRAIN_INTEL_REPORT,
            MessageType.INTEL_PROPAGATE,
        ):
            try:
                import json
                payload = json.loads(message.raw_payload.decode())
                if isinstance(payload, dict) and payload.get("records"):
                    for rec in payload.get("records", []):
                        extracted.append({
                            "type": "intel_report",
                            "source": payload.get("source"),
                            "account": payload.get("account"),
                            "record": rec,
                            "count": payload.get("count"),
                        })
                else:
                    extracted.append({
                        "type": "intel_report",
                        "payload": payload,
                    })
            except Exception as e:
                logger.debug(f"Could not parse intel payload: {e}")

        # Handle other message types
        elif message.header.msg_type in (MessageType.HEARTBEAT, MessageType.ACK, MessageType.NACK):
            extracted.append({
                "type": "control_message",
                "message_type": message.header.msg_type,
                "timestamp_ns": message.header.timestamp_ns,
                "sequence_num": message.header.sequence_num,
            })

        # MRAC Remote Control (0x2101-0x21FF)
        elif 0x2100 <= message.header.msg_type <= 0x21FF:
            try:
                mrac = self._parse_mrac(message)
                if mrac:
                    extracted.append(mrac)
            except Exception as e:
                logger.debug(f"MRAC parse failure: {e}")

        return extracted if extracted else None

    def _parse_mrac(self, message: MemshadowMessage) -> Optional[Dict[str, Any]]:
        payload = message.raw_payload
        if len(payload) < 24:
            return None
        nonce = payload[16:24]
        offset = 24

        def read_bytes(n: int) -> bytes:
            nonlocal offset
            if offset + n > len(payload):
                raise ValueError("payload too short")
            chunk = payload[offset:offset + n]
            offset += n
            return chunk

        msg_type = message.header.msg_type
        if msg_type == 0x2101:  # APP_REGISTER
            app_id = read_bytes(16).hex()
            cap_len = int.from_bytes(read_bytes(2), "big")
            name_len = int.from_bytes(read_bytes(1), "big")
            name = read_bytes(name_len).decode(errors="ignore") if name_len else ""
            capabilities = {}
            if cap_len:
                cap_bytes = read_bytes(cap_len)
                try:
                    import json
                    capabilities = json.loads(cap_bytes.decode())
                except Exception:
                    capabilities = {"raw": cap_bytes.hex()}
            update_register(app_id, name, capabilities)
            return {"type": "mrac_register", "app_id": app_id, "name": name, "capabilities": capabilities}

        if msg_type == 0x2106:  # APP_HEARTBEAT
            app_id = read_bytes(16).hex()
            uptime_ms = int.from_bytes(read_bytes(8), "big")
            load_pct = int.from_bytes(read_bytes(1), "big")
            temp_c = int.from_bytes(read_bytes(1), "big")
            telemetry = {"uptime_ms": uptime_ms, "load_pct": load_pct, "temp_c": temp_c}
            update_heartbeat(app_id, telemetry)
            return {"type": "mrac_heartbeat", "app_id": app_id, "telemetry": telemetry}

        if msg_type == 0x2104:  # APP_COMMAND_ACK
            app_id = read_bytes(16).hex()
            cmd_id = int.from_bytes(read_bytes(8), "big")
            status = int.from_bytes(read_bytes(1), "big")
            update_command_ack(app_id, str(cmd_id), str(status))
            return {"type": "mrac_command_ack", "app_id": app_id, "command_id": cmd_id, "status": status}

        # Generic MRAC fallback
        return {
            "type": "mrac_raw",
            "msg_type": msg_type,
            "nonce": nonce.hex(),
            "payload_hex": payload.hex(),
        }
