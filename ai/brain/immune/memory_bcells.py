#!/usr/bin/env python3
"""
Memory B-Cells for DSMIL Brain Digital Immune System

Rapid re-recognition of past threats:
- Long-term threat memory
- Quick response activation
- Pattern matching against known threats
"""

import hashlib
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MemoryBCell:
    """A memory cell for a recognized threat"""
    cell_id: str
    threat_hash: str
    threat_name: str

    # Recognition
    signature_pattern: str = ""
    recognition_confidence: float = 0.9

    # Response
    recommended_response: str = ""
    response_priority: int = 5

    # History
    first_encounter: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_encounter: Optional[datetime] = None
    encounter_count: int = 1

    # Affinity maturation
    affinity_score: float = 0.5

    def recognize(self, data: str) -> bool:
        """Check if data matches this memory"""
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        return data_hash == self.threat_hash or self.signature_pattern in data

    def encounter(self):
        """Record a new encounter"""
        self.last_encounter = datetime.now(timezone.utc)
        self.encounter_count += 1
        self.affinity_score = min(1.0, self.affinity_score + 0.1)


@dataclass
class RapidResponse:
    """Rapid response to a recognized threat"""
    response_id: str
    cell_id: str
    threat_name: str

    response_action: str
    priority: int
    latency_ms: float

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ThreatMemoryBank:
    """
    Threat Memory Bank

    Long-term storage of recognized threats for rapid response.

    Usage:
        bank = ThreatMemoryBank()

        # Store threat memory
        bank.store_memory(threat_hash, threat_name, response)

        # Check for recognition
        response = bank.recognize(suspicious_data)
    """

    def __init__(self, retention_days: int = 365):
        self._cells: Dict[str, MemoryBCell] = {}
        self._hash_index: Dict[str, str] = {}
        self._responses: List[RapidResponse] = []
        self._retention = timedelta(days=retention_days)
        self._lock = threading.RLock()

        logger.info("ThreatMemoryBank initialized")

    def store_memory(self, threat_hash: str, threat_name: str,
                    signature: str = "", response: str = "",
                    priority: int = 5) -> MemoryBCell:
        """Store a new threat memory"""
        cell_id = hashlib.sha256(f"{threat_hash}:{threat_name}".encode()).hexdigest()[:16]

        cell = MemoryBCell(
            cell_id=cell_id,
            threat_hash=threat_hash,
            threat_name=threat_name,
            signature_pattern=signature,
            recommended_response=response,
            response_priority=priority,
        )

        with self._lock:
            self._cells[cell_id] = cell
            self._hash_index[threat_hash] = cell_id

        return cell

    def recognize(self, data: str) -> Optional[RapidResponse]:
        """
        Check if data matches any known threats

        Returns RapidResponse if recognized
        """
        import time
        start = time.time()

        data_hash = hashlib.sha256(data.encode()).hexdigest()

        with self._lock:
            # Fast hash lookup
            if data_hash in self._hash_index:
                cell_id = self._hash_index[data_hash]
                cell = self._cells[cell_id]
                cell.encounter()

                response = RapidResponse(
                    response_id=hashlib.sha256(f"{cell_id}:{time.time()}".encode()).hexdigest()[:16],
                    cell_id=cell_id,
                    threat_name=cell.threat_name,
                    response_action=cell.recommended_response,
                    priority=cell.response_priority,
                    latency_ms=(time.time() - start) * 1000,
                )
                self._responses.append(response)
                return response

            # Pattern matching
            for cell in self._cells.values():
                if cell.signature_pattern and cell.signature_pattern in data:
                    cell.encounter()

                    response = RapidResponse(
                        response_id=hashlib.sha256(f"{cell.cell_id}:{time.time()}".encode()).hexdigest()[:16],
                        cell_id=cell.cell_id,
                        threat_name=cell.threat_name,
                        response_action=cell.recommended_response,
                        priority=cell.response_priority,
                        latency_ms=(time.time() - start) * 1000,
                    )
                    self._responses.append(response)
                    return response

        return None

    def get_stats(self) -> Dict:
        """Get memory bank statistics"""
        with self._lock:
            return {
                "total_memories": len(self._cells),
                "total_responses": len(self._responses),
                "avg_affinity": sum(c.affinity_score for c in self._cells.values()) / max(1, len(self._cells)),
            }


if __name__ == "__main__":
    print("Memory B-Cells Self-Test")
    print("=" * 50)

    bank = ThreatMemoryBank()

    print("\n[1] Store Threat Memories")
    bank.store_memory(
        threat_hash="abc123",
        threat_name="Malware.TestTrojan",
        signature="eval(base64_decode",
        response="quarantine",
        priority=8
    )
    bank.store_memory(
        threat_hash="def456",
        threat_name="Exploit.SQLInjection",
        signature="1=1 OR",
        response="block_request",
        priority=9
    )
    print(f"    Stored 2 threat memories")

    print("\n[2] Recognition Test")
    test_data = "This contains eval(base64_decode('malicious'))"
    response = bank.recognize(test_data)
    if response:
        print(f"    Recognized: {response.threat_name}")
        print(f"    Response: {response.response_action}")
        print(f"    Latency: {response.latency_ms:.3f}ms")

    print("\n[3] Statistics")
    stats = bank.get_stats()
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("Memory B-Cells test complete")

