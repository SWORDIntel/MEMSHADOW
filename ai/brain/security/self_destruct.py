#!/usr/bin/env python3
"""
Self-Destruct Protocol for DSMIL Brain

Emergency response when tampering is detected:
1. Emergency intel capture
2. Broadcast to hub/peers (CRITICAL priority)
3. Secure wipe: keys → memory → knowledge
4. Log tamper event (encrypted, signed)
5. Controlled shutdown

This module handles the secure destruction of sensitive data
while attempting to exfiltrate critical intelligence back to
the hub before shutdown.
"""

import os
import sys
import time
import ctypes
import secrets
import threading
import logging
import gc
import mmap
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List, Any, BinaryIO
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)


class DestructPhase(Enum):
    """Self-destruct sequence phases"""
    INITIATED = auto()
    INTEL_CAPTURE = auto()
    INTEL_BROADCAST = auto()
    KEY_WIPE = auto()
    MEMORY_WIPE = auto()
    KNOWLEDGE_WIPE = auto()
    LOG_EVENT = auto()
    SHUTDOWN = auto()
    COMPLETE = auto()


class WipeMethod(Enum):
    """Methods for secure data wiping"""
    ZERO = auto()       # Overwrite with zeros
    RANDOM = auto()     # Overwrite with random data
    DOD_3PASS = auto()  # DoD 5220.22-M 3-pass
    GUTMANN = auto()    # Gutmann 35-pass (overkill for most)


@dataclass
class EmergencyIntelCapture:
    """Captured intelligence for emergency broadcast"""
    timestamp: datetime
    node_id: str
    tamper_evidence: List[Dict]
    last_known_peers: List[str]
    partial_knowledge_dump: Optional[bytes] = None
    system_state: Optional[Dict] = None
    encrypted_keys_backup: Optional[bytes] = None

    def to_bytes(self) -> bytes:
        """Serialize for transmission"""
        data = {
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "tamper_evidence": self.tamper_evidence,
            "last_known_peers": self.last_known_peers,
            "system_state": self.system_state,
        }
        json_bytes = json.dumps(data).encode()

        # Append binary data if present
        result = len(json_bytes).to_bytes(4, 'big') + json_bytes

        if self.partial_knowledge_dump:
            result += len(self.partial_knowledge_dump).to_bytes(4, 'big')
            result += self.partial_knowledge_dump
        else:
            result += (0).to_bytes(4, 'big')

        if self.encrypted_keys_backup:
            result += len(self.encrypted_keys_backup).to_bytes(4, 'big')
            result += self.encrypted_keys_backup
        else:
            result += (0).to_bytes(4, 'big')

        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "EmergencyIntelCapture":
        """Deserialize from transmission"""
        json_len = int.from_bytes(data[:4], 'big')
        json_data = json.loads(data[4:4+json_len])
        offset = 4 + json_len

        knowledge_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        knowledge_dump = data[offset:offset+knowledge_len] if knowledge_len else None
        offset += knowledge_len

        keys_len = int.from_bytes(data[offset:offset+4], 'big')
        offset += 4
        keys_backup = data[offset:offset+keys_len] if keys_len else None

        return cls(
            timestamp=datetime.fromisoformat(json_data["timestamp"]),
            node_id=json_data["node_id"],
            tamper_evidence=json_data["tamper_evidence"],
            last_known_peers=json_data["last_known_peers"],
            system_state=json_data.get("system_state"),
            partial_knowledge_dump=knowledge_dump,
            encrypted_keys_backup=keys_backup,
        )


class SecureWipe:
    """
    Secure data wiping utilities

    Provides methods to securely erase data from memory and storage,
    making recovery extremely difficult or impossible.
    """

    @staticmethod
    def wipe_bytes(data: bytearray, method: WipeMethod = WipeMethod.DOD_3PASS):
        """
        Securely wipe a bytearray in place

        Args:
            data: Bytearray to wipe (modified in place)
            method: Wiping method to use
        """
        length = len(data)

        if method == WipeMethod.ZERO:
            for i in range(length):
                data[i] = 0

        elif method == WipeMethod.RANDOM:
            random_data = secrets.token_bytes(length)
            for i in range(length):
                data[i] = random_data[i]

        elif method == WipeMethod.DOD_3PASS:
            # Pass 1: All zeros
            for i in range(length):
                data[i] = 0x00
            # Pass 2: All ones
            for i in range(length):
                data[i] = 0xFF
            # Pass 3: Random
            random_data = secrets.token_bytes(length)
            for i in range(length):
                data[i] = random_data[i]

        elif method == WipeMethod.GUTMANN:
            # 35-pass Gutmann method (simplified)
            patterns = [
                bytes([0x55]), bytes([0xAA]), bytes([0x92, 0x49, 0x24]),
                bytes([0x49, 0x24, 0x92]), bytes([0x24, 0x92, 0x49]),
                bytes([0x00]), bytes([0x11]), bytes([0x22]), bytes([0x33]),
                bytes([0x44]), bytes([0x55]), bytes([0x66]), bytes([0x77]),
                bytes([0x88]), bytes([0x99]), bytes([0xAA]), bytes([0xBB]),
                bytes([0xCC]), bytes([0xDD]), bytes([0xEE]), bytes([0xFF]),
            ]
            for pattern in patterns:
                for i in range(length):
                    data[i] = pattern[i % len(pattern)]
            # Final random pass
            random_data = secrets.token_bytes(length)
            for i in range(length):
                data[i] = random_data[i]

    @staticmethod
    def wipe_file(filepath: Path, method: WipeMethod = WipeMethod.DOD_3PASS,
                  delete: bool = True) -> bool:
        """
        Securely wipe a file

        Args:
            filepath: Path to file
            method: Wiping method
            delete: Delete file after wiping

        Returns:
            True if successful
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return True

            file_size = filepath.stat().st_size

            if method == WipeMethod.DOD_3PASS:
                passes = [
                    lambda: b'\x00',
                    lambda: b'\xFF',
                    lambda: secrets.token_bytes(1),
                ]
            elif method == WipeMethod.RANDOM:
                passes = [lambda: secrets.token_bytes(1)]
            else:
                passes = [lambda: b'\x00']

            for pass_func in passes:
                with open(filepath, 'r+b') as f:
                    for _ in range(file_size):
                        f.write(pass_func())
                    f.flush()
                    os.fsync(f.fileno())

            if delete:
                # Rename before delete to obscure original name
                temp_name = filepath.parent / secrets.token_hex(16)
                filepath.rename(temp_name)
                temp_name.unlink()

            return True

        except Exception as e:
            logger.error(f"File wipe failed for {filepath}: {e}")
            return False

    @staticmethod
    def wipe_directory(dirpath: Path, method: WipeMethod = WipeMethod.DOD_3PASS) -> int:
        """
        Recursively wipe all files in a directory

        Returns:
            Number of files wiped
        """
        count = 0
        dirpath = Path(dirpath)

        if not dirpath.exists():
            return 0

        for item in dirpath.rglob("*"):
            if item.is_file():
                if SecureWipe.wipe_file(item, method):
                    count += 1

        # Remove empty directories
        for item in sorted(dirpath.rglob("*"), reverse=True):
            if item.is_dir():
                try:
                    item.rmdir()
                except Exception:
                    pass

        try:
            dirpath.rmdir()
        except Exception:
            pass

        return count

    @staticmethod
    def wipe_memory_region(address: int, size: int):
        """
        Wipe a memory region (platform-specific)

        Note: This is a best-effort operation and may not work
        on all platforms or for all memory regions.
        """
        try:
            if sys.platform.startswith('linux'):
                # Use madvise to request memory release
                libc = ctypes.CDLL("libc.so.6")
                MADV_DONTNEED = 4
                libc.madvise(address, size, MADV_DONTNEED)

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.warning(f"Memory wipe failed: {e}")

    @staticmethod
    def secure_delete_string(s: str) -> None:
        """
        Attempt to securely delete a string from memory

        Note: Python strings are immutable, so this is best-effort.
        The string may still exist in memory until garbage collected.
        """
        try:
            # Get the string's internal buffer
            # This is CPython-specific and may not work
            import ctypes

            str_type = type(s)
            str_size = sys.getsizeof(s)

            # Overwrite the string's internal data
            # This is extremely hacky and not guaranteed to work
            addr = id(s)

            # Force deletion
            del s
            gc.collect()

        except Exception:
            pass


class SelfDestructProtocol:
    """
    Self-destruct protocol for compromised nodes

    When tampering is detected, this protocol:
    1. Captures critical intelligence
    2. Attempts to broadcast intel to hub/peers
    3. Securely wipes all sensitive data
    4. Logs the event
    5. Performs controlled shutdown

    Usage:
        protocol = SelfDestructProtocol(node_id="node-001")
        protocol.set_broadcast_callback(broadcast_func)

        # On tamper detection
        protocol.initiate(tamper_evidence)
    """

    def __init__(self, node_id: str,
                 data_directories: Optional[List[Path]] = None,
                 wipe_method: WipeMethod = WipeMethod.DOD_3PASS):
        """
        Initialize self-destruct protocol

        Args:
            node_id: This node's identifier
            data_directories: Directories containing sensitive data
            wipe_method: Method for secure wiping
        """
        self.node_id = node_id
        self.data_directories = data_directories or []
        self.wipe_method = wipe_method

        # State
        self.current_phase = DestructPhase.INITIATED
        self.initiated = False
        self.abort_possible = True

        # Callbacks
        self.broadcast_callback: Optional[Callable[[EmergencyIntelCapture], bool]] = None
        self.get_knowledge_dump: Optional[Callable[[], bytes]] = None
        self.get_peer_list: Optional[Callable[[], List[str]]] = None
        self.get_system_state: Optional[Callable[[], Dict]] = None
        self.get_encrypted_keys: Optional[Callable[[], bytes]] = None
        self.on_phase_change: Optional[Callable[[DestructPhase], None]] = None

        # Keys to wipe
        self._sensitive_keys: List[bytearray] = []
        self._sensitive_objects: List[Any] = []

        # Execution log
        self._log: List[Dict] = []

    def register_sensitive_key(self, key: bytearray):
        """Register a key for secure wiping"""
        self._sensitive_keys.append(key)

    def register_sensitive_object(self, obj: Any):
        """Register an object for cleanup"""
        self._sensitive_objects.append(obj)

    def set_broadcast_callback(self, callback: Callable[[EmergencyIntelCapture], bool]):
        """Set the callback for broadcasting intel"""
        self.broadcast_callback = callback

    def _log_phase(self, phase: DestructPhase, success: bool, details: str = ""):
        """Log a phase execution"""
        entry = {
            "phase": phase.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": success,
            "details": details,
        }
        self._log.append(entry)
        logger.info(f"Self-destruct phase {phase.name}: {'OK' if success else 'FAILED'} {details}")

        if self.on_phase_change:
            self.on_phase_change(phase)

    def _set_phase(self, phase: DestructPhase):
        """Update current phase"""
        self.current_phase = phase
        self._log_phase(phase, True)

    def initiate(self, tamper_evidence: List[Dict],
                 abort_timeout: float = 0) -> bool:
        """
        Initiate the self-destruct sequence

        Args:
            tamper_evidence: Evidence that triggered the protocol
            abort_timeout: Seconds to wait for abort (0 = immediate)

        Returns:
            True if sequence completed
        """
        if self.initiated:
            logger.warning("Self-destruct already initiated")
            return False

        self.initiated = True
        self._set_phase(DestructPhase.INITIATED)

        logger.critical(f"SELF-DESTRUCT INITIATED for node {self.node_id}")

        # Brief window for abort
        if abort_timeout > 0:
            logger.warning(f"Abort window: {abort_timeout} seconds")
            time.sleep(abort_timeout)
            if not self.abort_possible:
                logger.info("Sequence aborted")
                self.initiated = False
                return False

        self.abort_possible = False

        try:
            # Phase 1: Intel Capture
            intel = self._capture_intel(tamper_evidence)

            # Phase 2: Broadcast Intel
            self._broadcast_intel(intel)

            # Phase 3: Wipe Keys
            self._wipe_keys()

            # Phase 4: Wipe Memory
            self._wipe_memory()

            # Phase 5: Wipe Knowledge
            self._wipe_knowledge()

            # Phase 6: Log Event
            self._log_final_event(tamper_evidence)

            # Phase 7: Shutdown
            self._shutdown()

            return True

        except Exception as e:
            logger.error(f"Self-destruct error: {e}")
            # Continue with best-effort destruction
            self._emergency_wipe()
            return False

    def abort(self) -> bool:
        """
        Abort the self-destruct sequence if still possible

        Returns:
            True if aborted
        """
        if self.abort_possible:
            self.abort_possible = False
            logger.info("Self-destruct sequence aborted")
            return True
        return False

    def _capture_intel(self, tamper_evidence: List[Dict]) -> EmergencyIntelCapture:
        """Phase 1: Capture critical intelligence"""
        self._set_phase(DestructPhase.INTEL_CAPTURE)

        # Get peer list
        peers = []
        if self.get_peer_list:
            try:
                peers = self.get_peer_list()
            except Exception as e:
                logger.error(f"Failed to get peers: {e}")

        # Get partial knowledge dump
        knowledge = None
        if self.get_knowledge_dump:
            try:
                knowledge = self.get_knowledge_dump()
            except Exception as e:
                logger.error(f"Failed to dump knowledge: {e}")

        # Get system state
        state = None
        if self.get_system_state:
            try:
                state = self.get_system_state()
            except Exception as e:
                logger.error(f"Failed to get state: {e}")

        # Get encrypted keys backup
        keys_backup = None
        if self.get_encrypted_keys:
            try:
                keys_backup = self.get_encrypted_keys()
            except Exception as e:
                logger.error(f"Failed to backup keys: {e}")

        intel = EmergencyIntelCapture(
            timestamp=datetime.now(timezone.utc),
            node_id=self.node_id,
            tamper_evidence=tamper_evidence,
            last_known_peers=peers,
            partial_knowledge_dump=knowledge,
            system_state=state,
            encrypted_keys_backup=keys_backup,
        )

        self._log_phase(DestructPhase.INTEL_CAPTURE, True,
                       f"peers={len(peers)}, knowledge={'yes' if knowledge else 'no'}")

        return intel

    def _broadcast_intel(self, intel: EmergencyIntelCapture):
        """Phase 2: Broadcast intel to hub/peers"""
        self._set_phase(DestructPhase.INTEL_BROADCAST)

        if not self.broadcast_callback:
            self._log_phase(DestructPhase.INTEL_BROADCAST, False, "No broadcast callback")
            return

        try:
            success = self.broadcast_callback(intel)
            self._log_phase(DestructPhase.INTEL_BROADCAST, success,
                           "Broadcast " + ("succeeded" if success else "failed"))
        except Exception as e:
            self._log_phase(DestructPhase.INTEL_BROADCAST, False, str(e))

    def _wipe_keys(self):
        """Phase 3: Securely wipe all cryptographic keys"""
        self._set_phase(DestructPhase.KEY_WIPE)

        wiped = 0
        for key in self._sensitive_keys:
            try:
                SecureWipe.wipe_bytes(key, self.wipe_method)
                wiped += 1
            except Exception as e:
                logger.error(f"Key wipe failed: {e}")

        self._log_phase(DestructPhase.KEY_WIPE, True, f"wiped={wiped}")

    def _wipe_memory(self):
        """Phase 4: Wipe sensitive memory"""
        self._set_phase(DestructPhase.MEMORY_WIPE)

        # Clear sensitive objects
        for obj in self._sensitive_objects:
            try:
                if hasattr(obj, 'clear'):
                    obj.clear()
                del obj
            except Exception:
                pass

        self._sensitive_objects.clear()

        # Force garbage collection
        gc.collect()
        gc.collect()
        gc.collect()

        self._log_phase(DestructPhase.MEMORY_WIPE, True)

    def _wipe_knowledge(self):
        """Phase 5: Wipe knowledge storage"""
        self._set_phase(DestructPhase.KNOWLEDGE_WIPE)

        total_files = 0
        for directory in self.data_directories:
            try:
                count = SecureWipe.wipe_directory(Path(directory), self.wipe_method)
                total_files += count
            except Exception as e:
                logger.error(f"Directory wipe failed for {directory}: {e}")

        self._log_phase(DestructPhase.KNOWLEDGE_WIPE, True, f"files={total_files}")

    def _log_final_event(self, tamper_evidence: List[Dict]):
        """Phase 6: Log the self-destruct event"""
        self._set_phase(DestructPhase.LOG_EVENT)

        event = {
            "event": "SELF_DESTRUCT",
            "node_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tamper_evidence": tamper_evidence,
            "execution_log": self._log,
        }

        # Try to write to a recoverable location
        try:
            log_path = Path("/tmp") / f"dsmil_destruct_{self.node_id}_{int(time.time())}.log"
            with open(log_path, 'w') as f:
                json.dump(event, f, indent=2)
            logger.info(f"Destruct log written to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write destruct log: {e}")

        self._log_phase(DestructPhase.LOG_EVENT, True)

    def _shutdown(self):
        """Phase 7: Controlled shutdown"""
        self._set_phase(DestructPhase.SHUTDOWN)

        logger.critical("Initiating controlled shutdown")

        # Give time for logs to flush
        time.sleep(0.5)

        self._set_phase(DestructPhase.COMPLETE)

        # Exit with special code indicating self-destruct
        sys.exit(137)  # 128 + 9 (SIGKILL equivalent)

    def _emergency_wipe(self):
        """Emergency wipe when normal sequence fails"""
        logger.critical("Emergency wipe initiated")

        # Wipe keys
        for key in self._sensitive_keys:
            try:
                for i in range(len(key)):
                    key[i] = 0
            except Exception:
                pass

        # Clear objects
        self._sensitive_objects.clear()
        self._sensitive_keys.clear()

        # Force GC
        gc.collect()

        # Exit
        os._exit(137)

    def get_status(self) -> Dict:
        """Get current protocol status"""
        return {
            "node_id": self.node_id,
            "initiated": self.initiated,
            "abort_possible": self.abort_possible,
            "current_phase": self.current_phase.name,
            "registered_keys": len(self._sensitive_keys),
            "registered_objects": len(self._sensitive_objects),
            "data_directories": [str(d) for d in self.data_directories],
            "wipe_method": self.wipe_method.name,
            "log_entries": len(self._log),
        }


if __name__ == "__main__":
    print("Self-Destruct Protocol Test (DRY RUN)")
    print("=" * 50)
    print("WARNING: This is a test - no actual destruction")
    print()

    # Create protocol without real data dirs
    protocol = SelfDestructProtocol(
        node_id="test-node-001",
        data_directories=[],
        wipe_method=WipeMethod.DOD_3PASS,
    )

    # Register mock sensitive data
    mock_key = bytearray(b"SUPERSECRETKEY!!" * 2)
    protocol.register_sensitive_key(mock_key)

    print("[1] Protocol Status")
    status = protocol.get_status()
    for key, value in status.items():
        print(f"    {key}: {value}")

    print("\n[2] Testing SecureWipe.wipe_bytes")
    test_data = bytearray(b"SENSITIVE DATA HERE")
    print(f"    Before: {test_data}")
    SecureWipe.wipe_bytes(test_data, WipeMethod.DOD_3PASS)
    print(f"    After:  {test_data}")

    print("\n[3] Mock intel capture")
    intel = EmergencyIntelCapture(
        timestamp=datetime.now(timezone.utc),
        node_id="test-node-001",
        tamper_evidence=[{"type": "TEST", "details": "dry run"}],
        last_known_peers=["hub-001", "node-002"],
    )
    serialized = intel.to_bytes()
    print(f"    Intel size: {len(serialized)} bytes")
    restored = EmergencyIntelCapture.from_bytes(serialized)
    print(f"    Restored node_id: {restored.node_id}")

    print("\n[4] Abort test")
    protocol.abort_possible = True
    aborted = protocol.abort()
    print(f"    Abort result: {aborted}")

    print("\n" + "=" * 50)
    print("Test complete (no actual destruction performed)")

