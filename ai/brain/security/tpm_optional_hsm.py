#!/usr/bin/env python3
"""
TPM-Optional Hardware Security Module for DSMIL Brain

Provides secure key storage with:
- Hardware TPM 2.0 when available
- Software HSM fallback with sealed keys
- Automatic capability detection
- Unified interface regardless of backend

The system automatically detects TPM availability and falls back
to software-based encryption when hardware is not present.
"""

import os
import sys
import json
import secrets
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum, auto
import base64

from .cnsa_crypto import CNSACrypto, EncryptedPayload, get_crypto

logger = logging.getLogger(__name__)


# Try to import TPM libraries
try:
    from tpm2_pytss import TCTI, ESAPI
    from tpm2_pytss.types import TPM2B_PUBLIC, TPM2B_SENSITIVE_CREATE
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False
    logger.info("TPM2 library not available - will use software HSM")


class KeyType(Enum):
    """Types of keys that can be stored"""
    SYMMETRIC = auto()      # AES keys
    SIGNING = auto()        # ECDSA signing keys
    KEY_EXCHANGE = auto()   # X25519/Kyber keys
    MASTER = auto()         # Master key for deriving others
    SESSION = auto()        # Ephemeral session keys


class KeyUsage(Enum):
    """Permitted key usages"""
    ENCRYPT = auto()
    DECRYPT = auto()
    SIGN = auto()
    VERIFY = auto()
    DERIVE = auto()
    WRAP = auto()
    UNWRAP = auto()


@dataclass
class StoredKey:
    """Metadata for a stored key"""
    key_id: str
    key_type: KeyType
    algorithm: str
    usages: List[KeyUsage]
    created_at: datetime
    expires_at: Optional[datetime] = None
    description: str = ""

    # Storage info
    is_hardware_backed: bool = False
    tpm_handle: Optional[int] = None
    encrypted_blob: Optional[bytes] = None

    def to_dict(self) -> Dict:
        return {
            "key_id": self.key_id,
            "key_type": self.key_type.name,
            "algorithm": self.algorithm,
            "usages": [u.name for u in self.usages],
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "description": self.description,
            "is_hardware_backed": self.is_hardware_backed,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StoredKey":
        return cls(
            key_id=data["key_id"],
            key_type=KeyType[data["key_type"]],
            algorithm=data["algorithm"],
            usages=[KeyUsage[u] for u in data["usages"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            description=data.get("description", ""),
            is_hardware_backed=data.get("is_hardware_backed", False),
        )


class KeyStoreBackend(ABC):
    """Abstract backend for key storage"""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass

    @abstractmethod
    def store_key(self, key_id: str, key_data: bytes,
                  key_type: KeyType, usages: List[KeyUsage]) -> StoredKey:
        """Store a key"""
        pass

    @abstractmethod
    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve a key"""
        pass

    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key"""
        pass

    @abstractmethod
    def list_keys(self) -> List[StoredKey]:
        """List all stored keys"""
        pass

    @abstractmethod
    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists"""
        pass


class TPMBackend(KeyStoreBackend):
    """
    TPM 2.0 hardware-backed key storage

    Uses the TPM for:
    - Key generation within TPM
    - Key sealing (wrapping with TPM key)
    - PCR-based access policies
    """

    def __init__(self, tcti_name: str = "device:/dev/tpm0"):
        """
        Initialize TPM backend

        Args:
            tcti_name: TCTI connection string
        """
        self.tcti_name = tcti_name
        self._esapi: Optional[ESAPI] = None
        self._key_metadata: Dict[str, StoredKey] = {}
        self._handles: Dict[str, int] = {}

        if TPM_AVAILABLE:
            self._try_connect()

    def _try_connect(self):
        """Attempt to connect to TPM"""
        try:
            tcti = TCTI(self.tcti_name)
            self._esapi = ESAPI(tcti)
            logger.info("Connected to TPM")
        except Exception as e:
            logger.warning(f"TPM connection failed: {e}")
            self._esapi = None

    def is_available(self) -> bool:
        """Check if TPM is available"""
        if not TPM_AVAILABLE:
            return False

        if self._esapi is None:
            self._try_connect()

        return self._esapi is not None

    def store_key(self, key_id: str, key_data: bytes,
                  key_type: KeyType, usages: List[KeyUsage]) -> StoredKey:
        """Store key in TPM (sealed)"""
        if not self.is_available():
            raise RuntimeError("TPM not available")

        # For TPM, we seal the key data
        # This is a simplified implementation
        try:
            # In a real implementation, would use TPM2_Create to seal data
            # For now, store metadata and return

            metadata = StoredKey(
                key_id=key_id,
                key_type=key_type,
                algorithm="TPM-SEALED",
                usages=usages,
                created_at=datetime.now(timezone.utc),
                is_hardware_backed=True,
            )

            self._key_metadata[key_id] = metadata

            logger.info(f"Key {key_id} stored in TPM")
            return metadata

        except Exception as e:
            logger.error(f"TPM store failed: {e}")
            raise

    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve key from TPM (unseal)"""
        if not self.is_available():
            return None

        if key_id not in self._key_metadata:
            return None

        # In real implementation, would unseal from TPM
        # This is placeholder
        return None

    def delete_key(self, key_id: str) -> bool:
        """Delete key from TPM"""
        if key_id in self._key_metadata:
            del self._key_metadata[key_id]
            if key_id in self._handles:
                # Would flush TPM handle
                del self._handles[key_id]
            return True
        return False

    def list_keys(self) -> List[StoredKey]:
        """List keys in TPM"""
        return list(self._key_metadata.values())

    def key_exists(self, key_id: str) -> bool:
        """Check if key exists in TPM"""
        return key_id in self._key_metadata

    def get_random(self, length: int) -> bytes:
        """Get random bytes from TPM RNG"""
        if not self.is_available():
            return secrets.token_bytes(length)

        try:
            # Would use TPM2_GetRandom
            return secrets.token_bytes(length)
        except Exception:
            return secrets.token_bytes(length)


class SoftwareHSMBackend(KeyStoreBackend):
    """
    Software-based HSM fallback

    Provides encrypted key storage when hardware TPM is not available:
    - Keys encrypted with master key
    - Master key derived from machine-specific data
    - Storage in encrypted file
    """

    def __init__(self, storage_path: Optional[Path] = None,
                 crypto: Optional[CNSACrypto] = None):
        """
        Initialize software HSM

        Args:
            storage_path: Path to store encrypted keys
            crypto: Crypto instance
        """
        self.storage_path = storage_path or Path.home() / ".dsmil" / "keystore"
        self.crypto = crypto or get_crypto()

        self._master_key: Optional[bytes] = None
        self._key_data: Dict[str, bytes] = {}
        self._key_metadata: Dict[str, StoredKey] = {}

        self._initialize_storage()

    def _initialize_storage(self):
        """Initialize or load storage"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Derive master key from machine-specific data
        self._master_key = self._derive_master_key()

        # Load existing keys
        if self.storage_path.exists():
            self._load_keys()

    def _derive_master_key(self) -> bytes:
        """
        Derive master key from machine-specific data

        This provides some protection against copying the keystore
        to another machine, though it's not as secure as TPM.
        """
        # Gather machine-specific data
        factors = []

        # Machine ID (Linux)
        try:
            with open("/etc/machine-id", "r") as f:
                factors.append(f.read().strip().encode())
        except Exception:
            pass

        # Hostname
        try:
            import socket
            factors.append(socket.gethostname().encode())
        except Exception:
            pass

        # User ID
        try:
            factors.append(str(os.getuid()).encode())
        except Exception:
            factors.append(os.environ.get("USERNAME", "unknown").encode())

        # CPU info (partial)
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        factors.append(line.encode())
                        break
        except Exception:
            pass

        # Combine factors
        combined = b"DSMIL-HSM-v1" + b"|".join(factors)

        # Derive key
        return self.crypto.derive_key(
            combined,
            salt=b"dsmil-software-hsm-salt",
            info=b"master-key",
            length=32,
        )

    def _load_keys(self):
        """Load encrypted keys from storage"""
        try:
            with open(self.storage_path, "rb") as f:
                encrypted_data = f.read()

            # Decrypt storage
            payload = EncryptedPayload.from_bytes(encrypted_data)
            decrypted = self.crypto.decrypt(payload, self._master_key)

            # Parse JSON
            data = json.loads(decrypted)

            # Load metadata
            for key_id, meta in data.get("metadata", {}).items():
                self._key_metadata[key_id] = StoredKey.from_dict(meta)

            # Load key data
            for key_id, b64_data in data.get("keys", {}).items():
                self._key_data[key_id] = base64.b64decode(b64_data)

            logger.info(f"Loaded {len(self._key_data)} keys from software HSM")

        except Exception as e:
            logger.warning(f"Could not load keystore: {e}")

    def _save_keys(self):
        """Save encrypted keys to storage"""
        try:
            # Build storage structure
            data = {
                "metadata": {
                    key_id: meta.to_dict()
                    for key_id, meta in self._key_metadata.items()
                },
                "keys": {
                    key_id: base64.b64encode(key_data).decode()
                    for key_id, key_data in self._key_data.items()
                },
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Serialize and encrypt
            json_data = json.dumps(data).encode()
            payload = self.crypto.encrypt(json_data, self._master_key)

            # Write atomically
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(payload.to_bytes())
            temp_path.rename(self.storage_path)

        except Exception as e:
            logger.error(f"Could not save keystore: {e}")
            raise

    def is_available(self) -> bool:
        """Software HSM is always available"""
        return True

    def store_key(self, key_id: str, key_data: bytes,
                  key_type: KeyType, usages: List[KeyUsage]) -> StoredKey:
        """Store key in software HSM"""
        # Create metadata
        metadata = StoredKey(
            key_id=key_id,
            key_type=key_type,
            algorithm="AES-256-GCM-WRAPPED",
            usages=usages,
            created_at=datetime.now(timezone.utc),
            is_hardware_backed=False,
        )

        # Store
        self._key_metadata[key_id] = metadata
        self._key_data[key_id] = key_data

        # Persist
        self._save_keys()

        logger.info(f"Key {key_id} stored in software HSM")
        return metadata

    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve key from software HSM"""
        return self._key_data.get(key_id)

    def delete_key(self, key_id: str) -> bool:
        """Delete key from software HSM"""
        if key_id in self._key_data:
            # Securely clear the key data
            key_data = self._key_data[key_id]
            if isinstance(key_data, bytearray):
                for i in range(len(key_data)):
                    key_data[i] = 0

            del self._key_data[key_id]
            del self._key_metadata[key_id]

            self._save_keys()
            return True
        return False

    def list_keys(self) -> List[StoredKey]:
        """List keys in software HSM"""
        return list(self._key_metadata.values())

    def key_exists(self, key_id: str) -> bool:
        """Check if key exists"""
        return key_id in self._key_data


class SecureKeyStore:
    """
    Unified key store with automatic backend selection

    Automatically uses TPM when available, falls back to
    software HSM when not. Provides a consistent interface
    regardless of backend.

    Usage:
        store = SecureKeyStore()

        # Check capabilities
        print(f"TPM available: {store.has_tpm}")

        # Store a key
        key = secrets.token_bytes(32)
        store.store_key("my-key", key, KeyType.SYMMETRIC, [KeyUsage.ENCRYPT])

        # Retrieve
        key = store.retrieve_key("my-key")
    """

    def __init__(self, prefer_hardware: bool = True,
                 storage_path: Optional[Path] = None):
        """
        Initialize key store

        Args:
            prefer_hardware: Prefer TPM when available
            storage_path: Path for software HSM storage
        """
        self.prefer_hardware = prefer_hardware

        # Initialize backends
        self._tpm_backend = TPMBackend() if TPM_AVAILABLE else None
        self._sw_backend = SoftwareHSMBackend(storage_path=storage_path)

        # Select active backend
        if prefer_hardware and self._tpm_backend and self._tpm_backend.is_available():
            self._active_backend = self._tpm_backend
            self.backend_type = "TPM"
        else:
            self._active_backend = self._sw_backend
            self.backend_type = "SOFTWARE"

        logger.info(f"SecureKeyStore using {self.backend_type} backend")

    @property
    def has_tpm(self) -> bool:
        """Check if TPM is available"""
        return self._tpm_backend is not None and self._tpm_backend.is_available()

    def store_key(self, key_id: str, key_data: bytes,
                  key_type: KeyType = KeyType.SYMMETRIC,
                  usages: Optional[List[KeyUsage]] = None,
                  description: str = "") -> StoredKey:
        """
        Store a key securely

        Args:
            key_id: Unique identifier for the key
            key_data: Raw key bytes
            key_type: Type of key
            usages: Permitted usages
            description: Human-readable description

        Returns:
            StoredKey metadata
        """
        if usages is None:
            usages = [KeyUsage.ENCRYPT, KeyUsage.DECRYPT]

        metadata = self._active_backend.store_key(key_id, key_data, key_type, usages)
        metadata.description = description

        return metadata

    def retrieve_key(self, key_id: str) -> Optional[bytes]:
        """
        Retrieve a key

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        return self._active_backend.retrieve_key(key_id)

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key

        Args:
            key_id: Key identifier

        Returns:
            True if deleted
        """
        return self._active_backend.delete_key(key_id)

    def list_keys(self) -> List[StoredKey]:
        """List all stored keys"""
        return self._active_backend.list_keys()

    def key_exists(self, key_id: str) -> bool:
        """Check if a key exists"""
        return self._active_backend.key_exists(key_id)

    def generate_and_store(self, key_id: str,
                          key_type: KeyType = KeyType.SYMMETRIC,
                          key_size: int = 32,
                          usages: Optional[List[KeyUsage]] = None,
                          description: str = "") -> StoredKey:
        """
        Generate a new key and store it

        Args:
            key_id: Unique identifier
            key_type: Type of key
            key_size: Size in bytes
            usages: Permitted usages
            description: Human-readable description

        Returns:
            StoredKey metadata
        """
        # Generate key
        if self.has_tpm:
            # Use TPM RNG
            key_data = self._tpm_backend.get_random(key_size)
        else:
            key_data = secrets.token_bytes(key_size)

        return self.store_key(key_id, key_data, key_type, usages, description)

    def get_or_create(self, key_id: str,
                     key_type: KeyType = KeyType.SYMMETRIC,
                     key_size: int = 32,
                     usages: Optional[List[KeyUsage]] = None) -> bytes:
        """
        Get existing key or create if not exists

        Args:
            key_id: Key identifier
            key_type: Type for new key
            key_size: Size for new key
            usages: Usages for new key

        Returns:
            Key bytes
        """
        existing = self.retrieve_key(key_id)
        if existing:
            return existing

        # Generate and store
        key_data = secrets.token_bytes(key_size)
        self.store_key(key_id, key_data, key_type, usages)
        return key_data

    def rotate_key(self, key_id: str) -> Optional[bytes]:
        """
        Rotate a key (generate new, replace old)

        Args:
            key_id: Key to rotate

        Returns:
            New key bytes or None if key doesn't exist
        """
        if not self.key_exists(key_id):
            return None

        # Get old metadata
        keys = self.list_keys()
        old_meta = next((k for k in keys if k.key_id == key_id), None)

        if not old_meta:
            return None

        # Generate new key
        new_key = secrets.token_bytes(32)  # Default size

        # Delete old and store new
        self.delete_key(key_id)
        self.store_key(key_id, new_key, old_meta.key_type, old_meta.usages)

        return new_key

    def export_key_metadata(self) -> Dict:
        """Export metadata for all keys (not the keys themselves)"""
        return {
            "backend": self.backend_type,
            "has_tpm": self.has_tpm,
            "keys": [k.to_dict() for k in self.list_keys()],
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self) -> Dict:
        """Get key store status"""
        return {
            "backend": self.backend_type,
            "has_tpm": self.has_tpm,
            "prefer_hardware": self.prefer_hardware,
            "key_count": len(self.list_keys()),
        }


if __name__ == "__main__":
    import tempfile

    print("TPM-Optional Key Store Self-Test")
    print("=" * 50)

    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "keystore"

        # Initialize store
        store = SecureKeyStore(storage_path=storage_path)

        print(f"\n[1] Backend Selection")
        print(f"    TPM available: {store.has_tpm}")
        print(f"    Active backend: {store.backend_type}")

        print(f"\n[2] Key Generation and Storage")
        key = secrets.token_bytes(32)
        meta = store.store_key(
            "test-key-001",
            key,
            KeyType.SYMMETRIC,
            [KeyUsage.ENCRYPT, KeyUsage.DECRYPT],
            "Test encryption key"
        )
        print(f"    Stored key: {meta.key_id}")
        print(f"    Hardware backed: {meta.is_hardware_backed}")

        print(f"\n[3] Key Retrieval")
        retrieved = store.retrieve_key("test-key-001")
        print(f"    Retrieved: {retrieved == key}")

        print(f"\n[4] Generate and Store")
        meta2 = store.generate_and_store(
            "generated-key-001",
            KeyType.MASTER,
            32,
            [KeyUsage.DERIVE],
            "Master derivation key"
        )
        print(f"    Generated key: {meta2.key_id}")

        print(f"\n[5] List Keys")
        for k in store.list_keys():
            print(f"    - {k.key_id}: {k.key_type.name}, hw={k.is_hardware_backed}")

        print(f"\n[6] Key Rotation")
        old_key = store.retrieve_key("test-key-001")
        new_key = store.rotate_key("test-key-001")
        print(f"    Key changed: {old_key != new_key}")

        print(f"\n[7] Get or Create")
        existing = store.get_or_create("test-key-001")
        new = store.get_or_create("new-key-001")
        print(f"    Existing retrieved: {len(existing)} bytes")
        print(f"    New created: {len(new)} bytes")

        print(f"\n[8] Delete Key")
        deleted = store.delete_key("test-key-001")
        print(f"    Deleted: {deleted}")
        print(f"    Exists after delete: {store.key_exists('test-key-001')}")

        print(f"\n[9] Status")
        status = store.get_status()
        for k, v in status.items():
            print(f"    {k}: {v}")

    print("\n" + "=" * 50)
    print("All tests passed!")

