"""
TEMPEST-Grade Audit Logging System

Implements operational security features inspired by TEMPEST standards:
- Encrypted audit logs with integrity verification
- Automatic credential redaction
- Tamper-evident logging
- Secure log rotation
- Cryptographic signatures
"""

import hashlib
import hmac
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

from app.core.config import settings

logger = structlog.get_logger()


class TEMPESTLogger:
    """
    TEMPEST-grade audit logger with encryption and integrity verification
    """

    # Patterns to redact from logs
    CREDENTIAL_PATTERNS = [
        (r'password[\"\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', '[REDACTED_PASSWORD]'),
        (r'token[\"\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', '[REDACTED_TOKEN]'),
        (r'api[_-]?key[\"\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', '[REDACTED_API_KEY]'),
        (r'secret[\"\']?\s*[:=]\s*["\']?([^"\'}\s,]+)', '[REDACTED_SECRET]'),
        (r'Bearer\s+([A-Za-z0-9\-._~+/]+)', 'Bearer [REDACTED_TOKEN]'),
        (r'Basic\s+([A-Za-z0-9+/=]+)', 'Basic [REDACTED_CREDENTIALS]'),
        # Credit cards
        (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED_CC]'),
        # SSN
        (r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]'),
    ]

    def __init__(self, log_dir: str = "/var/log/memshadow/audit"):
        """
        Initialize TEMPEST logger

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption key from settings
        self.encryption_key = self._derive_encryption_key()
        self.cipher = Fernet(self.encryption_key)

        # HMAC key for integrity verification
        self.hmac_key = settings.FIELD_ENCRYPTION_KEY.encode()

        # Chain hash for tamper detection
        self.last_log_hash = self._load_last_hash()

        logger.info("TEMPEST audit logger initialized", log_dir=str(self.log_dir))

    def _derive_encryption_key(self) -> bytes:
        """
        Derive encryption key from settings using PBKDF2

        Returns:
            Fernet-compatible encryption key
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'memshadow_tempest_salt',  # In production, use secure random salt
            iterations=100000,
        )

        key = kdf.derive(settings.FIELD_ENCRYPTION_KEY.encode())

        # Fernet requires base64-encoded 32-byte key
        import base64
        return base64.urlsafe_b64encode(key)

    def _load_last_hash(self) -> str:
        """
        Load the hash of the last log entry for chain verification

        Returns:
            Last log hash or genesis hash
        """
        chain_file = self.log_dir / "chain.hash"

        if chain_file.exists():
            with open(chain_file, 'r') as f:
                return f.read().strip()

        # Genesis hash
        return hashlib.sha256(b"TEMPEST_GENESIS").hexdigest()

    def _save_last_hash(self, log_hash: str):
        """
        Save the hash of the current log entry

        Args:
            log_hash: Hash to save
        """
        chain_file = self.log_dir / "chain.hash"

        with open(chain_file, 'w') as f:
            f.write(log_hash)

    def _redact_credentials(self, message: str) -> str:
        """
        Redact sensitive information from log message

        Args:
            message: Original message

        Returns:
            Redacted message
        """
        redacted = message

        for pattern, replacement in self.CREDENTIAL_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

        return redacted

    def _calculate_integrity_hash(self, log_entry: Dict[str, Any]) -> str:
        """
        Calculate HMAC-SHA256 hash for integrity verification

        Args:
            log_entry: Log entry dictionary

        Returns:
            HMAC hash
        """
        # Create canonical representation
        canonical = json.dumps(log_entry, sort_keys=True)

        # Include previous hash for chain
        chain_data = f"{self.last_log_hash}:{canonical}"

        # Calculate HMAC
        h = hmac.new(
            self.hmac_key,
            chain_data.encode(),
            hashlib.sha256
        )

        return h.hexdigest()

    def audit(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        action: str = "",
        resource: str = "",
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ) -> str:
        """
        Create an audit log entry

        Args:
            event_type: Type of event (authentication, authorization, data_access, etc.)
            user_id: User ID performing the action
            action: Action performed
            resource: Resource affected
            status: Success or failure
            details: Additional details
            severity: Log severity (info, warning, error, critical)

        Returns:
            Log entry ID
        """
        # Create log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'status': status,
            'severity': severity,
            'details': details or {},
            'system': 'MEMSHADOW',
            'version': settings.VERSION
        }

        # Redact sensitive information
        log_entry_str = json.dumps(log_entry)
        redacted_str = self._redact_credentials(log_entry_str)
        log_entry = json.loads(redacted_str)

        # Calculate integrity hash (before encryption)
        integrity_hash = self._calculate_integrity_hash(log_entry)
        log_entry['integrity_hash'] = integrity_hash
        log_entry['chain_hash'] = self.last_log_hash

        # Update chain
        self.last_log_hash = integrity_hash
        self._save_last_hash(integrity_hash)

        # Encrypt the entire log entry
        log_json = json.dumps(log_entry)
        encrypted_log = self.cipher.encrypt(log_json.encode())

        # Write to file (one file per day)
        log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.log.enc"

        with open(log_file, 'ab') as f:
            # Write length-prefixed encrypted entry
            f.write(len(encrypted_log).to_bytes(4, byteorder='big'))
            f.write(encrypted_log)
            f.write(b'\n')

        # Also log to standard logger (redacted)
        logger.info(
            "AUDIT",
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            status=status,
            severity=severity,
            integrity_hash=integrity_hash[:16]
        )

        return integrity_hash

    def verify_integrity(self, log_file_path: str) -> Dict[str, Any]:
        """
        Verify the integrity of an audit log file

        Args:
            log_file_path: Path to encrypted log file

        Returns:
            Verification results
        """
        results = {
            'total_entries': 0,
            'verified_entries': 0,
            'failed_entries': 0,
            'chain_valid': True,
            'errors': []
        }

        try:
            with open(log_file_path, 'rb') as f:
                previous_hash = hashlib.sha256(b"TEMPEST_GENESIS").hexdigest()

                while True:
                    # Read length prefix
                    length_bytes = f.read(4)
                    if not length_bytes:
                        break

                    length = int.from_bytes(length_bytes, byteorder='big')

                    # Read encrypted entry
                    encrypted_log = f.read(length)
                    f.read(1)  # Skip newline

                    # Decrypt
                    try:
                        decrypted = self.cipher.decrypt(encrypted_log)
                        log_entry = json.loads(decrypted)

                        results['total_entries'] += 1

                        # Verify chain
                        if log_entry.get('chain_hash') != previous_hash:
                            results['chain_valid'] = False
                            results['failed_entries'] += 1
                            results['errors'].append(
                                f"Chain break at entry {results['total_entries']}"
                            )
                        else:
                            # Verify HMAC
                            stored_hash = log_entry.pop('integrity_hash')
                            chain_hash = log_entry.pop('chain_hash')

                            # Recalculate with previous hash
                            self.last_log_hash = previous_hash
                            calculated_hash = self._calculate_integrity_hash(log_entry)

                            if calculated_hash == stored_hash:
                                results['verified_entries'] += 1
                            else:
                                results['failed_entries'] += 1
                                results['errors'].append(
                                    f"HMAC mismatch at entry {results['total_entries']}"
                                )

                            previous_hash = stored_hash

                    except Exception as e:
                        results['failed_entries'] += 1
                        results['errors'].append(
                            f"Failed to decrypt entry {results['total_entries']}: {str(e)}"
                        )

        except Exception as e:
            results['errors'].append(f"Failed to verify log file: {str(e)}")

        logger.info(
            "Audit log verification complete",
            file=log_file_path,
            **results
        )

        return results

    def read_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> list:
        """
        Read and decrypt audit logs within a date range

        Args:
            start_date: Start date
            end_date: End date
            event_type: Filter by event type
            user_id: Filter by user ID

        Returns:
            List of log entries
        """
        entries = []

        # Iterate through log files in date range
        current_date = start_date.date()
        while current_date <= end_date.date():
            log_file = self.log_dir / f"audit_{current_date.strftime('%Y%m%d')}.log.enc"

            if log_file.exists():
                try:
                    with open(log_file, 'rb') as f:
                        while True:
                            # Read length prefix
                            length_bytes = f.read(4)
                            if not length_bytes:
                                break

                            length = int.from_bytes(length_bytes, byteorder='big')

                            # Read encrypted entry
                            encrypted_log = f.read(length)
                            f.read(1)  # Skip newline

                            # Decrypt
                            decrypted = self.cipher.decrypt(encrypted_log)
                            log_entry = json.loads(decrypted)

                            # Apply filters
                            if event_type and log_entry.get('event_type') != event_type:
                                continue

                            if user_id and log_entry.get('user_id') != user_id:
                                continue

                            # Check timestamp
                            entry_time = datetime.fromisoformat(log_entry['timestamp'])
                            if start_date <= entry_time <= end_date:
                                entries.append(log_entry)

                except Exception as e:
                    logger.error(f"Failed to read log file {log_file}: {str(e)}")

            # Next day
            from datetime import timedelta
            current_date = current_date + timedelta(days=1)

        return entries


# Global instance
tempest_logger = TEMPESTLogger()
