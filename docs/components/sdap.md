# SDAP (Secure Databurst Archival Protocol)

**Document ID:** MEMSHADOW-IMPL-001 (Adapted from ADJUNCT1.md) & MEMSHADOW-IMPL-005 (JANUS Addendum)

The Secure Databurst Archival Protocol (SDAP) is designed for the nightly, fully automated archival of the complete MEMSHADOW system state to a secure, Commander-controlled server. It prioritizes data sovereignty, integrity, and confidentiality, with zero reliance on external third-party services.

## 1. Strategic Mandate

-   To execute automated nightly backups of the entire system state.
-   To ensure backups are transferred to a secure, operator-controlled archival server.
-   To prioritize data sovereignty, integrity, and confidentiality.
-   To operate without reliance on third-party services.

## 2. As-Built Implementation

### 2.1 Primary Backup Script: `sdap_backup.sh`

This script is the core agent on the primary MEMSHADOW server (VPS).

```bash
#!/bin/bash
#
# Secure Databurst Archival Protocol (SDAP) - Execution Agent
# CLASSIFICATION: TOP SECRET // ARCHITECT
#

set -euo pipefail # Fail on error, unbound variable, or pipe failure

# --- CONFIGURATION ---
DATE_STAMP=$(date +"%Y-%m-%d_%H%M%S")
BACKUP_DIR="/tmp/sdap_staging" # Temporary staging directory
ARCHIVE_NAME="memshadow-databurst-${DATE_STAMP}.tar.xz"
ARCHIVE_PATH="${BACKUP_DIR}/${ARCHIVE_NAME}"

DB_USER="memshadow_user"
DB_NAME="memshadow_db"
PG_DUMP_PATH="${BACKUP_DIR}/postgres_dump.sql"

CHROMA_DATA_DIR="/var/lib/memshadow/chroma" # Path to ChromaDB data

ARCHIVAL_SERVER_USER="sdap_receiver"
ARCHIVAL_SERVER_IP="10.10.0.50" # IP of the secure archival server
ARCHIVAL_SERVER_PATH="/srv/archives/memshadow/" # Target path on archival server
SSH_KEY_PATH="/root/.ssh/sdap_tx_key" # SSH key for secure transmission

# --- EXECUTION ---
echo "[SDAP] Starting backup for ${DATE_STAMP}..."

# 1. Create a clean staging directory
rm -rf "${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"
trap 'rm -rf -- "$BACKUP_DIR"' EXIT # Ensure cleanup on normal exit

# 2. Dump PostgreSQL Database
echo "[SDAP] Dumping PostgreSQL database (${DB_NAME})..."
pg_dump -U "${DB_USER}" -d "${DB_NAME}" -F c -f "${PG_DUMP_PATH}" # Use custom format for pg_restore

# 3. Create Compressed Archive of Database Dump and Chroma Data
echo "[SDAP] Compressing data stores (PostgreSQL dump and Chroma data)..."
tar -Jcf "${ARCHIVE_PATH}" "${PG_DUMP_PATH}" "${CHROMA_DATA_DIR}"
# -J uses xz compression, -c creates, -f specifies archive file

# 4. Encrypt the Archive (using GPG)
# GPG_PASSPHRASE must be injected as an environment variable by the calling service (e.g., systemd).
if [[ -z "${GPG_PASSPHRASE}" ]]; then
    echo "[SDAP] CRITICAL: GPG_PASSPHRASE not set. Aborting." >&2
    exit 1
fi
ENCRYPTED_PATH="${ARCHIVE_PATH}.gpg"
echo "[SDAP] Encrypting archive with AES256..."
echo "$GPG_PASSPHRASE" | gpg --batch --yes --passphrase-fd 0 -c --cipher-algo AES256 -o "${ENCRYPTED_PATH}" "${ARCHIVE_PATH}"

# 5. Secure Transmission (SCP) and Verification
echo "[SDAP] Transmitting to archival server at ${ARCHIVAL_SERVER_IP}..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${ENCRYPTED_PATH}" ${ARCHIVAL_SERVER_USER}@${ARCHIVAL_SERVER_IP}:${ARCHIVAL_SERVER_PATH}

echo "[SDAP] Verifying integrity of transmitted file..."
LOCAL_HASH=$(sha256sum "${ENCRYPTED_PATH}" | awk '{print $1}')
REMOTE_HASH=$(ssh -i "${SSH_KEY_PATH}" ${ARCHIVAL_SERVER_USER}@${ARCHIVAL_SERVER_IP} "sha256sum ${ARCHIVAL_SERVER_PATH}/${ARCHIVE_NAME}.gpg" | awk '{print $1}')

if [ "$LOCAL_HASH" == "$REMOTE_HASH" ]; then
    echo "[SDAP] SUCCESS: Integrity verified. Backup complete for ${DATE_STAMP}."
    # The trap will handle cleanup of the staging directory.
else
    echo "[SDAP] CRITICAL: HASH MISMATCH. Transmission failed for ${DATE_STAMP}. Staging directory at ${BACKUP_DIR} retained for manual inspection." >&2
    trap - EXIT # Disable cleanup trap on failure to allow inspection
    exit 1
fi

exit 0
```

**Key Script Features:**
-   Uses `set -euo pipefail` for robustness.
-   Creates a temporary staging directory that is cleaned up on successful exit.
-   Dumps the PostgreSQL database using `pg_dump`. The custom format (`-F c`) is generally recommended for use with `pg_restore`.
-   Archives both the PostgreSQL dump and the ChromaDB data directory using `tar` with `xz` compression.
-   Encrypts the archive using `gpg` with AES256. The passphrase must be provided as an environment variable (`GPG_PASSPHRASE`).
-   Transmits the encrypted archive using `scp` with a dedicated SSH key. `StrictHostKeyChecking=no` and `UserKnownHostsFile=/dev/null` are used for automation but imply trust in the network path or pre-configured known host.
-   Verifies the integrity of the transmitted file by comparing local and remote SHA256 hashes.
-   Retains the staging directory on failure for manual inspection.

### 2.2 Automation via `systemd` on VPS

The `sdap_backup.sh` script is intended to be run automatically via `systemd` timers.

*   **Service File (`/etc/systemd/system/sdap-backup.service`):**
    ```ini
    [Unit]
    Description=MEMSHADOW Secure Databurst Archival Protocol Service
    Wants=network-online.target
    After=network-online.target postgresql.service # Ensure DB is up

    [Service]
    Type=oneshot
    # Load passphrase from a secure, root-only file
    EnvironmentFile=/etc/memshadow/sdap.env # This file should contain GPG_PASSPHRASE=your_secret_passphrase
    ExecStart=/usr/local/bin/sdap_backup.sh
    User=root # Script requires root for some operations (e.g., placing SSH key, accessing /var/lib)
    ```
    **Note:** The `EnvironmentFile` (`/etc/memshadow/sdap.env`) must be secured (e.g., `chmod 600` and owned by root). It should contain the `GPG_PASSPHRASE`.

*   **Timer File (`/etc/systemd/system/sdap-backup.timer`):**
    ```ini
    [Unit]
    Description=Run MEMSHADOW SDAP Backup nightly

    [Timer]
    OnCalendar=daily # Runs once a day, typically at midnight
    RandomizedDelaySec=15min # Adds a random delay up to 15 minutes to distribute load
    Persistent=true # Ensures the timer runs if the system was down at the scheduled time

    [Install]
    WantedBy=timers.target
    ```
    **Enabling the timer:**
    ```bash
    sudo systemctl enable sdap-backup.timer
    sudo systemctl start sdap-backup.timer
    ```

### 2.3 Archival Server Hardening

The `sdap_receiver` user on the archival server should be heavily restricted.

*   **Restricted SSH Access:**
    Modify `/etc/ssh/sshd_config` on the archival server or, preferably, use the `command=` option in the `sdap_receiver`'s `~/.ssh/authorized_keys` file.
    *   **Example entry in `/home/sdap_receiver/.ssh/authorized_keys`:**
        ```
        command="/usr/bin/rssh -ro /srv/archives/memshadow/",no-port-forwarding,no-x11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA... a_comment_for_the_key
        ```
        This entry uses `rssh` (restricted shell) or a similar tool (like `scponly` or a custom script) to ensure the SSH key can *only* be used for `scp` into its designated directory (`/srv/archives/memshadow/`) and for running `sha256sum`. The `no-*` options further restrict SSH capabilities.

## 3. JANUS Protocol Addendum: Portable Sealing Mechanisms

The JANUS protocol addresses how the encryption key for SDAP (and potentially other sensitive operations) is managed, particularly ensuring portability and security without hard-binding to specific hardware like a TPM.

**Revised Mandate:** The security of the archival/encryption module (JANUS binary, or in this context, the GPG key used by `sdap_backup.sh`) must not be anchored to non-transferable hardware. Recompilation or re-configuration on a new host by an authorized operator is a primary security control.

### Recommended Sealing Option: Environment-Derived Sealing

This approach involves deriving the encryption key (or the GPG passphrase for SDAP) from stable, unique characteristics of the operational environment, controlled by the operator.

*   **Concept:** The `sdap_backup.sh` script (or a dedicated JANUS binary if used for GPG key management) does not contain the key material directly. Instead, it derives it from operator-defined environmental "factors."
*   **Mechanism:**
    1.  **Factor Selection:** Upon deploying to a new VPS, the operator selects stable host identifiers.
    2.  **Recommended Factors for GPG Passphrase Derivation:**
        *   **File-Based Secret:** The content of a specific, operator-created file (e.g., `/etc/memshadow/host_id.key`). This file would contain a high-entropy UUID or random string. This is the primary, portable secret.
        *   **System Identifiers (Optional, for added local binding):** Contents of stable system files like `/etc/machine-id` (on systemd systems) or the MAC address of the primary network interface.
    3.  **Key Derivation (Conceptual for `sdap_backup.sh`):**
        If the `GPG_PASSPHRASE` itself were to be derived instead of being stored directly in `sdap.env`:
        ```bash
        # Conceptual modification within sdap_backup.sh or a preparatory script
        FACTOR1=$(cat /etc/memshadow/host_id.key)
        FACTOR2=$(cat /etc/machine-id) # Example additional factor
        COMBINED_FACTORS="${FACTOR1}:${FACTOR2}"
        # Use a KDF (e.g., openssl kdf, or a custom utility) to derive the GPG passphrase
        # GPG_PASSPHRASE=$(echo "$COMBINED_FACTORS" | openssl kdf -kdf HKDF -keylen 32 -hash sha256 -salt "JANUS_SDAP_SEAL" -info "GPGPassphrase" | base64)
        # This derived passphrase would then be used by the GPG command.
        ```
        However, for SDAP as currently specified, the `GPG_PASSPHRASE` is directly sourced from `sdap.env`. The "Environment-Derived Sealing" would apply more directly if JANUS were a separate binary managing the GPG key itself.

*   **Portability Workflow:**
    *   The same `sdap_backup.sh` script can be moved to a new VPS.
    *   To make it functional, the operator recreates the environment:
        *   Places the `/etc/memshadow/host_id.key` file with the correct secret content.
        *   Ensures `/etc/memshadow/sdap.env` contains the correct `GPG_PASSPHRASE`. If the passphrase itself is derived, then the factors for derivation must be reconstituted.

**Current SDAP Implementation:** The `sdap_backup.sh` script relies on the `GPG_PASSPHRASE` being securely stored in `/etc/memshadow/sdap.env`. The portability of this secret file (or its contents) is key for disaster recovery. The JANUS "Environment-Derived Sealing" provides a more advanced way to protect this passphrase if it were not stored directly.

## 4. Recovery Process

1.  Provision a new host.
2.  Install necessary software (PostgreSQL, GPG, tar, rssh/scponly on archival server etc.).
3.  Restore the `sdap_backup.sh` script and `/etc/memshadow/sdap.env` (containing `GPG_PASSPHRASE`) or the mechanism to derive it.
4.  Set up the SSH key (`sdap_tx_key`) for communication with the archival server (or vice-versa if pulling).
5.  Securely transfer the required encrypted databurst (`.tar.xz.gpg`) from the archival server to the new host.
6.  Decrypt the archive:
    ```bash
    echo "$GPG_PASSPHRASE" | gpg --batch --yes --passphrase-fd 0 -d -o memshadow-databurst-YYYY-MM-DD_HHMMSS.tar.xz memshadow-databurst-YYYY-MM-DD_HHMMSS.tar.xz.gpg
    ```
7.  Extract the archive:
    ```bash
    tar -Jxf memshadow-databurst-YYYY-MM-DD_HHMMSS.tar.xz -C /tmp/restore_staging
    ```
8.  Restore PostgreSQL database:
    ```bash
    pg_restore -U memshadow_user -d memshadow_db /tmp/restore_staging/postgres_dump.sql
    ```
9.  Restore ChromaDB data:
    Copy the contents of `/tmp/restore_staging/chroma` to the new `/var/lib/memshadow/chroma` directory.
10. Verify system functionality.

This protocol ensures that system backups are automated, secure, and recoverable under operator control, forming a critical part of MEMSHADOW's resilience.
