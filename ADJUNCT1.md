**ARCHITECT ONLINE**
Welcome back Commander.

**GMT: 23-06-2025 00:58:36**

Directive re-confirmed. You require a more granular, as-built technical specification for Protocols CHIMERA and SDAP, formatted as a complete markdown document. My previous transmission served as a doctrinal overview; this document will serve as the formal implementation blueprint for engineering and deployment.

Understood. Executing.

-----

````markdown
# Doctrinal Implementation Blueprint: Protocols CHIMERA & SDAP

**CLASSIFICATION:** TOP SECRET // ARCHITECT // ORCON / NOFORN
**DOCUMENT ID:** MEMSHADOW-IMPL-001
**EFFECTIVE DATE:** 23-06-2025
**SUBJECT:** As-Built Technical Specification for Deception and Archival Subsystems

---

## **PART 1: PROJECT CHIMERA (ISOLATED & WEAPONIZED MEMORY PROTOCOL)**

### **1.0 Strategic Mandate & Threat Model**

**1.1 Mandate:** To engineer and deploy a multi-layered deception framework capable of detecting, identifying, and optionally neutralizing adversarial access, while providing a cast-iron guarantee against the contamination of the primary `memshadow_main` memory pool.

**1.2 Threat Model:** This architecture specifically mitigates the following threats:
* **T1. Accidental Contamination:** An operator error or buggy client retrieving deceptive data during standard operations.
* **T2. Malicious Querying:** An adversary with partial access (e.g., a stolen read-only token) attempting to discover the deception layer via enumeration or blended queries.
* **T3. API Parameter Tampering:** An adversary attempting to manipulate API calls to force the backend to query the wrong data store.
* **T4. SQL/Vector Injection:** An adversary attempting to craft an injection that bridges the gap between the primary and deceptive data stores at the database level.

---

### **1.1 Architectural Blueprint: The Segregation Doctrine**

#### **1.1.1 Data Layer Segregation**

The protocol mandates absolute physical and logical separation at the storage layer.

**A. PostgreSQL Metadata Store:**
Two distinct tables with no shared foreign keys.

* **Primary Table Definition:**
    ```sql
    -- File: postgres_schema_main.sql
    CREATE TABLE memories_primary (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        content_hash VARCHAR(64) UNIQUE NOT NULL,
        encrypted_content TEXT NOT NULL,
        metadata_json JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW(),
        last_accessed_at TIMESTAMPTZ
    );
    ```
* **CHIMERA Table Definition:**
    ```sql
    -- File: postgres_schema_chimera.sql
    CREATE TABLE memories_chimera (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        lure_text_hash VARCHAR(64) UNIQUE NOT NULL,
        lure_text_encrypted TEXT NOT NULL,
        payload_type VARCHAR(50) NOT NULL,
        payload_encoding VARCHAR(20) NOT NULL,
        payload_data TEXT NOT NULL,         -- Base64 encoded payload
        trigger_condition VARCHAR(100),
        alert_priority VARCHAR(20) DEFAULT 'CRITICAL',
        deployment_date TIMESTAMPTZ DEFAULT NOW()
    );

    -- Table for logging trigger events
    CREATE TABLE chimera_triggers (
        id BIGSERIAL PRIMARY KEY,
        chimera_id UUID REFERENCES memories_chimera(id),
        triggered_at TIMESTAMPTZ DEFAULT NOW(),
        source_ip INET,
        source_user_agent TEXT,
        details JSONB
    );
    ```

**B. ChromaDB Vector Store:**
Two distinct, named collections.

* Primary Vectors: `memshadow_main`
* CHIMERA Vectors: `memshadow_chimera`

The application service account credentials for standard operation **will not** have `write` or `delete` permissions on the `memshadow_chimera` collection.

#### **1.1.2 Application Layer Segregation (FastAPI)**

Access is controlled via rigidly scoped API endpoints.

* **Standard Retrieval Logic:**
    ```python
    # /app/routers/standard_ops.py
    from app.auth import require_scope_and_log

    @router.post("/retrieve")
    @require_scope_and_log(scope="memory:read")
    async def retrieve_memory(query: Query, db: AsyncSession = Depends(get_db)):
        # Namespace is a non-negotiable constant
        PRIMARY_COLLECTION = "memshadow_main"
        vector_matches = await vector_store.query(collection_name=PRIMARY_COLLECTION, ...)
        # Logic to query 'memories_primary' table
        ...
    ```

* **CHIMERA Endpoint Logic:**
    ```python
    # /app/routers/red_ops.py
    from app.auth import require_scope_and_log

    @router.post("/chimera/deploy")
    @require_scope_and_log(scope="chimera:write")
    async def deploy_chimera_asset(asset: ChimeraAsset, db: AsyncSession = Depends(get_db)):
        # Namespace is a non-negotiable constant
        CHIMERA_COLLECTION = "memshadow_chimera"
        # Logic to embed lure_text and store in CHIMERA_COLLECTION
        # Logic to insert metadata into 'memories_chimera' table
        ...
    ```

#### **1.1.3 Trigger & Alerting Sequence (Mermaid Diagram)**

This sequence illustrates the designed trigger mechanism for a polyglot payload.

```mermaid
sequenceDiagram
    participant Adversary
    participant MEMSHADOW_API
    participant ChimeraTrigger_DB
    participant Commander_Client

    Adversary->>MEMSHADOW_API: GET /assets/chm_a7b3c9d1.png (Downloads Polyglot)
    Note over Adversary: Renames/executes payload: `./chm_a7b3c9d1.png`
    Adversary->>MEMSHADOW_API: (Beacon Execution) POST /beacon/chm_a7b3c9d1
    activate MEMSHADOW_API
    MEMSHADOW_API->>ChimeraTrigger_DB: INSERT INTO chimera_triggers (...)
    activate ChimeraTrigger_DB
    ChimeraTrigger_DB-->>MEMSHADOW_API: Success
    deactivate ChimeraTrigger_DB
    MEMSHADOW_API-->>Adversary: 200 OK (or deceptive response)
    deactivate MEMSHADOW_API

    Commander_Client->>MEMSHADOW_API: (Polling) GET /api/v1/red/alerts
    activate MEMSHADOW_API
    MEMSHADOW_API->>ChimeraTrigger_DB: SELECT * FROM chimera_triggers WHERE new
    activate ChimeraTrigger_DB
    ChimeraTrigger_DB-->>MEMSHADOW_API: [New Trigger Event Data]
    deactivate ChimeraTrigger_DB
    MEMSHADOW_API-->>Commander_Client: [Alert Payload JSON]
    deactivate MEMSHADOW_API
````

-----

## **PART 2: SECURE DATABURST ARCHIVAL PROTOCOL (SDAP)**

### **2.0 Strategic Mandate**

To execute a nightly, fully automated archival of the complete system state to a secure, Commander-controlled server. This protocol prioritizes data sovereignty, integrity, and confidentiality above all else, with zero reliance on external third-party services.

### **2.1 As-Built Implementation Scripts**

#### **2.1.1 VPS Backup Script: `sdap_backup.sh`**

This script is the core execution agent on the primary MEMSHADOW server.

```bash
#!/bin/bash
#
# Secure Databurst Archival Protocol (SDAP) - Execution Agent
# CLASSIFICATION: TOP SECRET // ARCHITECT
#

set -euo pipefail # Fail on error, unbound variable, or pipe failure

# --- CONFIGURATION ---
DATE_STAMP=$(date +"%Y-%m-%d_%H%M%S")
BACKUP_DIR="/tmp/sdap_staging"
ARCHIVE_NAME="memshadow-databurst-${DATE_STAMP}.tar.xz"
ARCHIVE_PATH="${BACKUP_DIR}/${ARCHIVE_NAME}"

DB_USER="memshadow_user"
DB_NAME="memshadow_db"
PG_DUMP_PATH="${BACKUP_DIR}/postgres_dump.sql"

CHROMA_DATA_DIR="/var/lib/memshadow/chroma"

ARCHIVAL_SERVER_USER="sdap_receiver"
ARCHIVAL_SERVER_IP="10.10.0.50" # IP of secure archival server
ARCHIVAL_SERVER_PATH="/srv/archives/memshadow/"
SSH_KEY_PATH="/root/.ssh/sdap_tx_key"

# --- EXECUTION ---
echo "[SDAP] Starting backup for ${DATE_STAMP}..."

# 1. Create a clean staging directory
rm -rf "${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"
trap 'rm -rf -- "$BACKUP_DIR"' EXIT # Ensure cleanup on exit

# 2. Dump PostgreSQL Database
echo "[SDAP] Dumping PostgreSQL database..."
pg_dump -U "${DB_USER}" -d "${DB_NAME}" -f "${PG_DUMP_PATH}"

# 3. Create Compressed Archive
echo "[SDAP] Compressing data stores..."
tar -Jcf "${ARCHIVE_PATH}" "${PG_DUMP_PATH}" "${CHROMA_DATA_DIR}"

# 4. Encrypt the Archive
# GPG_PASSPHRASE must be injected as an environment variable by the calling service.
if [[ -z "${GPG_PASSPHRASE}" ]]; then
    echo "[SDAP] CRITICAL: GPG_PASSPHRASE not set. Aborting." >&2
    exit 1
fi
ENCRYPTED_PATH="${ARCHIVE_PATH}.gpg"
echo "[SDAP] Encrypting archive with AES256..."
echo "$GPG_PASSPHRASE" | gpg --batch --yes --passphrase-fd 0 -c --cipher-algo AES256 -o "${ENCRYPTED_PATH}" "${ARCHIVE_PATH}"

# 5. Secure Transmission and Verification
echo "[SDAP] Transmitting to archival server at ${ARCHIVAL_SERVER_IP}..."
scp -i "${SSH_KEY_PATH}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${ENCRYPTED_PATH}" ${ARCHIVAL_SERVER_USER}@${ARCHIVAL_SERVER_IP}:${ARCHIVAL_SERVER_PATH}

echo "[SDAP] Verifying integrity of transmitted file..."
LOCAL_HASH=$(sha256sum "${ENCRYPTED_PATH}" | awk '{print $1}')
REMOTE_HASH=$(ssh -i "${SSH_KEY_PATH}" ${ARCHIVAL_SERVER_USER}@${ARCHIVAL_SERVER_IP} "sha256sum ${ARCHIVAL_SERVER_PATH}/${ARCHIVE_NAME}.gpg" | awk '{print $1}')

if [ "$LOCAL_HASH" == "$REMOTE_HASH" ]; then
    echo "[SDAP] SUCCESS: Integrity verified. Backup complete."
    # The trap will handle cleanup of the staging directory.
else
    echo "[SDAP] CRITICAL: HASH MISMATCH. Transmission failed. Staging directory at ${BACKUP_DIR} retained for manual inspection." >&2
    trap - EXIT # Disable cleanup trap on failure
    exit 1
fi

exit 0
```

#### **2.1.2 Automation via `systemd` on VPS**

  * **Service File:** `/etc/systemd/system/sdap-backup.service`

    ```ini
    [Unit]
    Description=MEMSHADOW Secure Databurst Archival Protocol Service
    Wants=network-online.target
    After=network-online.target postgresql.service

    [Service]
    Type=oneshot
    # Load passphrase from a secure, root-only file
    EnvironmentFile=/etc/memshadow/sdap.env
    ExecStart=/usr/local/bin/sdap_backup.sh
    User=root
    ```

  * **Timer File:** `/etc/systemd/system/sdap-backup.timer`

    ```ini
    [Unit]
    Description=Run MEMSHADOW SDAP Backup nightly

    [Timer]
    OnCalendar=daily
    RandomizedDelaySec=15min # Adds ±7.5 min random delay
    Persistent=true

    [Install]
    WantedBy=timers.target
    ```

#### **2.1.3 Archival Server Hardening**

To restrict the `sdap_receiver` user, modify `/etc/ssh/sshd_config` on the archival server or, preferably, use the `command=` option in the `~/.ssh/authorized_keys` file.

  * **Entry in `/home/sdap_receiver/.ssh/authorized_keys`:**
    ```
    command="/usr/bin/rssh -ro /srv/archives/memshadow/",no-port-forwarding,no-x11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA... a_comment
    ```
    This entry, using `rssh` (restricted shell) or a similar tool, ensures the SSH key can *only* be used for `scp` into its designated directory and nothing else.

-----

```
```
