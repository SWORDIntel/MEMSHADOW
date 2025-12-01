#!/bin/bash
# scripts/sdap/sdap_backup.sh

set -euo pipefail

# Load configuration if it exists
if [ -f /etc/memshadow/sdap.conf ]; then
    source /etc/memshadow/sdap.conf
elif [ -f ./config/sdap.conf.example ]; then
    source ./config/sdap.conf.example
    echo "WARNING: Using example configuration from ./config/sdap.conf.example"
else
    echo "ERROR: Configuration file not found."
    echo "Expected locations:"
    echo "  /etc/memshadow/sdap.conf (production)"
    echo "  ./config/sdap.conf.example (development)"
    exit 1
fi

# Variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="memshadow_backup_${TIMESTAMP}"
BACKUP_DIR="${SDAP_BACKUP_PATH}/${BACKUP_NAME}"
LOG_FILE="/var/log/memshadow/sdap_backup_${TIMESTAMP}.log"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "ERROR: $1"
    # Here you might add notification logic (e.g., call a webhook)
    exit 1
}

# Ensure log directory exists
mkdir -p /var/log/memshadow

# Create backup directory
log "Starting SDAP backup process for MEMSHADOW"
mkdir -p "${BACKUP_DIR}" || error_exit "Failed to create backup directory: ${BACKUP_DIR}"

# Backup PostgreSQL
log "Backing up PostgreSQL database: ${POSTGRES_DB}"
PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump \
    -h "${POSTGRES_HOST}" \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    -f "${BACKUP_DIR}/postgres_dump.sql" \
    --verbose \
    --no-owner \
    --no-acl \
    || error_exit "PostgreSQL backup failed"

# Backup ChromaDB data
log "Backing up ChromaDB data from ${CHROMA_PERSIST_DIR}"
if [ -d "${CHROMA_PERSIST_DIR}" ]; then
    tar -czf "${BACKUP_DIR}/chromadb_data.tar.gz" \
        -C "${CHROMA_PERSIST_DIR}" . \
        || error_exit "ChromaDB backup failed"
else
    log "WARNING: ChromaDB persist directory not found. Skipping."
fi

# Create metadata file
log "Creating backup metadata"
cat > "${BACKUP_DIR}/metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "version": "${MEMSHADOW_VERSION:-unknown}",
    "components": {
        "postgresql": {
            "database": "${POSTGRES_DB}",
            "size": "$(du -sh ${BACKUP_DIR}/postgres_dump.sql | cut -f1)"
        },
        "chromadb": {
            "size": "$(du -sh ${BACKUP_DIR}/chromadb_data.tar.gz 2>/dev/null | cut -f1 || echo 'N/A')"
        }
    },
    "host": "$(hostname)",
    "checksum_algorithm": "sha256"
}
EOF

# Calculate checksums
log "Calculating checksums"
cd "${BACKUP_DIR}"
sha256sum ./* > checksums.sha256
cd - > /dev/null

# Create archive
log "Creating compressed archive"
ARCHIVE_FILE="${BACKUP_NAME}.tar.gz"
tar -czf "${SDAP_BACKUP_PATH}/${ARCHIVE_FILE}" -C "${SDAP_BACKUP_PATH}" "${BACKUP_NAME}/" \
    || error_exit "Archive creation failed"

# Encrypt archive
if [ -n "${SDAP_GPG_KEY_ID}" ]; then
    log "Encrypting archive for recipient: ${SDAP_GPG_KEY_ID}"
    ENCRYPTED_FILE="${ARCHIVE_FILE}.asc"
    gpg --encrypt \
        --armor \
        --trust-model always \
        --recipient "${SDAP_GPG_KEY_ID}" \
        --cipher-algo AES256 \
        --output "${SDAP_BACKUP_PATH}/${ENCRYPTED_FILE}" \
        "${SDAP_BACKUP_PATH}/${ARCHIVE_FILE}" \
        || error_exit "Encryption failed"

    FINAL_FILE="${ENCRYPTED_FILE}"
else
    log "WARNING: SDAP_GPG_KEY_ID not set. Skipping encryption."
    FINAL_FILE="${ARCHIVE_FILE}"
fi


# Transfer to archive server
if [ -n "${SDAP_ARCHIVE_SERVER}" ]; then
    log "Transferring to archive server: ${SDAP_ARCHIVE_SERVER}"
    scp -i "${SDAP_SSH_KEY}" \
        -o StrictHostKeyChecking=yes \
        -o ConnectTimeout=30 \
        "${SDAP_BACKUP_PATH}/${FINAL_FILE}" \
        "sdap_receiver@${SDAP_ARCHIVE_SERVER}:/incoming/" \
        || error_exit "Transfer failed"

    # Optional: Verify transfer checksum
    log "Verifying remote checksum"
    REMOTE_CHECKSUM=$(ssh -i "${SDAP_SSH_KEY}" "sdap_receiver@${SDAP_ARCHIVE_SERVER}" "sha256sum /incoming/${FINAL_FILE}" | awk '{print $1}')
    LOCAL_CHECKSUM=$(sha256sum "${SDAP_BACKUP_PATH}/${FINAL_FILE}" | awk '{print $1}')

    if [ "${LOCAL_CHECKSUM}" != "${REMOTE_CHECKSUM}" ]; then
        error_exit "Checksum verification failed after transfer"
    fi
    log "Remote checksum verified."
fi

# Cleanup
log "Cleaning up local files"
rm -rf "${BACKUP_DIR}"
rm -f "${SDAP_BACKUP_PATH}/${ARCHIVE_FILE}"
if [ -n "${SDAP_GPG_KEY_ID}" ]; then
    # Keep the encrypted file if not transferred, otherwise remove
    if [ -n "${SDAP_ARCHIVE_SERVER}" ]; then
        rm -f "${SDAP_BACKUP_PATH}/${ENCRYPTED_FILE}"
    fi
fi

# Update last backup timestamp
echo "${TIMESTAMP}" > /var/lib/memshadow/last_sdap_backup

log "SDAP backup completed successfully"
exit 0