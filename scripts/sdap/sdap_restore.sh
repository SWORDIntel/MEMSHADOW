#!/bin/bash
# scripts/sdap/sdap_restore.sh
# MEMSHADOW SDAP Restore Script

set -euo pipefail

# Load configuration
if [ -f /etc/memshadow/sdap.conf ]; then
    source /etc/memshadow/sdap.conf
elif [ -f "$(dirname "$0")/../../sdap.conf.example" ]; then
    source "$(dirname "$0")/../../sdap.conf.example"
else
    echo "ERROR: SDAP configuration not found"
    exit 1
fi

# Variables
BACKUP_ARCHIVE="${1:-}"
RESTORE_DIR="${SDAP_BACKUP_PATH}/restore_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="/var/log/memshadow/sdap_restore_$(date +%Y%m%d_%H%M%S).log"

# Functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

error_exit() {
    log "ERROR: $1"
    exit 1
}

# Validate input
if [ -z "$BACKUP_ARCHIVE" ]; then
    error_exit "Usage: $0 <backup_archive.tar.gz.asc>"
fi

if [ ! -f "$BACKUP_ARCHIVE" ]; then
    error_exit "Backup archive not found: $BACKUP_ARCHIVE"
fi

log "Starting SDAP restore process"
log "Backup archive: $BACKUP_ARCHIVE"

# Create restore directory
mkdir -p "${RESTORE_DIR}" || error_exit "Failed to create restore directory"

# Decrypt archive
log "Decrypting archive"
gpg --decrypt \
    --output "${RESTORE_DIR}/backup.tar.gz" \
    "${BACKUP_ARCHIVE}" \
    || error_exit "Decryption failed"

# Extract archive
log "Extracting archive"
cd "${RESTORE_DIR}"
tar -xzf backup.tar.gz || error_exit "Extraction failed"

# Find the extracted directory
BACKUP_DIR=$(find . -maxdepth 1 -type d -name "memshadow_backup_*" | head -1)
if [ -z "$BACKUP_DIR" ]; then
    error_exit "Backup directory not found in archive"
fi

log "Backup directory: $BACKUP_DIR"

# Verify checksums
log "Verifying checksums"
cd "$BACKUP_DIR"
if [ -f checksums.sha256 ]; then
    sha256sum -c checksums.sha256 || error_exit "Checksum verification failed"
    log "Checksums verified successfully"
fi

# Read metadata
if [ -f metadata.json ]; then
    log "Backup metadata:"
    cat metadata.json | tee -a "${LOG_FILE}"
fi

# Confirm restore
read -p "WARNING: This will overwrite the current database. Continue? (yes/no) " -r
if [ "$REPLY" != "yes" ]; then
    log "Restore cancelled by user"
    exit 0
fi

# Stop services (optional, uncomment if needed)
# log "Stopping services"
# systemctl stop memshadow-api memshadow-worker

# Restore PostgreSQL
if [ -f postgres_dump.sql ]; then
    log "Restoring PostgreSQL database"
    
    # Drop existing database and recreate
    PGPASSWORD="${POSTGRES_PASSWORD}" psql \
        -h "${POSTGRES_SERVER}" \
        -U "${POSTGRES_USER}" \
        -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};" \
        postgres || error_exit "Failed to drop existing database"
    
    PGPASSWORD="${POSTGRES_PASSWORD}" psql \
        -h "${POSTGRES_SERVER}" \
        -U "${POSTGRES_USER}" \
        -c "CREATE DATABASE ${POSTGRES_DB};" \
        postgres || error_exit "Failed to create database"
    
    # Restore from dump
    PGPASSWORD="${POSTGRES_PASSWORD}" psql \
        -h "${POSTGRES_SERVER}" \
        -U "${POSTGRES_USER}" \
        -d "${POSTGRES_DB}" \
        -f postgres_dump.sql \
        || error_exit "PostgreSQL restore failed"
    
    log "PostgreSQL database restored successfully"
fi

# Restore ChromaDB
if [ -f chromadb_data.tar.gz ]; then
    log "Restoring ChromaDB data"
    
    # Backup existing data
    if [ -d "${CHROMA_PERSIST_DIR}" ]; then
        mv "${CHROMA_PERSIST_DIR}" "${CHROMA_PERSIST_DIR}.backup.$(date +%s)"
    fi
    
    # Extract ChromaDB data
    mkdir -p "${CHROMA_PERSIST_DIR}"
    tar -xzf chromadb_data.tar.gz -C "${CHROMA_PERSIST_DIR}" \
        || error_exit "ChromaDB restore failed"
    
    log "ChromaDB data restored successfully"
fi

# Restart services (optional, uncomment if needed)
# log "Starting services"
# systemctl start memshadow-api memshadow-worker

# Update last restore timestamp
echo "$(date +%Y%m%d_%H%M%S)" > /var/lib/memshadow/last_sdap_restore

log "SDAP restore completed successfully"
log "Restore directory: ${RESTORE_DIR}"

# Send notification
if [ -n "${SDAP_NOTIFICATION_WEBHOOK:-}" ]; then
    curl -X POST "${SDAP_NOTIFICATION_WEBHOOK}" \
        -H "Content-Type: application/json" \
        -d "{\"status\": \"success\", \"operation\": \"restore\", \"archive\": \"${BACKUP_ARCHIVE}\"}" \
        2>/dev/null || true
fi

log "Restore process complete. Please verify system functionality."
