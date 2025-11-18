# MEMSHADOW Production Dockerfile
# Multi-stage build for optimized production deployment
# Classification: UNCLASSIFIED

# Stage 1: Builder
FROM python:3.11-slim as builder

LABEL maintainer="MEMSHADOW Team"
LABEL classification="UNCLASSIFIED"
LABEL version="2.1"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
WORKDIR /build
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MEMSHADOW_HOME="/app"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # PostgreSQL client
    libpq5 \
    # Networking tools
    curl \
    wget \
    netcat-openbsd \
    # Chromium for web scraping
    chromium \
    chromium-driver \
    # WiFi tools (for recon operations)
    aircrack-ng \
    wireless-tools \
    # Security tools
    nmap \
    # Process management
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash memshadow && \
    mkdir -p /app /data /logs && \
    chown -R memshadow:memshadow /app /data /logs

# Copy virtual environment from builder
COPY --from=builder --chown=memshadow:memshadow /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=memshadow:memshadow . .

# Create necessary directories
RUN mkdir -p \
    /app/missions \
    /app/exfil \
    /app/payloads \
    /app/logs \
    /data/chromadb \
    /data/redis \
    && chown -R memshadow:memshadow /app /data /logs

# Switch to non-root user
USER memshadow

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose ports
# 8000: Main API
# 8443: C2 HTTPS
# 8080: TEMPEST Dashboard
EXPOSE 8000 8443 8080

# Default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
