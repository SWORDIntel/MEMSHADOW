# MEMSHADOW Docker Image
# Suitable for development and production use

FROM python:3.11-slim

# Metadata
LABEL maintainer="MEMSHADOW Team"
LABEL version="1.0.0"
LABEL description="MEMSHADOW - Advanced Memory System"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt requirements-web.txt requirements-dev.txt ./
COPY requirements/ ./requirements/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY migrations/ ./migrations/
COPY run.py run_web.py pytest.ini Makefile ./

# Create data and logs directories
RUN mkdir -p /app/data /app/logs /app/tmp

# Expose ports
EXPOSE 8000 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "run_web.py"]
