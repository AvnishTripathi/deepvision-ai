# ==============================================================================
# DeepVision AI — Production Hardened Backend Dockerfile
# Multi-stage security, non-root execution, minimal attack surface
# ==============================================================================

FROM python:3.12-slim AS runtime

WORKDIR /app

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Install required system dependencies (including ffmpeg and OpenCV graphics libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    libpq-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -g 1000 deepvision && \
    useradd -u 1000 -g deepvision -s /bin/bash -m deepvision && \
    mkdir -p /app/storage /app/models && \
    chown -R deepvision:deepvision /app

# Copy application source code and database migrations
COPY --chown=deepvision:deepvision app ./app
COPY --chown=deepvision:deepvision alembic ./alembic
COPY --chown=deepvision:deepvision alembic.ini .

# Drop root privileges
USER deepvision

EXPOSE 8000

# Container healthcheck against live health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

