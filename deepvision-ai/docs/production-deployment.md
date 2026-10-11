# DeepVision AI — Production Deployment & Systems Architecture Guide

## 1. Systems Architecture Overview

DeepVision AI is an enterprise-grade AI computer vision platform delivering high-throughput face detection, landmark localization, optical quality evaluation, and multi-face trajectory tracking.

The production runtime separates API orchestration, queue distribution, asynchronous AI workers, relational metadata storage, and media artifacts.

```
                    ┌────────────────────────┐
                    │  Internet / Clients    │
                    └───────────┬────────────┘
                                │ HTTPS (TLS 1.3)
                                ▼
                    ┌────────────────────────┐
                    │ Reverse Proxy (NGINX)  │
                    │ - SSL Termination      │
                    │ - Static Assets Cache  │
                    │ - Security Headers     │
                    └─────┬────────────┬─────┘
                          │            │
            Proxy /api/*  │            │ Proxy /*
                          ▼            ▼
             ┌────────────────┐    ┌─────────────────┐
             │ FastAPI Engine │    │ Next.js App     │
             │ (Python 3.12)  │    │ (Node.js 20)    │
             │ UID 1000       │    │ UID 1001        │
             └────────┬───────┘    └─────────────────┘
                      │
        ┌─────────────┼─────────────────────────────┐
        ▼             ▼                             ▼
  ┌────────────┐ ┌───────────────────┐ ┌─────────────────────────┐
  │ PostgreSQL │ │ Redis 7           │ │ Storage Abstraction     │
  │ (Alembic)  │ │ ├── Rate Limiting │ │ ├── Temp (Frames)       │
  │            │ │ ├── Cache         │ │ ├── Media (Uploads)     │
  │            │ │ └── Job Queue     │ │ └── Results (Annotated) │
  └────────────┘ └─────────┬─────────┘ └─────────────────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ DeepVision Worker │
                 │ (video_worker.py) │
                 │ UID 1000          │
                 └───────────────────┘
```

---

## 2. Background Processing & Queue Architecture

### 2.1 Queue & Worker Pipeline (Step 7.7)
For production workloads, video processing jobs are decoupled from synchronous HTTP request loops:
1. **FastAPI Client Route**: Receives upload, enforces magic bytes/MIME/size/resolution validations, and submits job via `RedisVideoJobManager`.
2. **Concurrency Throttling**: Verifies active job count against `MAX_CONCURRENT_VIDEO_JOBS` (`VIDEO_MAX_CONCURRENT_JOBS`). Saturated queues return HTTP `429 Too Many Requests`.
3. **Queue Ingestion**: Serializes job metadata into Redis Hash (`deepvision:job:{job_id}`) and pushes job identifier to Redis List (`deepvision:queue:video_jobs`).
4. **Worker Execution**: Dedicated background worker daemon (`app/workers/video_worker.py`) executes `BLPOP` to process jobs sequentially or across worker processes.
5. **Storage & Status**: Saves annotated MP4 outputs to partitioned `results/` storage; sets status `completed` and records full metrics.
6. **Graceful Fallback**: When Redis is not configured (`REDIS_URL` empty or disabled), `InMemoryVideoJobManager` automatically executes jobs via thread pool executor without breaking development workflows.

### 2.2 Worker Process Execution
Run the standalone worker process:
```bash
python -m app.workers.video_worker
```

---

## 3. Database Architecture & Alembic Migrations

### 3.1 PostgreSQL Domain Schema (Step 7.9)
The application utilizes SQLAlchemy 2.0 with the modern `psycopg 3` binary driver (`postgresql+psycopg://`).

Entities:
- `users`: Platform authorization and access credentials.
- `media_assets`: Media catalog recording dimensions, codecs, file sizes, and opaque `storage_key` references (raw media is NEVER stored in SQL).
- `model_versions`: Registry of active neural network weights, tasks, versions, and SHA-256 hashes.
- `analysis_jobs`: Asynchronous execution records tracking status (`queued`, `processing`, `completed`, `failed`, `cancelled`), progress (0–100), and timings.
- `analysis_results`: Structured numerical output: bounding boxes, landmarks, optical scores, and latency telemetry.
- `system_logs`: Structured audit trail with request correlation and error classification.

### 3.2 Running Migrations (Step 7.10)
All schema changes are tracked via Alembic migrations under `backend/alembic/`:
```bash
cd backend

# Apply all migrations to the latest revision
alembic upgrade head

# Verify migration history
alembic current
```

---

## 4. Media Storage Architecture

### 4.1 Segregated Storage Namespaces (Step 7.11)
`LocalStorageBackend` segregates physical disk storage:
- `storage/temp/`: Ephemeral extracted video frames and temporary scratch buffers.
- `storage/media/`: Consented user-uploaded image and video inputs.
- `storage/results/`: Output annotated video files and analysis exports.

### 4.2 Object Storage Abstraction
`ObjectStorageBackend` provides unified interfaces (`store_file`, `store_bytes`, `get_local_path`, `exists`, `delete`):
- Prepared for Amazon S3 (`s3://`), Google Cloud Storage (`gs://`), and Azure Blob (`az://`).
- Emulated mode activates automatically during local development without requiring cloud credentials or mocking libraries.

---

## 5. Health, Liveness & Readiness Probes

The service implements cloud-native probes under `/api/v1/health`:
- `GET /api/v1/health`: Service identity and status.
- `GET /api/v1/health/live`: Fast liveness probe verifying event loop responsiveness.
- `GET /api/v1/health/ready`: Readiness probe verifying PostgreSQL connection, Redis reachability, and AI model weight readiness. Returns HTTP 503 if a critical dependency is down.

---

## 6. Structured Logging & Request Correlation

### 6.1 Standardized Log Schema (Step 7.12)
All log entries capture:
- `timestamp`: ISO-8601 UTC timestamp.
- `level`: Log severity (INFO, WARNING, ERROR, CRITICAL).
- `service`: Service identifier (`deepvision-core`).
- `request_id`: Correlated UUID tracing header (`X-Request-ID`).
- `job_id`: Video job identifier (when applicable).
- `endpoint`: HTTP route path.
- `processing_time_ms`: Execution latency in milliseconds.
- `error_type`: Standardized error classification string.

### 6.2 Data Leakage Prevention
Application logs are filtered through `SensitiveDataFilter`:
- Automatically redacts passwords, tokens, API keys, and session secrets.
- Prevents logging of raw image bytes, bytearrays, or base64 data URIs.

---

## 7. Local Docker Deployment

The entire stack is configured to start with a single command without external prerequisites:

```bash
docker compose up --build
```

### Services Started:
* `postgres`: PostgreSQL 16 on port 5432 with health check.
* `redis`: Redis 7 on port 6379 with health check.
* `backend`: FastAPI API engine on port 8000 (depends on postgres + redis healthy).
* `worker`: Standalone video processing background daemon (depends on redis + postgres healthy).
* `frontend`: Next.js 14 web dashboard on port 3000.

---

## 8. Cloud Virtual Machine Deployment (Ubuntu 22.04 / 24.04 LTS)

### 8.1 System Preparation
```bash
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y docker.io docker-compose-v2 git nginx certbot python3-certbot-nginx
sudo usermod -aG docker $USER
```

### 8.2 Environment Configuration
Clone the repository, create production secrets, and place model binaries:
```bash
git clone https://github.com/organization/deepvision-ai.git /opt/deepvision-ai
cd /opt/deepvision-ai

# Generate secure production configuration
cp .env.production.example .env.production
# Edit .env.production with strong passwords and production SECRET_KEY
nano .env.production
```

### 8.3 Systemd Service Management
Create a systemd unit to guarantee automatic restart across server reboots:
```ini
# /etc/systemd/system/deepvision.service
[Unit]
Description=DeepVision AI Container Stack
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/deepvision-ai
ExecStart=/usr/bin/docker compose --env-file .env.production up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable deepvision.service
sudo systemctl start deepvision.service
```

### 8.4 NGINX Reverse Proxy & TLS Configuration
```nginx
# /etc/nginx/sites-available/deepvision
server {
    server_name vision.yourdomain.com;

    client_max_body_size 500M;

    # Frontend Next.js routes
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API routes
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }
}
```

Issue Let's Encrypt TLS certificate:
```bash
sudo certbot --nginx -d vision.yourdomain.com
```

---

## 9. GPU Server Deployment (NVIDIA CUDA Acceleration)

DeepVision AI can utilize discrete NVIDIA GPUs for real-time video stream processing at 150+ FPS:

### 9.1 Host Driver & Container Toolkit Setup
Install official NVIDIA drivers and the container toolkit:
```bash
sudo apt-get install -y nvidia-driver-535 nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU availability:
```bash
nvidia-smi
```

### 9.2 Docker Compose GPU Pass-Through
In `docker-compose.yml`, assign GPU resources to the `worker` service:
```yaml
  worker:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - AI_ENABLE_GPU=true
      - AI_DEVICE=cuda
```

---

## 10. Managed Cloud Container Platforms (Architectural Guidance)

The decoupled microservices model maps cleanly onto major enterprise cloud platforms without requiring vendor lock-in:

### 10.1 Amazon Web Services (AWS)
* **API Service**: AWS ECS on Fargate (autoscaled based on HTTP request count / ALB metrics).
* **Video Worker Pool**: AWS ECS on EC2 with G4dn / G5 instances (NVIDIA T4 / A10G) autoscaled via KEDA monitoring Redis queue depth.
* **Database**: Amazon RDS for PostgreSQL with automated backups.
* **Queue / Cache**: Amazon ElastiCache for Redis (in-memory cluster).
* **Storage**: Amazon S3 for media storage; configure `STORAGE_BACKEND=s3` and bucket policies.

### 10.2 Google Cloud Platform (GCP)
* **API Service**: Google Cloud Run (serverless container autoscaling).
* **Video Worker Pool**: Google Kubernetes Engine (GKE) with L4 GPU node pools.
* **Database**: Cloud SQL for PostgreSQL.
* **Queue / Cache**: Memorystore for Redis.
* **Storage**: Google Cloud Storage (GCS); configure `STORAGE_BACKEND=gcs`.

### 10.3 Microsoft Azure
* **API Service**: Azure Container Apps (managed serverless ingress).
* **Video Worker Pool**: Azure Kubernetes Service (AKS) with NCasT4_v3 GPU node pools.
* **Database**: Azure Database for PostgreSQL Flexible Server.
* **Queue / Cache**: Azure Cache for Redis.
* **Storage**: Azure Blob Storage; configure `STORAGE_BACKEND=azure`.


