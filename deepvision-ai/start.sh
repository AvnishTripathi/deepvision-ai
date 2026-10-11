#!/usr/bin/env bash
set -e

echo "=== DeepVision AI — Unified Single Service Startup ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/frontend" ]; then
  REPO_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../frontend" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$SCRIPT_DIR"
fi

echo "Repository root detected at: $REPO_ROOT"

# 1. Start FastAPI backend on 127.0.0.1:8000
echo "Starting FastAPI backend service on 127.0.0.1:8000..."
cd "$REPO_ROOT"
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend --workers 1 &
BACKEND_PID=$!

trap "kill -TERM $BACKEND_PID 2>/dev/null || true" EXIT SIGINT SIGTERM

# 2. Wait for FastAPI backend health check
echo "Waiting for FastAPI engine health check..."
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:8000/api/v1/health >/dev/null 2>&1; then
    echo "FastAPI engine is healthy and operational."
    break
  fi
  sleep 1
done

# 3. Start Next.js frontend on public Render port
PORT="${PORT:-3000}"
echo "Starting Next.js frontend on public port $PORT..."
cd "$REPO_ROOT/frontend"
exec npx next start -p "$PORT"
