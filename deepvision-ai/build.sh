#!/usr/bin/env bash
set -e

echo "=== DeepVision AI — Unified Single Service Build ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/frontend" ]; then
  REPO_ROOT="$SCRIPT_DIR"
elif [ -d "$SCRIPT_DIR/../frontend" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$SCRIPT_DIR"
fi

echo "Repository root detected at: $REPO_ROOT"

# 1. Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
if [ -f "$REPO_ROOT/backend/requirements.txt" ]; then
  pip install -r "$REPO_ROOT/backend/requirements.txt"
elif [ -f "$REPO_ROOT/requirements.txt" ]; then
  pip install -r "$REPO_ROOT/requirements.txt"
fi

# 2. Build Next.js frontend
echo "Installing frontend dependencies and building Next.js..."
cd "$REPO_ROOT/frontend"
npm install
npm run build

echo "=== Build Complete ==="
