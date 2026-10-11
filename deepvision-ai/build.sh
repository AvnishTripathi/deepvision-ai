#!/usr/bin/env bash
set -e

echo "=== DeepVision AI — Unified Single Service Build ==="

# 1. Install backend Python dependencies
echo "Installing Python backend dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# 2. Install frontend Node dependencies and build Next.js
echo "Installing frontend dependencies and building Next.js..."
cd frontend
npm install
npm run build
cd ..

echo "=== Build Complete ==="
