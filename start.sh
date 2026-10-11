#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/deepvision-ai" ]; then
  echo "Navigating to deepvision-ai project directory..."
  cd "$SCRIPT_DIR/deepvision-ai"
fi

exec bash start.sh
