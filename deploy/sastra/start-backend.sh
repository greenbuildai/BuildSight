#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${BUILDSIGHT_ROOT:-/opt/buildsight}"
cd "$ROOT_DIR"

if [[ -f deploy/sastra/backend.env ]]; then
  set -a
  source deploy/sastra/backend.env
  set +a
fi

exec "$ROOT_DIR/.venv-inference/bin/python" "$ROOT_DIR/dashboard/backend/server.py"
