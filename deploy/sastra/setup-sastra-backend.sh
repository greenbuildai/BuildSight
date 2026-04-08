#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/opt/buildsight}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

if [[ ! -d .venv-inference ]]; then
  "$PYTHON_BIN" -m venv .venv-inference
fi

source .venv-inference/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r deploy/backend/requirements-inference.txt

echo "SASTRA inference environment ready in $ROOT_DIR/.venv-inference"
