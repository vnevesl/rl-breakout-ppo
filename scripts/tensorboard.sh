#!/usr/bin/env bash
# Abre o TensorBoard em http://localhost:6006 com todos os experimentos de runs/.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/tensorboard --logdir runs --port "${1:-6006}"
