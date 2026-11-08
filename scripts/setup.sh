#!/usr/bin/env bash
# Cria o ambiente virtual e instala o projeto (macOS/Linux). Requer uv: https://docs.astral.sh/uv/
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,play]"
# No macOS, arquivos .pth podem receber a flag "hidden" e o Python 3.12+ os ignora.
if [[ "$(uname)" == "Darwin" ]]; then
  chflags nohidden .venv/lib/python3.12/site-packages/*.pth 2>/dev/null || true
fi
.venv/bin/python -c "import breakout_rl, ale_py, stable_baselines3; print('Ambiente OK')"
