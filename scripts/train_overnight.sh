#!/usr/bin/env bash
# Treino longo em segundo plano no Mac, sem deixar o computador dormir.
#
#   ./scripts/train_overnight.sh                         # config padrão
#   ./scripts/train_overnight.sh configs/ppo_breakout.yaml
#
# É seguro rodar de novo depois de uma interrupção: --resume continua do último
# checkpoint (salvo a cada 250 mil passos). Acompanhe com:
#   tail -f runs/ppo_breakout/train.log
#   ./scripts/tensorboard.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/ppo_breakout.yaml}"
NAME="$(grep -E '^name:' "$CONFIG" | awk '{print $2}')"
RUN_DIR="runs/${NAME}"
mkdir -p "$RUN_DIR"

if [[ -f "$RUN_DIR/train.pid" ]] && kill -0 "$(cat "$RUN_DIR/train.pid")" 2>/dev/null; then
  echo "Já existe um treino rodando (PID $(cat "$RUN_DIR/train.pid"))."
  exit 1
fi

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
# caffeinate -i: impede o sono do sistema enquanto o processo viver (a tela pode apagar).
# Mantenha o Mac na tomada e com a tampa aberta.
nohup caffeinate -i .venv/bin/python -m breakout_rl.train --config "$CONFIG" --resume \
  >> "$RUN_DIR/train.log" 2>&1 &
# O PID do Python é gravado pelo próprio processo em $RUN_DIR/train.pid.
echo "Treino iniciado. Log: $RUN_DIR/train.log"
