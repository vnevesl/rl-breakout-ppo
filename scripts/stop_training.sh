#!/usr/bin/env bash
# Para o treino com SIGTERM: o processo salva um checkpoint antes de sair.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-ppo_breakout}"
PID_FILE="runs/${NAME}/train.pid"
[[ -f "$PID_FILE" ]] || { echo "Sem PID em $PID_FILE"; exit 1; }
PID="$(cat "$PID_FILE")"
kill -TERM "$PID"
echo "Sinal enviado. Aguarde o checkpoint ser salvo (veja o train.log)."
