#!/usr/bin/env bash
# Roda os experimentos de ablação (configs/ablation/*.yaml) em sequência, em background.
#   ./scripts/run_ablation.sh
#   tail -f runs/ablation.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
nohup caffeinate -i bash -c '
  for cfg in configs/ablation/*.yaml; do
    echo "=== $(date) :: $cfg"
    .venv/bin/python -m breakout_rl.train --config "$cfg" --resume
  done
  echo "=== $(date) :: ablação concluída"
' >> runs/ablation.log 2>&1 &
echo "Ablação iniciada. Log: runs/ablation.log"
