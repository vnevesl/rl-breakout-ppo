"""Callbacks customizados do treinamento."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MilestoneCallback(BaseCallback):
    """Registra quando o agente atinge certas pontuações médias pela primeira vez.

    Responde ao item (i) — "quanto tempo seu agente demorou para aprender uma
    política minimamente inteligente?" — com timesteps E tempo de relógio.

    Também loga ``time/wall_clock_hours`` no TensorBoard. O estado fica em
    ``milestones.json`` e sobrevive a retomadas (``--resume``): o tempo de relógio
    continua contando a partir do total já salvo.
    """

    def __init__(self, path: Path, thresholds: list[float], flush_every_s: float = 60.0,
                 min_episodes: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.path = Path(path)
        self.thresholds = sorted(thresholds)
        self.flush_every_s = flush_every_s
        # Média de poucas partidas é ruído (uma partida aleatória sortuda faz 5+ pontos).
        self.min_episodes = min_episodes
        self.state: dict = {"elapsed_seconds": 0.0, "milestones": {}}
        if self.path.exists():
            self.state.update(json.loads(self.path.read_text()))
        self._elapsed_before = float(self.state["elapsed_seconds"])
        self._t0 = time.time()
        self._last_flush = 0.0

    def _elapsed(self) -> float:
        return self._elapsed_before + (time.time() - self._t0)

    def _on_training_start(self) -> None:
        self._t0 = time.time()

    def _on_step(self) -> bool:
        buf = self.model.ep_info_buffer
        if buf and len(buf) >= self.min_episodes:
            mean_rew = float(np.mean([ep["r"] for ep in buf]))
            for thr in self.thresholds:
                key = f"{thr:g}"
                if key not in self.state["milestones"] and mean_rew >= thr:
                    hours = self._elapsed() / 3600
                    self.state["milestones"][key] = {
                        "timesteps": int(self.num_timesteps),
                        "wall_clock_hours": round(hours, 3),
                        "mean_reward_last_100": round(mean_rew, 2),
                    }
                    if self.verbose:
                        print(f"[milestone] média {mean_rew:.1f} >= {thr:g} em "
                              f"{self.num_timesteps:,} passos / {hours:.2f} h")
                    self._flush()
        if time.time() - self._last_flush > self.flush_every_s:
            self.logger.record("time/wall_clock_hours", self._elapsed() / 3600)
            self._flush()
        return True

    def _on_training_end(self) -> None:
        self._flush()

    def _flush(self) -> None:
        self._last_flush = time.time()
        self.state["elapsed_seconds"] = round(self._elapsed(), 1)
        self.state["last_timesteps"] = int(self.num_timesteps)
        self.path.write_text(json.dumps(self.state, indent=2))
