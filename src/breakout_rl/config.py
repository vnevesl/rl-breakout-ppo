"""Configuração tipada de um experimento, carregada de um YAML em ``configs/``.

Manter todos os hiperparâmetros num arquivo versionado (e não espalhados pelo
código) permite comparar experimentos e responder ao item (h) do trabalho:
"quais parâmetros você teve que ajustar em relação ao código base".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    env_id: str = "ALE/Breakout-v5"
    n_envs: int = 8
    frame_stack: int = 4
    # O AtariWrapper do SB3 já aplica frame skip = 4 (MaxAndSkipEnv). O v5 do ALE
    # também pula 4 frames por padrão; sem frameskip=1 o agente pularia 16 frames.
    frameskip: int = 1
    # 0.25 é o padrão do v5 ("sticky actions"); 0.0 reproduz o BreakoutNoFrameskip-v4
    # usado nos benchmarks do RL Zoo.
    repeat_action_probability: float = 0.0
    terminal_on_life_loss: bool = True
    clip_reward: bool = True
    vec_env: str = "subproc"  # "subproc" (paralelo) ou "dummy" (sequencial)


@dataclass
class PPOConfig:
    policy: str = "CnnPolicy"
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    lr_schedule: str = "linear"  # "linear" ou "constant"
    clip_range: float = 0.1
    clip_schedule: str = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainConfig:
    total_timesteps: int = 10_000_000
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "mps" ou "cuda"
    checkpoint_freq: int = 250_000  # em timesteps totais (somando todos os envs)
    eval_freq: int = 250_000
    eval_episodes: int = 5
    # Recompensa média (episódio completo, 5 vidas) que consideramos uma política
    # "minimamente inteligente" — usada para responder ao item (i).
    smart_threshold: float = 30.0


@dataclass
class ExperimentConfig:
    name: str = "ppo_breakout"
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            name=raw.get("name", cls.name),
            env=EnvConfig(**raw.get("env", {})),
            ppo=PPOConfig(**raw.get("ppo", {})),
            train=TrainConfig(**raw.get("train", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True))
