"""Construção dos ambientes Atari (treino, avaliação e gravação de vídeo).

Pipeline de pré-processamento (padrão DQN/Nature, aplicado pelo ``AtariWrapper``):

    frame RGB 210x160  ->  NoopReset (até 30 no-ops aleatórios no início)
                       ->  MaxAndSkip (repete a ação 4 frames, max dos 2 últimos)
                       ->  EpisodicLife (perder vida = fim de episódio para o agente)
                       ->  FireReset (aperta FIRE para lançar a bola)
                       ->  escala de cinza + resize 84x84 ("se livre de pixels inúteis")
                       ->  ClipReward (sinal da recompensa: -1, 0, +1)
                       ->  VecFrameStack(4) (dá noção de velocidade/direção da bola)

O ``Monitor`` fica ANTES do AtariWrapper, então ``rollout/ep_rew_mean`` no
TensorBoard é a pontuação real do jogo (sem clipping, com as 5 vidas).
"""

from __future__ import annotations

from typing import Any

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
)

from breakout_rl.config import EnvConfig

gym.register_envs(ale_py)


def _env_kwargs(cfg: EnvConfig, render_mode: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "frameskip": cfg.frameskip,
        "repeat_action_probability": cfg.repeat_action_probability,
    }
    if render_mode:
        kwargs["render_mode"] = render_mode
    return kwargs


def make_train_env(cfg: EnvConfig, seed: int, n_envs: int | None = None) -> VecEnv:
    """VecEnv de treino com N cópias do jogo rodando em paralelo + frame stacking."""
    vec_cls = SubprocVecEnv if cfg.vec_env == "subproc" else DummyVecEnv
    venv = make_atari_env(
        cfg.env_id,
        n_envs=n_envs or cfg.n_envs,
        seed=seed,
        env_kwargs=_env_kwargs(cfg),
        wrapper_kwargs={
            "terminal_on_life_loss": cfg.terminal_on_life_loss,
            "clip_reward": cfg.clip_reward,
        },
        vec_env_cls=vec_cls,
    )
    return VecFrameStack(venv, n_stack=cfg.frame_stack)


def make_eval_env(
    cfg: EnvConfig, seed: int, render_mode: str | None = None, n_envs: int = 1
) -> VecEnv:
    """Ambiente de avaliação/gravação: mesma observação do treino, mas sem clipping
    da recompensa e sem encerrar o episódio a cada vida perdida."""
    venv = make_atari_env(
        cfg.env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=_env_kwargs(cfg, render_mode),
        wrapper_kwargs={"terminal_on_life_loss": False, "clip_reward": False},
        vec_env_cls=DummyVecEnv,
    )
    return VecFrameStack(venv, n_stack=cfg.frame_stack)


def describe_env(env_id: str = "ALE/Breakout-v5") -> dict[str, Any]:
    """Resumo do ambiente cru (itens c, d e e do trabalho)."""
    env = gym.make(env_id)
    try:
        env.reset(seed=0)
        meanings = env.unwrapped.get_action_meanings()
        spec = env.spec
        return {
            "env_id": env_id,
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "n_actions": int(env.action_space.n),
            "action_meanings": dict(enumerate(meanings)),
            "frameskip": spec.kwargs.get("frameskip") if spec else None,
            "repeat_action_probability": (
                spec.kwargs.get("repeat_action_probability") if spec else None
            ),
            "max_episode_steps": spec.max_episode_steps if spec else None,
            "lives": env.unwrapped.ale.lives(),
        }
    finally:
        env.close()
