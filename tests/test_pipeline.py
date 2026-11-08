"""Testes rápidos (segundos) do pipeline — não treinam nada de verdade."""

from pathlib import Path

import numpy as np
import pytest

from breakout_rl.config import ExperimentConfig
from breakout_rl.envs import describe_env, make_eval_env, make_train_env
from breakout_rl.train import latest_checkpoint, schedule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["ppo_breakout", "baseline_professor", "smoke"])
def test_configs_load(name):
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / f"{name}.yaml")
    assert cfg.env.env_id == "ALE/Breakout-v5"
    assert cfg.ppo.batch_size <= cfg.ppo.n_steps * cfg.env.n_envs


def test_main_config_avoids_double_frameskip():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "ppo_breakout.yaml")
    assert cfg.env.frameskip == 1, "AtariWrapper já pula 4 frames"


def test_describe_env():
    info = describe_env()
    assert info["n_actions"] == 4
    assert info["action_meanings"] == {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}
    assert info["lives"] == 5


def test_observation_pipeline():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs" / "smoke.yaml")
    env = make_train_env(cfg.env, seed=0)
    try:
        obs = env.reset()
        assert obs.shape == (cfg.env.n_envs, 84, 84, 4)
        obs, rew, done, _ = env.step(np.zeros(cfg.env.n_envs, dtype=int))
        assert set(np.unique(rew)) <= {-1.0, 0.0, 1.0}  # reward clipping no treino
    finally:
        env.close()
    eval_env = make_eval_env(cfg.env, seed=0, render_mode="rgb_array")
    try:
        eval_env.reset()
        assert eval_env.render().shape == (210, 160, 3)
    finally:
        eval_env.close()


def test_linear_schedule():
    f = schedule("linear", 2.5e-4)
    assert f(1.0) == pytest.approx(2.5e-4) and f(0.0) == 0.0
    assert schedule("constant", 0.1) == 0.1


def test_latest_checkpoint_orders_numerically(tmp_path):
    for n in (250000, 1000000, 750000):
        (tmp_path / f"ppo_breakout_{n}_steps.zip").touch()
    assert latest_checkpoint(tmp_path).name == "ppo_breakout_1000000_steps.zip"
    assert latest_checkpoint(tmp_path / "vazio") is None
