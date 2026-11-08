"""Avalia um modelo (ou a política aleatória) em N partidas completas.

Uso:
    python -m breakout_rl.evaluate --policy random --episodes 20
    python -m breakout_rl.evaluate --policy runs/ppo_breakout/best_model/best_model.zip
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from breakout_rl.config import EnvConfig, ExperimentConfig
from breakout_rl.envs import make_eval_env


def evaluate(policy: str, env_cfg: EnvConfig, episodes: int = 20, seed: int = 123,
             deterministic: bool = False, max_steps: int = 27_000) -> dict:
    env = make_eval_env(env_cfg, seed=seed)
    model = None
    if policy != "random":
        from stable_baselines3 import PPO

        model = PPO.load(policy, device="cpu")
    scores = []
    try:
        for _ in range(episodes):
            obs, score = env.reset(), 0.0
            for _ in range(max_steps):
                if model is None:
                    action = np.array([env.action_space.sample()])
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, _ = env.step(action)
                score += float(reward[0])
                if done[0]:
                    break
            scores.append(score)
    finally:
        env.close()
    arr = np.array(scores)
    return {"policy": policy, "episodes": episodes, "mean": float(arr.mean()),
            "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max()),
            "scores": scores}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="random")
    ap.add_argument("--config", default="configs/ppo_breakout.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    print(json.dumps(evaluate(args.policy, cfg.env, args.episodes,
                              deterministic=args.deterministic), indent=2))


if __name__ == "__main__":
    main()
