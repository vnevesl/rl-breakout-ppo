"""Jogue Breakout você mesmo (item b): ← → movem a raquete, ESPAÇO lança a bola.

Uso:
    python -m breakout_rl.play            # requer: pip install -e ".[play]"
"""

from __future__ import annotations

import argparse

import ale_py
import gymnasium as gym

gym.register_envs(ale_py)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env-id", default="ALE/Breakout-v5")
    ap.add_argument("--zoom", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    import pygame
    from gymnasium.utils.play import play

    keys = {
        (pygame.K_SPACE,): 1,  # FIRE
        (pygame.K_RIGHT,): 2,  # RIGHT
        (pygame.K_LEFT,): 3,  # LEFT
    }
    env = gym.make(args.env_id, render_mode="rgb_array")
    play(env, keys_to_action=keys, noop=0, zoom=args.zoom, fps=args.fps)


if __name__ == "__main__":
    main()
