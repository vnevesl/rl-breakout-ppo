"""Grava vídeos do agente (itens f e k) com uma faixa de informações sobreposta.

A faixa mostra tempo do vídeo, passo, placar e vidas, e um ``*_events.csv`` lista
cada tijolo quebrado e cada vida perdida com o instante no vídeo — isso facilita
apontar a jogada "inteligente" e a "burra" (item l).

Uso:
    python -m breakout_rl.record --policy random --out reports/videos/random.mp4
    python -m breakout_rl.record --policy runs/ppo_breakout/best_model/best_model.zip \
        --out reports/videos/trained.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from breakout_rl.config import EnvConfig, ExperimentConfig
from breakout_rl.envs import make_eval_env

SCALE = 3
BAR_H = 36


def _font(size: int = 15) -> ImageFont.ImageFont:
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(frame: np.ndarray, text: str, font: ImageFont.ImageFont) -> np.ndarray:
    h, w = frame.shape[:2]
    big = Image.fromarray(frame).resize((w * SCALE, h * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (big.width, big.height + BAR_H), (15, 15, 15))
    canvas.paste(big, (0, BAR_H))
    ImageDraw.Draw(canvas).text((8, 9), text, fill=(240, 240, 240), font=font)
    return np.asarray(canvas)


def record(
    policy: str,
    out: Path,
    env_cfg: EnvConfig,
    max_steps: int = 4000,
    episodes: int = 1,
    fps: int = 30,
    deterministic: bool = True,
    stall_steps: int = 1500,
    seed: int = 7,
) -> dict:
    """Roda ``episodes`` partidas completas (5 vidas) e salva MP4 + CSV de eventos.

    ``policy`` = "random" ou caminho para um .zip do PPO. ``stall_steps`` encerra a
    partida se o agente ficar esse tempo sem pontuar (ex.: esquecer de apertar FIRE).
    """
    env = make_eval_env(env_cfg, seed=seed, render_mode="rgb_array")
    model = None
    if policy != "random":
        from stable_baselines3 import PPO

        model = PPO.load(policy, device="cpu")

    out.parent.mkdir(parents=True, exist_ok=True)
    font = _font()
    events: list[dict] = []
    scores: list[float] = []
    writer = imageio.get_writer(out, fps=fps, macro_block_size=1)
    step_total = 0
    try:
        for ep in range(episodes):
            obs = env.reset()
            score, lives, since_reward = 0.0, None, 0
            for step in range(max_steps):
                if model is None:
                    action = np.array([env.action_space.sample()])
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                r, info = float(reward[0]), infos[0]
                t = step_total / fps
                score += r
                since_reward = 0 if r > 0 else since_reward + 1
                cur_lives = info.get("lives")
                if r > 0:
                    events.append({"episode": ep, "video_time_s": round(t, 2), "step": step,
                                   "event": "tijolo", "reward": r, "score": score})
                if lives is not None and cur_lives is not None and cur_lives < lives:
                    events.append({"episode": ep, "video_time_s": round(t, 2), "step": step,
                                   "event": "perdeu_vida", "reward": 0, "score": score})
                lives = cur_lives
                frame = env.render()
                label = "ALEATÓRIO" if model is None else "PPO"
                writer.append_data(annotate(
                    frame,
                    f"{label} ep{ep + 1}  t={t:5.1f}s  placar {score:3.0f}  vidas {lives}",
                    font,
                ))
                step_total += 1
                if done[0] or since_reward >= stall_steps:
                    if not done[0]:
                        events.append({"episode": ep, "video_time_s": round(t, 2), "step": step,
                                       "event": "travou_sem_pontuar", "reward": 0,
                                       "score": score})
                    break
            scores.append(score)
    finally:
        writer.close()
        env.close()

    csv_path = out.with_name(out.stem + "_events.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "video_time_s", "step", "event",
                                          "reward", "score"])
        w.writeheader()
        w.writerows(events)
    summary = {"video": str(out), "events": str(csv_path), "scores": scores,
               "duration_s": round(step_total / fps, 1)}
    print(summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="random", help='"random" ou caminho do modelo .zip')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", default="configs/ppo_breakout.yaml")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stochastic", action="store_true",
                    help="amostra ações da política em vez de pegar a mais provável")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    record(args.policy, args.out, cfg.env, max_steps=args.max_steps, episodes=args.episodes,
           fps=args.fps, deterministic=not args.stochastic, seed=args.seed)


if __name__ == "__main__":
    main()
