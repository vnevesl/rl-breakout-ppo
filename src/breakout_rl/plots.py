"""Lê os eventos do TensorBoard e gera gráficos estáticos para o relatório.

Uso:
    python -m breakout_rl.plots --run runs/ppo_breakout --out reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

TAGS = {
    "rollout/ep_rew_mean": "Pontuação média por partida (últimas 100)",
    "rollout/ep_len_mean": "Duração média da partida (passos)",
    "eval/mean_reward": "Pontuação na avaliação periódica",
    "train/entropy_loss": "Entropy loss (exploração)",
    "train/value_loss": "Value loss (crítico)",
    "train/approx_kl": "Approx. KL (tamanho da atualização)",
}


def load_scalars(run_dir: Path) -> pd.DataFrame:
    """Junta todos os escalares de todas as sessões (inclusive retomadas)."""
    frames = []
    for ev_file in sorted((run_dir / "tensorboard").rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            frames.append(pd.DataFrame(
                [(tag, e.step, e.value, e.wall_time) for e in acc.Scalars(tag)],
                columns=["tag", "step", "value", "wall_time"],
            ))
    if not frames:
        return pd.DataFrame(columns=["tag", "step", "value", "wall_time"])
    df = pd.concat(frames).sort_values(["tag", "step"])
    return df.drop_duplicates(["tag", "step"], keep="last")


def plot_run(run_dir: Path, out_dir: Path) -> list[Path]:
    df = load_scalars(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    tags = [t for t in TAGS if t in set(df["tag"])]
    if not tags:
        print("Nenhum escalar encontrado ainda.")
        return paths
    fig, axes = plt.subplots(len(tags), 1, figsize=(9, 2.6 * len(tags)), sharex=True)
    axes = [axes] if len(tags) == 1 else axes
    for ax, tag in zip(axes, tags, strict=True):
        d = df[df["tag"] == tag]
        ax.plot(d["step"] / 1e6, d["value"], lw=1.4)
        ax.set_title(TAGS[tag], fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Passos de treino (milhões)")
    fig.tight_layout()
    p = out_dir / f"{run_dir.name}_curves.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    print(f"Salvo: {p}")
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=Path("runs/ppo_breakout"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    args = ap.parse_args()
    plot_run(args.run, args.out)


if __name__ == "__main__":
    main()
