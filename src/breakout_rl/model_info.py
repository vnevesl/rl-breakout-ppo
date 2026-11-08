"""Conta os parâmetros da rede neural da política (item j).

Uso:
    python -m breakout_rl.model_info                       # rede recém-criada
    python -m breakout_rl.model_info --model runs/.../best_model.zip
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO


def parameter_table(policy) -> list[tuple[str, tuple[int, ...], int]]:
    return [(name, tuple(p.shape), p.numel()) for name, p in policy.named_parameters()]


def summarize(policy) -> dict:
    rows = parameter_table(policy)
    groups: dict[str, int] = {}
    for name, _, n in rows:
        key = name.split(".")[0]  # features_extractor, mlp_extractor, action_net, value_net
        groups[key] = groups.get(key, 0) + n
    return {"total": sum(n for *_, n in rows),
            "trainable": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "by_block": groups, "layers": rows}


def _fresh_policy():
    from breakout_rl.config import ExperimentConfig
    from breakout_rl.envs import make_eval_env

    cfg = ExperimentConfig()
    env = make_eval_env(cfg.env, seed=0)
    try:
        return PPO("CnnPolicy", env, device="cpu").policy
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="caminho do .zip; sem ele, cria uma rede nova")
    args = ap.parse_args()
    policy = PPO.load(args.model, device="cpu").policy if args.model else _fresh_policy()
    s = summarize(policy)
    print(policy)
    print(f"\n{'camada':55s} {'shape':>20s} {'params':>12s}")
    for name, shape, n in s["layers"]:
        print(f"{name:55s} {str(shape):>20s} {n:>12,}")
    print(f"\nPor bloco: { {k: f'{v:,}' for k, v in s['by_block'].items()} }")
    print(f"TOTAL: {s['total']:,} parâmetros ({s['trainable']:,} treináveis)")


if __name__ == "__main__":
    main()
