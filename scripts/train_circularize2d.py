#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train + evaluate a PPO agent on the Circularize-2D milestone.

Uses Stable-Baselines3 PPO (a known-correct continuous-control implementation) so
that when the agent doesn't learn, the problem is the env/reward, not an agent
bug. Evaluation reports success rate and mean Δv used vs the analytic
single-impulse apoapsis-circularization baseline.

    python scripts/train_circularize2d.py --timesteps 300000 --seed 0

This is glue/experiment code (excluded from the strict-typed library).
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tbot.envs.circularize2d import Circularize2DConfig, Circularize2DEnv


def build_env():
    return Circularize2DEnv(Circularize2DConfig())


def evaluate(model, n_episodes: int = 50, seed: int = 10_000) -> dict:
    env = build_env()
    successes = 0
    dv_ratios = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        baseline = info["baseline_dv"]
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        if info["success"]:
            successes += 1
            dv_ratios.append(info["dv_used"] / baseline)
    return {
        "n": n_episodes,
        "success_rate": successes / n_episodes,
        "mean_dv_ratio": float(np.mean(dv_ratios)) if dv_ratios else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--save", type=str, default="models/circularize2d_ppo")
    ap.add_argument("--eval-episodes", type=int, default=50)
    args = ap.parse_args()

    vec_env = make_vec_env(build_env, n_envs=args.n_envs, seed=args.seed)
    model = PPO(
        "MlpPolicy", vec_env,
        gamma=Circularize2DConfig().gamma,
        n_steps=2048, batch_size=256, gae_lambda=0.95,
        learning_rate=3e-4, ent_coef=0.0, verbose=1, seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.save)
    print(f"saved model to {args.save}.zip")

    stats = evaluate(model, n_episodes=args.eval_episodes)
    print(f"EVAL: success_rate={stats['success_rate']:.2%}  "
          f"mean_dv_ratio(agent/baseline)={stats['mean_dv_ratio']:.3f}  "
          f"(n={stats['n']})")


if __name__ == "__main__":
    main()
