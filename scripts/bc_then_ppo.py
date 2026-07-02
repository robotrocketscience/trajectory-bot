#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior-cloning warm-start, then PPO fine-tune, on Circularize-2D.

Model-free PPO from scratch plateaus below solving this task (it learns to
approach the target orbit but can't reliably hit the tight tolerance, and
destabilizes). We already have a known-good, near-optimal scripted controller
(coast to apoapsis, burn prograde to circular speed). So:

  1. Collect demonstrations from the scripted controller (guaranteed-feasible,
     Δv ≈ analytic baseline).
  2. Behavior-clone the PPO policy to those demos (supervised: regress the
     policy's mean action onto the demo action).
  3. Fine-tune with PPO to squeeze Δv / robustness.

Evaluated at each stage: success rate + mean Δv-vs-baseline. Glue/experiment
code (excluded from the strict-typed library).

    python scripts/bc_then_ppo.py --bc-episodes 300 --bc-epochs 40 --timesteps 300000
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from tbot import orbital as orb
from tbot.envs.circularize2d import Circularize2DConfig, Circularize2DEnv


def build_env():
    return Circularize2DEnv(Circularize2DConfig())


def collect_demos(n_episodes: int, seed0: int = 100_000):
    """Roll out the VERIFIED latching apoapsis-burn controller (the one the env
    solvability test proves reaches e≈0.002 at the default thrust). It coasts
    until the first apoapsis crossing (radial speed + -> -), then burns prograde
    toward the target circular speed, stopping once circular. Returns
    (obs, act) demo transitions plus the controller's own success rate."""
    env = build_env()
    cfg = env.cfg
    # Δv delivered by one full-throttle DECISION (dt * decision_repeat substeps).
    dv_per_step = cfg.thrust_acc_max * cfg.dt * cfg.decision_repeat
    obs_list, act_list = [], []
    successes = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        rt = env._r_target
        vc_target = orb.speed_circular(rt, cfg.mu)
        prev_rdot = None
        burning = False
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            s = env._state
            x, y, vx, vy = (float(s[i]) for i in range(4))
            r = float(np.hypot(x, y)); v = float(np.hypot(vx, vy))
            rdot = (x * vx + y * vy) / r
            el = orb.orbital_elements(s, cfg.mu)
            if not burning and prev_rdot is not None and prev_rdot > 0.0 >= rdot:
                burning = True
            prev_rdot = rdot
            if burning and el.e >= cfg.e_tol and v < vc_target:
                mag = min(1.0, (vc_target - v) / dv_per_step)
                act = np.array([mag, 0.0], dtype=np.float32)
            else:
                act = np.zeros(2, dtype=np.float32)
            obs_list.append(obs)
            act_list.append(act)
            obs, _, terminated, truncated, info = env.step(act)
        successes += int(info.get("success", False))
    return (np.asarray(obs_list, dtype=np.float32),
            np.asarray(act_list, dtype=np.float32),
            successes / n_episodes)


def evaluate(model, n_episodes: int = 100, seed: int = 200_000) -> dict:
    env = build_env()
    successes, dv_ratios = 0, []
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
    return {"success_rate": successes / n_episodes,
            "mean_dv_ratio": float(np.mean(dv_ratios)) if dv_ratios else float("nan")}


def behavior_clone(model, obs, act, epochs: int, batch: int = 512,
                   burn_weight: float = 10.0, lr: float = 1e-3) -> None:
    """Supervised-regress the policy's mean action onto demo actions. Burn steps
    are the minority, so they are up-weighted, else the net just learns to coast.
    Uses a dedicated optimizer at a higher LR than PPO's for a proper fit."""
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    is_burn = (act_t.abs().sum(dim=1) > 0).float()
    weight = 1.0 + burn_weight * is_burn                     # per-sample weight
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(obs_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            dist = model.policy.get_distribution(obs_t[idx])
            mean = dist.distribution.mean          # deterministic policy output
            per = ((mean - act_t[idx]) ** 2).mean(dim=1)
            loss = (weight[idx] * per).sum() / weight[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  BC epoch {ep:3d}  wmse={total / n:.5f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc-episodes", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="models/circularize2d_bcppo")
    args = ap.parse_args()

    print(f"collecting {args.bc_episodes} scripted demo episodes...")
    obs, act, demo_success = collect_demos(args.bc_episodes)
    burn_frac = float((np.abs(act).sum(axis=1) > 0).mean())
    print(f"  demos: {len(obs)} transitions, {burn_frac:.1%} burn steps, "
          f"controller success={demo_success:.1%}")

    vec_env = make_vec_env(build_env, n_envs=args.n_envs, seed=args.seed)
    model = PPO("MlpPolicy", vec_env, gamma=Circularize2DConfig().gamma,
                n_steps=2048, batch_size=256, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, verbose=0, seed=args.seed,
                device="auto")

    print(f"behavior-cloning for {args.bc_epochs} epochs...")
    behavior_clone(model, obs, act, args.bc_epochs)
    bc_stats = evaluate(model)
    print(f"AFTER BC:   success={bc_stats['success_rate']:.2%}  "
          f"dv_ratio={bc_stats['mean_dv_ratio']:.3f}")

    print(f"PPO fine-tuning for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.save)
    ft_stats = evaluate(model)
    print(f"AFTER PPO:  success={ft_stats['success_rate']:.2%}  "
          f"dv_ratio={ft_stats['mean_dv_ratio']:.3f}")


if __name__ == "__main__":
    main()
