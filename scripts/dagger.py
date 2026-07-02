#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAgger: imitation that beats behaviour-cloning's covariate-shift ceiling.

Plain BC caps ~16% here: the cloned policy makes small action errors that
compound, drifting it off the states the expert visited, where it has never been
trained. DAgger fixes exactly this — iterate: (1) fit the policy to the current
dataset, (2) roll out the *policy* and, at every state it actually visits, record
the *expert's* action, (3) aggregate and refit. The policy learns to recover on
its own distribution and converges toward the (near-optimal) expert.

Expert = the verified latching apoapsis-burn controller (100% success, Δv≈baseline).

    python scripts/dagger.py --iters 6 --episodes 300 --bc-epochs 120
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


class Expert:
    """Stateful latching apoapsis-burn controller (velocity-frame action)."""

    def __init__(self, env: Circularize2DEnv) -> None:
        self.cfg = env.cfg
        self.rt = env._r_target
        self.vc = orb.speed_circular(self.rt, self.cfg.mu)
        self.dv_dec = self.cfg.thrust_acc_max * self.cfg.dt * self.cfg.decision_repeat
        self.prev_rdot = None
        self.burning = False

    def act(self, state) -> np.ndarray:
        x, y, vx, vy = (float(state[i]) for i in range(4))
        r = float(np.hypot(x, y)); v = float(np.hypot(vx, vy))
        rdot = (x * vx + y * vy) / r
        el = orb.orbital_elements(state, self.cfg.mu)
        if not self.burning and self.prev_rdot is not None and self.prev_rdot > 0.0 >= rdot:
            self.burning = True
        self.prev_rdot = rdot
        if self.burning and el.e >= self.cfg.e_tol and v < self.vc:
            mag = min(1.0, (self.vc - v) / self.dv_dec)
            return np.array([mag, 0.0], dtype=np.float32)
        return np.zeros(2, dtype=np.float32)


def collect_expert(n_episodes: int, seed0: int):
    """Pure expert rollouts (expert both acts and is recorded)."""
    env = build_env()
    obs_l, act_l = [], []
    succ = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        exp = Expert(env)
        term = trunc = False
        info = {}
        while not (term or trunc):
            a = exp.act(env._state)
            obs_l.append(obs); act_l.append(a)
            obs, _, term, trunc, info = env.step(a)
        succ += int(info.get("success", False))
    return (np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32),
            succ / n_episodes)


def dagger_rollout(model, n_episodes: int, seed0: int):
    """Policy drives; expert labels every visited state (the DAgger step)."""
    env = build_env()
    obs_l, act_l = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        exp = Expert(env)
        term = trunc = False
        while not (term or trunc):
            a_exp = exp.act(env._state)               # expert label for this state
            obs_l.append(obs); act_l.append(a_exp)
            a_pol, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a_pol)  # policy drives
    return np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32)


def fit(model, obs, act, epochs: int, batch: int = 512,
        burn_weight: float = 10.0, lr: float = 1e-3) -> float:
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    weight = 1.0 + burn_weight * (act_t.abs().sum(dim=1) > 0).float()
    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    n = len(obs_t)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mean = model.policy.get_distribution(obs_t[idx]).distribution.mean
            per = ((mean - act_t[idx]) ** 2).mean(dim=1)
            loss = (weight[idx] * per).sum() / weight[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        last = tot / n
    return last


def evaluate(model, n_episodes: int = 200, seed: int = 500_000) -> dict:
    env = build_env()
    succ, ratios = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        base = info["baseline_dv"]
        term = trunc = False
        while not (term or trunc):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
        if info["success"]:
            succ += 1
            ratios.append(info["dv_used"] / base)
    return {"success": succ / n_episodes,
            "dv_ratio": float(np.mean(ratios)) if ratios else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=120)
    ap.add_argument("--save", type=str, default="models/circularize2d_dagger")
    args = ap.parse_args()

    vec_env = make_vec_env(build_env, n_envs=1)
    model = PPO("MlpPolicy", vec_env, policy_kwargs={"net_arch": [128, 128]},
                verbose=0, device="auto")

    obs, act, exp_succ = collect_expert(args.episodes, seed0=0)
    print(f"expert demos: {len(obs)} transitions, controller success={exp_succ:.1%}")

    for it in range(args.iters):
        wmse = fit(model, obs, act, args.bc_epochs)
        stats = evaluate(model)
        print(f"iter {it}: dataset={len(obs)}  wmse={wmse:.5f}  "
              f"success={stats['success']:.2%}  dv_ratio={stats['dv_ratio']:.3f}")
        if it < args.iters - 1:                      # aggregate on-policy states
            n_obs, n_act = dagger_rollout(model, args.episodes // 2,
                                          seed0=10_000 * (it + 1))
            obs = np.concatenate([obs, n_obs])
            act = np.concatenate([act, n_act])

    model.save(args.save)
    print(f"saved {args.save}.zip")


if __name__ == "__main__":
    main()
