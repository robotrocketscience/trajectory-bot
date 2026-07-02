#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DAgger imitation for Circularize-3D (decision-layer) from the scripted expert.

The 3D env is provably solvable — a scripted controller (continuously command
prograde so the vehicle slews to point during coast, then burn at apoapsis) hits
12/12. Yet the diff-sim / SHAC learners flatline in the coast local-optimum. So,
exactly as in 2D, clone the expert and DAgger to beat covariate shift.

Expert (velocity/orbit-frame decision-layer): always command prograde direction
[1,0,0] (the deterministic pointing controller slews to it during coast), throttle
to circular speed once past apoapsis.

    python scripts/dagger3d.py --iters 6 --episodes 300 --bc-epochs 150

Experiment code (excluded from the strict-typed library).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

import train_diffsim3d as base           # Policy, evaluate
from tbot import orbital as orb
from tbot import orbital3d as orb3
from tbot.envs.circularize3d import Circularize3DConfig, Circularize3DEnv


def build_env():
    return Circularize3DEnv(Circularize3DConfig())


class Expert:
    def __init__(self, env: Circularize3DEnv) -> None:
        self.cfg = env.cfg
        self.rt = env._r_target
        self.vc = orb.speed_circular(self.rt, self.cfg.mu)
        self.dv_dec = self.cfg.sc.a_thrust * self.cfg.dt * self.cfg.decision_repeat
        self.prev_rdot = None
        self.burning = False

    def act(self, state) -> np.ndarray:
        r_vec, v_vec = state[0:3], state[3:6]
        r = float(np.linalg.norm(r_vec)); v = float(np.linalg.norm(v_vec))
        rdot = float(np.dot(r_vec, v_vec) / r)
        el = orb3.orbital_elements3d(r_vec, v_vec, self.cfg.mu)
        if not self.burning and self.prev_rdot is not None and self.prev_rdot > 0 >= rdot:
            self.burning = True
        self.prev_rdot = rdot
        if self.burning and el.e >= self.cfg.e_tol and v < self.vc:
            mag = min(1.0, (self.vc - v) / self.dv_dec)
            return np.array([1.0, 0.0, 0.0, mag], dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)   # coast, keep pointing prograde


def collect_expert(n_episodes: int, seed0: int):
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
    return (np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32), succ / n_episodes)


def dagger_rollout(model, n_episodes: int, seed0: int, device):
    env = build_env()
    obs_l, act_l = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        exp = Expert(env)
        term = trunc = False
        while not (term or trunc):
            a_exp = exp.act(env._state)
            obs_l.append(obs); act_l.append(a_exp)
            with torch.no_grad():
                a_pol = model(torch.as_tensor(obs, dtype=torch.float32, device=device)
                              .unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, _ = env.step(a_pol)
    return np.asarray(obs_l, np.float32), np.asarray(act_l, np.float32)


def fit(model, obs, act, epochs, device, batch=512, burn_weight=15.0, lr=1e-3):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(act, dtype=torch.float32, device=device)
    weight = 1.0 + burn_weight * (act_t[:, 3] > 0).float()       # burn steps rare
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(obs_t)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model(obs_t[idx])
            per = ((pred - act_t[idx]) ** 2).mean(dim=1)
            loss = (weight[idx] * per).sum() / weight[idx].sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        last = tot / n
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--bc-epochs", type=int, default=150)
    ap.add_argument("--save", type=str, default="models/circularize3d_dagger.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = base.Policy().to(device)

    obs, act, exp_succ = collect_expert(args.episodes, seed0=0)
    print(f"expert demos: {len(obs)} transitions, controller success={exp_succ:.1%}", flush=True)

    for it in range(args.iters):
        wmse = fit(model, obs, act, args.bc_epochs, device)
        s = base.evaluate(model, device, n_episodes=100)
        print(f"iter {it}: dataset={len(obs)}  wmse={wmse:.5f}  success={s:.2%}", flush=True)
        if it < args.iters - 1:
            n_obs, n_act = dagger_rollout(model, args.episodes // 2, 10_000 * (it + 1), device)
            obs = np.concatenate([obs, n_obs])
            act = np.concatenate([act, n_act])

    torch.save(model.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
