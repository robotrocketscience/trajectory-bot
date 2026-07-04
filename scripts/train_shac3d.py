#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SHAC (Short-Horizon Actor-Critic) for the 3D decision-layer circularize env.

Motivation: backpropagating the analytic policy gradient through the full ~900-step
rollout is a documented gradient-pathology regime (Metz 2021; Suh 2022) — the
product of per-step Jacobians explodes/vanishes and thruster on/off adds bias. Our
plain diff-sim (scripts/train_diffsim3d.py) flatlined at 0% with a barely-moving
loss, consistent with that.

SHAC (Xu et al. 2022, arXiv:2204.07137) truncates BPTT to short windows of `H`
decisions and bootstraps the tail with a learned critic (GAE/TD(λ) target, Polyak
target critic). Each window's backward chain is H*decision_repeat substeps — an
order of magnitude shorter — which conditions the gradient AND cuts backward cost.

Reuses the batched torch dynamics/observation from train_diffsim3d (decision-layer:
policy outputs an orbit-frame thrust direction + throttle; a deterministic pointing
controller slews the vehicle).

    python scripts/train_shac3d.py --updates 400 --batch 256 --window 6 --n-windows 8

Experiment code (excluded from the strict-typed library).
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import train_diffsim3d as base  # dynamics, obs, Policy, sample_orbits, evaluate

W_E = 1.0
K_FUEL = 0.1


def step_decision(state, fuel, coeffs, throttle):
    """One decision = decision_repeat substeps with the pointing controller (differentiable)."""
    dv = torch.zeros(state.shape[0], device=state.device)
    for _ in range(base.REPEAT):
        t, w, s = base.orbit_frame(state[:, 0:3], state[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(1e-6)
        omega = base.point_rate(state[:, 6:10], d)
        gate = (fuel > 0).float()
        thr = throttle * gate
        dv_sub = thr * base.A_THRUST * base.DT
        fuel = fuel - dv_sub
        dv = dv + dv_sub
        state = base.rk4(state, omega, thr)
    return state, fuel, dv


def decision_reward(state, rt, dv):
    """Dense per-decision reward: drive toward circular (a_err, e -> 0) at low Δv."""
    a, e, r = base.elements(state)
    a_err = ((a - rt).abs() / rt).clamp(max=5.0)
    return -(a_err + W_E * e.clamp(max=2.0)) - K_FUEL * dv


class Critic(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def shac_update(policy, critic, target_critic, actor_opt, critic_opt,
                B, H, n_windows, gamma, lam, device, gen):
    state, rt = base.sample_orbits(B, device, gen)
    fuel = torch.full((B,), base.DV_BUDGET, device=device)
    obs_hist, rew_hist = [], []

    actor_opt.zero_grad()
    for _ in range(n_windows):
        window_loss = torch.zeros((), device=device)
        discount = 1.0
        for _ in range(H):
            obs = base.observe(state, rt, fuel.clamp_min(0.0))
            act = policy(obs)
            coeffs = act[:, 0:3]
            throttle = act[:, 3].clamp(0.0, 1.0)
            obs_hist.append(obs.detach())
            state, fuel, dv = step_decision(state, fuel, coeffs, throttle)
            r = decision_reward(state, rt, dv)
            rew_hist.append(r.detach())
            window_loss = window_loss - (discount * r).mean()
            discount *= gamma
        # bootstrap terminal value (gradient flows through state -> policy)
        v_end = critic(base.observe(state, rt, fuel.clamp_min(0.0)))
        window_loss = window_loss - (discount * v_end).mean()
        window_loss.backward()                       # short chain: H decisions only
        state = state.detach()
        fuel = fuel.detach()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    actor_opt.step()

    # ---- critic: TD(λ) / GAE return targets from the collected episode ----
    with torch.no_grad():
        obs_all = torch.stack(obs_hist)              # [T, B, 13]
        rews = torch.stack(rew_hist)                 # [T, B]
        T = obs_all.shape[0]
        vals = target_critic(obs_all)                # [T, B]
        v_final = target_critic(base.observe(state, rt, fuel.clamp_min(0.0)))
        returns = torch.zeros_like(rews)
        gae = torch.zeros(B, device=device)
        next_val = v_final
        for t in reversed(range(T)):
            delta = rews[t] + gamma * next_val - vals[t]
            gae = delta + gamma * lam * gae
            returns[t] = gae + vals[t]
            next_val = vals[t]

    critic_opt.zero_grad()
    pred = critic(obs_all)
    critic_loss = ((pred - returns) ** 2).mean()
    critic_loss.backward()
    critic_opt.step()
    with torch.no_grad():
        for p, tp in zip(critic.parameters(), target_critic.parameters()):
            tp.mul_(0.99).add_(0.01 * p)
    return float(rews.mean()), float(critic_loss.detach())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=6)        # H decisions per BPTT window
    ap.add_argument("--n-windows", type=int, default=8)     # windows per episode
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="models/circularize3d_shac.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    policy = base.Policy().to(device)
    critic = Critic().to(device)
    target_critic = Critic().to(device)
    target_critic.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(policy.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    best = -1.0
    for it in range(args.updates):
        rmean, closs = shac_update(policy, critic, target_critic, actor_opt, critic_opt,
                                   args.batch, args.window, args.n_windows,
                                   args.gamma, args.lam, device, gen)
        if it % args.eval_every == 0 or it == args.updates - 1:
            s = base.evaluate(policy, device, n_episodes=30)
            print(f"update {it:4d}  r_mean={rmean:.3f}  critic_loss={closs:.3f}  "
                  f"success={s:.2%}", flush=True)
            if s > best:
                best = s
                torch.save(policy.state_dict(), args.save)
    policy.load_state_dict(torch.load(args.save))
    s = base.evaluate(policy, device, n_episodes=200)
    print(f"FINAL(best): success={s:.2%}  saved {args.save}")


if __name__ == "__main__":
    main()
