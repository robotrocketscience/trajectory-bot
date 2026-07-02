#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable-simulation policy gradient for 2-D circularization.

Model-free RL and imitation wall out here because they discard the fact that the
dynamics are known and exactly differentiable. This does the opposite: it
reimplements the two-body RK4 dynamics + orbital elements + observation in
PyTorch (batched over parallel orbits), rolls out the policy through them, and
backpropagates the episode loss (final orbit error + Δv + impact penalty) through
the entire rollout to get an *analytic* policy gradient. No exploration noise, no
demonstrations — direct optimization of the true objective via the physics.

The Torch dynamics/observation mirror `tbot.dynamics` / `tbot.orbital` /
`Circularize2DEnv` exactly, so a policy trained here transfers to the Gymnasium
env for an apples-to-apples eval against the model-free/imitation scoreboard.

    python scripts/train_diffsim.py --iters 2000 --batch 256 --horizon 60

Experiment code (excluded from the strict-typed library).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from tbot import orbital as orb
from tbot.envs.circularize2d import Circularize2DConfig, Circularize2DEnv

MU = orb.MU_EARTH
R_BODY = orb.R_EARTH
DT = 10.0
REPEAT = 20
THRUST_MAX = 5e-3
DV_BUDGET = 2.0
ALT_PERI = (400.0, 800.0)
RA_RP = (1.3, 2.5)


# ---- differentiable two-body dynamics (batched [B,4] state) ----------------
def grav_accel(pos: torch.Tensor) -> torch.Tensor:
    r = torch.linalg.norm(pos, dim=1, keepdim=True).clamp_min(1.0)
    return -MU * pos / r**3


def deriv(state: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
    vel = state[:, 2:]
    acc = grav_accel(state[:, :2]) + thrust
    return torch.cat([vel, acc], dim=1)


def rk4(state: torch.Tensor, thrust: torch.Tensor) -> torch.Tensor:
    k1 = deriv(state, thrust)
    k2 = deriv(state + 0.5 * DT * k1, thrust)
    k3 = deriv(state + 0.5 * DT * k2, thrust)
    k4 = deriv(state + DT * k3, thrust)
    return state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def elements(state: torch.Tensor):
    """Differentiable (a, e, r) for a batch of planar states."""
    pos, vel = state[:, :2], state[:, 2:]
    r = torch.linalg.norm(pos, dim=1).clamp_min(1.0)
    v = torch.linalg.norm(vel, dim=1)
    energy = 0.5 * v**2 - MU / r
    a = -MU / (2.0 * energy)                       # <0 for hyperbolic
    rv = (pos * vel).sum(dim=1)
    factor = (v**2 - MU / r).unsqueeze(1)
    e_vec = (factor * pos - rv.unsqueeze(1) * vel) / MU
    e = torch.linalg.norm(e_vec, dim=1)
    return a, e, r


def observe(state: torch.Tensor, rt: torch.Tensor, fuel: torch.Tensor) -> torch.Tensor:
    """Mirror Circularize2DEnv._obs (8-D, normalized, clipped)."""
    a, e, r = elements(state)
    vscale = torch.sqrt(MU / rt)
    o = torch.stack([
        state[:, 0] / rt, state[:, 1] / rt,
        state[:, 2] / vscale, state[:, 3] / vscale,
        a / rt - 1.0, e, r / rt - 1.0, fuel / DV_BUDGET,
    ], dim=1)
    return o.clamp(-10.0, 10.0)


class Policy(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2), nn.Tanh(),       # action in [-1, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def sample_orbits(batch: int, device, gen: torch.Generator):
    """Random elliptical initial states at random true anomaly (matches env)."""
    def u(lo, hi):
        return lo + (hi - lo) * torch.rand(batch, generator=gen, device=device)
    r_p = R_BODY + u(*ALT_PERI)
    r_a = r_p * u(*RA_RP)
    a = 0.5 * (r_p + r_a)
    e = (r_a - r_p) / (r_a + r_p)
    p = a * (1 - e**2)
    h = torch.sqrt(MU * p)
    nu = u(0.0, 2 * np.pi)
    r = p / (1 + e * torch.cos(nu))
    pf_r = torch.stack([r * torch.cos(nu), r * torch.sin(nu)], dim=1)
    pf_v = torch.stack([(MU / h) * (-torch.sin(nu)),
                        (MU / h) * (e + torch.cos(nu))], dim=1)
    w = u(0.0, 2 * np.pi)
    cw, sw = torch.cos(w), torch.sin(w)
    rot = torch.stack([torch.stack([cw, -sw], 1), torch.stack([sw, cw], 1)], 1)
    pos = torch.einsum("bij,bj->bi", rot, pf_r)
    vel = torch.einsum("bij,bj->bi", rot, pf_v)
    state = torch.cat([pos, vel], dim=1)
    return state, r_a


def rollout_loss(policy, state, rt, horizon, w_dv=0.5, w_crash=5.0, w_prog=0.02):
    """Differentiable episode loss over `horizon` decisions."""
    device = state.device
    fuel = torch.full((state.shape[0],), DV_BUDGET, device=device)
    dv_total = torch.zeros(state.shape[0], device=device)
    crash_pen = torch.zeros(state.shape[0], device=device)
    prog = torch.zeros(state.shape[0], device=device)
    for _ in range(horizon):
        obs = observe(state, rt, fuel.clamp_min(0.0))
        act = policy(obs)                          # [B,2] tangential, radial
        for _ in range(REPEAT):
            v = state[:, 2:]
            r_vec = state[:, :2]
            vmag = torch.linalg.norm(v, dim=1, keepdim=True).clamp_min(1e-6)
            rmag = torch.linalg.norm(r_vec, dim=1, keepdim=True).clamp_min(1e-6)
            t_hat = v / vmag
            r_hat = r_vec / rmag
            thrust = (act[:, :1] * t_hat + act[:, 1:] * r_hat) * THRUST_MAX
            dv_sub = torch.linalg.norm(thrust, dim=1) * DT
            gate = (fuel > 0).float()              # soft fuel gate
            thrust = thrust * gate.unsqueeze(1)
            dv_sub = dv_sub * gate
            fuel = fuel - dv_sub
            dv_total = dv_total + dv_sub
            state = rk4(state, thrust)
            rnow = torch.linalg.norm(state[:, :2], dim=1)
            crash_pen = crash_pen + torch.relu(R_BODY - rnow) ** 2
        a, e, r = elements(state)
        a_err = (a - rt).abs() / rt
        prog = prog + a_err.clamp(max=5.0) + e.clamp(max=2.0)
    a, e, r = elements(state)
    a_err = ((a - rt).abs() / rt).clamp(max=5.0)
    orbit = a_err + e.clamp(max=2.0)
    loss = orbit.mean() + w_dv * dv_total.mean() \
        + w_crash * crash_pen.mean() + w_prog * prog.mean()
    return loss


@torch.no_grad()
def evaluate(policy, device, n_episodes=200, seed=500_000):
    """Run the trained policy on the real Gymnasium env (with termination/gating)."""
    env = Circularize2DEnv(Circularize2DConfig())
    succ, ratios = 0, []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        base = info["baseline_dv"]
        term = trunc = False
        while not (term or trunc):
            a = policy(torch.as_tensor(obs, dtype=torch.float32, device=device)
                       .unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(a)
        if info["success"]:
            succ += 1
            ratios.append(info["dv_used"] / base)
    return succ / n_episodes, (float(np.mean(ratios)) if ratios else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="models/circularize2d_diffsim.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    policy = Policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    for it in range(args.iters):
        state, rt = sample_orbits(args.batch, device, gen)
        loss = rollout_loss(policy, state, rt, args.horizon)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if it % 100 == 0 or it == args.iters - 1:
            s, dvr = evaluate(policy, device, n_episodes=100)
            print(f"iter {it:4d}  loss={float(loss):.4f}  "
                  f"success={s:.2%}  dv_ratio={dvr:.3f}", flush=True)

    torch.save(policy.state_dict(), args.save)
    s, dvr = evaluate(policy, device, n_episodes=200)
    print(f"FINAL: success={s:.2%}  dv_ratio={dvr:.3f}  saved {args.save}")


if __name__ == "__main__":
    main()
