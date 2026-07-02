#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable-simulation policy gradient for Circularize-3D (full attitude).

The 3-D analogue of the winning 2-D method. Batched Torch reimplementation of the
coupled orbit+attitude dynamics — gravity, body-axis thrust, quaternion kinematics,
and Euler's rotational equations — mirroring `tbot.dynamics3d` / `tbot.quaternion`
exactly so the policy transfers to `Circularize3DEnv` for eval. The policy commands
body torque + throttle; the episode loss (final orbit error + Δv + a pointing term
to help the slew + impact penalty) is backpropagated through the whole rollout.

    python scripts/train_diffsim3d.py --iters 800 --batch 256 --horizon 60

Experiment code (excluded from the strict-typed library).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from tbot import orbital as orb
from tbot.envs.circularize3d import Circularize3DConfig, Circularize3DEnv

MU = orb.MU_EARTH
R_BODY = orb.R_EARTH
DT = 10.0
REPEAT = 20
A_THRUST = 5e-3
MAX_RATE = 0.05
RATE_GAIN = 0.1
DV_BUDGET = 2.0
ALT_PERI = (400.0, 800.0)
RA_RP = (1.3, 2.5)
INC_MAX = np.radians(40.0)


# ---- batched torch quaternion + dynamics ----------------------------------
def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=1)


def qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=1)


def qrotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros(v.shape[0], 1, device=v.device)
    qv = torch.cat([z, v], dim=1)
    r = qmul(qmul(q, qv), qconj(q))
    return r[:, 1:]


def qnorm(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-9)


def deriv(state, omega_cmd, throttle):
    r = state[:, 0:3]; v = state[:, 3:6]; q = state[:, 6:10]; w = state[:, 10:13]
    rmag = torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1.0)
    grav = -MU * r / rmag**3
    b_hat = torch.zeros_like(v); b_hat[:, 0] = 1.0
    tdir = qrotate(q, b_hat)
    acc = grav + throttle.unsqueeze(1) * A_THRUST * tdir
    z = torch.zeros(w.shape[0], 1, device=w.device)
    qdot = 0.5 * qmul(q, torch.cat([z, w], dim=1))      # q̇ = ½ q ⊗ [0,ω]
    wdot = RATE_GAIN * (omega_cmd - w)                  # first-order rate tracking
    return torch.cat([v, acc, qdot, wdot], dim=1)


def rk4(state, omega_cmd, throttle):
    k1 = deriv(state, omega_cmd, throttle)
    k2 = deriv(state + 0.5 * DT * k1, omega_cmd, throttle)
    k3 = deriv(state + 0.5 * DT * k2, omega_cmd, throttle)
    k4 = deriv(state + DT * k3, omega_cmd, throttle)
    s = state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    q = qnorm(s[:, 6:10])
    return torch.cat([s[:, 0:6], q, s[:, 10:13]], dim=1)


def elements(state):
    r_vec = state[:, 0:3]; v_vec = state[:, 3:6]
    r = torch.linalg.norm(r_vec, dim=1).clamp_min(1.0)
    v = torch.linalg.norm(v_vec, dim=1)
    energy = 0.5 * v**2 - MU / r
    a = -MU / (2.0 * energy)
    rv = (r_vec * v_vec).sum(dim=1)
    factor = (v**2 - MU / r).unsqueeze(1)
    e_vec = (factor * r_vec - rv.unsqueeze(1) * v_vec) / MU
    e = torch.linalg.norm(e_vec, dim=1)
    return a, e, r


def observe(state, rt, fuel):
    a, e, r = elements(state)
    L = rt; V = torch.sqrt(MU / rt)
    tdir = qrotate(state[:, 6:10],
                   torch.tensor([1.0, 0.0, 0.0], device=state.device)
                   .expand(state.shape[0], 3))
    o = torch.cat([
        state[:, 0:3] / L.unsqueeze(1), state[:, 3:6] / V.unsqueeze(1),
        (a / rt - 1.0).unsqueeze(1), e.unsqueeze(1), (r / rt - 1.0).unsqueeze(1),
        tdir, state[:, 10:13], (fuel / DV_BUDGET).unsqueeze(1),
    ], dim=1)
    return o.clamp(-10.0, 10.0)


class Policy(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


def sample_orbits(batch, device, gen):
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
    pf = torch.stack([r * torch.cos(nu), r * torch.sin(nu), torch.zeros_like(r)], 1)
    pfv = torch.stack([(MU / h) * (-torch.sin(nu)),
                       (MU / h) * (e + torch.cos(nu)), torch.zeros_like(r)], 1)
    # tilt into a random plane: rotate about x by inc, then about z by raan
    inc = u(0.0, INC_MAX); raan = u(0.0, 2 * np.pi)
    ci, si = torch.cos(inc), torch.sin(inc)
    cr, sr = torch.cos(raan), torch.sin(raan)

    def rot(vec):
        # Rx(inc)
        y = vec[:, 1] * ci - vec[:, 2] * si
        zc = vec[:, 1] * si + vec[:, 2] * ci
        x = vec[:, 0]
        # Rz(raan)
        return torch.stack([x * cr - y * sr, x * sr + y * cr, zc], dim=1)

    r_vec = rot(pf); v_vec = rot(pfv)
    q0 = qnorm(torch.randn(batch, 4, generator=gen, device=device))
    w0 = (u(-0.01, 0.01).unsqueeze(1)).repeat(1, 3)
    state = torch.cat([r_vec, v_vec, q0, w0], dim=1)
    return state, r_a


def rollout_loss(policy, state, rt, horizon,
                 w_dv=0.5, w_crash=5.0, w_point=0.5, w_prog=0.02):
    fuel = torch.full((state.shape[0],), DV_BUDGET, device=state.device)
    dv_total = torch.zeros(state.shape[0], device=state.device)
    crash = torch.zeros(state.shape[0], device=state.device)
    prog = torch.zeros(state.shape[0], device=state.device)
    point = torch.zeros(state.shape[0], device=state.device)
    for _ in range(horizon):
        obs = observe(state, rt, fuel.clamp_min(0.0))
        act = policy(obs)
        omega_cmd = act[:, 0:3] * MAX_RATE
        throttle = act[:, 3].clamp(0.0, 1.0)
        for _ in range(REPEAT):
            gate = (fuel > 0).float()
            thr = throttle * gate
            dv_sub = thr * A_THRUST * DT
            fuel = fuel - dv_sub
            dv_total = dv_total + dv_sub
            state = rk4(state, omega_cmd, thr)
            rnow = torch.linalg.norm(state[:, 0:3], dim=1)
            crash = crash + torch.relu(R_BODY - rnow) ** 2
        # pointing: encourage body axis aligned with prograde
        v = state[:, 3:6]
        vmag = torch.linalg.norm(v, dim=1).clamp_min(1e-6)
        tdir = qrotate(state[:, 6:10],
                       torch.tensor([1.0, 0.0, 0.0], device=state.device)
                       .expand(state.shape[0], 3))
        point = point + (1.0 - (tdir * v).sum(1) / vmag)
        a, e, r = elements(state)
        prog = prog + ((a - rt).abs() / rt).clamp(max=5.0) + e.clamp(max=2.0)
    a, e, r = elements(state)
    orbit = ((a - rt).abs() / rt).clamp(max=5.0) + e.clamp(max=2.0)
    return (orbit.mean() + w_dv * dv_total.mean() + w_crash * crash.mean()
            + w_point * point.mean() / horizon + w_prog * prog.mean())


@torch.no_grad()
def evaluate(policy, device, n_episodes=50, seed=500_000):
    env = Circularize3DEnv(Circularize3DConfig())
    succ = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        term = trunc = False
        while not (term or trunc):
            a = policy(torch.as_tensor(obs, dtype=torch.float32, device=device)
                       .unsqueeze(0)).squeeze(0).cpu().numpy()
            obs, _, term, trunc, info = env.step(a)
        succ += int(info["success"])
    return succ / n_episodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="models/circularize3d_diffsim.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    policy = Policy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters,
                                                       eta_min=args.lr * 0.1)
    best = -1.0
    for it in range(args.iters):
        state, rt = sample_orbits(args.batch, device, gen)
        loss = rollout_loss(policy, state, rt, args.horizon)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); sched.step()
        if it % args.eval_every == 0 or it == args.iters - 1:
            s = evaluate(policy, device)
            print(f"iter {it:4d}  loss={float(loss.detach()):.4f}  success={s:.2%}",
                  flush=True)
            if s > best:
                best = s
                torch.save(policy.state_dict(), args.save)
    policy.load_state_dict(torch.load(args.save))
    s = evaluate(policy, device, n_episodes=200)
    print(f"FINAL(best): success={s:.2%}  saved {args.save}")


if __name__ == "__main__":
    main()
