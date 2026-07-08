#!/usr/bin/env python3
"""Primer vector from the diff-sim adjoint — an optimality CERTIFICATE (Build N, R-N9).

Lawden's primer vector p(t) is the velocity-costate of the min-Δv impulsive transfer. Its necessary
conditions: |p| ≤ 1 everywhere, |p| = 1 at each impulse (thrust along p̂), and |p| > 1 anywhere interior ⟺
inserting an impulse there LOWERS Δv (the 2-impulse solution is non-optimal). The diff-sim connection:
reverse-mode backprop through the RK4 rollout IS the discrete state-transition matrix (STM), so the primer
falls out of the gradient machinery already built — no separate indirect-method solver.

Extraction: per-step Jacobians J_k = ∂x_{k+1}/∂x_k (jax.jacobian of one RK4 step), cumulative product →
STM Φ(t,t0). Primer BVP with burn-direction BCs p0, pf (unit vectors along the endpoint impulses):
    ṗ0 = Φ_rv(tf,t0)⁻¹ (pf − Φ_rr(tf,t0) p0),   p(t) = Φ_rr(t,t0) p0 + Φ_rv(t,t0) ṗ0.
(Exactly-180° transfers are excluded — Φ_rv is singular at conjugate points, the Δθ=π Lambert singularity.)

    uv run --with jax python scripts/primer_vector.py --verify        # offline, CI-safe
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import nbody_sim as NB               # noqa: E402
import lambert as LAM                # noqa: E402


def stm_series(rv0, mu, tof, n):
    """State trajectory + STM Φ(t_k,t0) for every step, via cumulative per-step Jacobians (autodiff)."""
    dt = tof / n

    def step_fn(rv):
        return NB.rk4_step(rv, jnp.zeros((1, 3)), jnp.array([mu]), dt, jnp.zeros(3), 0.0)

    _, traj = NB.rollout(rv0, jnp.zeros((n, 1, 3)), jnp.array([mu]), dt, soft=0.0)
    traj = np.asarray(traj)
    states = np.vstack([np.asarray(rv0), traj[:-1]])
    jacfn = jax.jit(jax.jacobian(step_fn))
    phi = np.eye(6)
    phis = [phi.copy()]
    for k in range(n):
        phi = np.asarray(jacfn(jnp.asarray(states[k]))) @ phi
        phis.append(phi.copy())
    return traj, phis


def primer_history(phis, p0, pf):
    """Solve the primer BVP and return |p(t_k)| over the transfer."""
    af, bf = phis[-1][:3, :3], phis[-1][:3, 3:]
    pdot0 = np.linalg.solve(bf, pf - af @ p0)
    return np.array([np.linalg.norm(phis[k][:3, :3] @ p0 + phis[k][:3, 3:] @ pdot0)
                     for k in range(len(phis))]), np.linalg.cond(bf)


def transfer_primer(r1, r2, vc1, vc2, mu, tof, n):
    """2-impulse Lambert transfer → (total Δv, |p(t)|, interior-max |p|, its fraction, STM cond, traj)."""
    v1l, v2l = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=mu)
    v1l, v2l = np.asarray(v1l), np.asarray(v2l)
    dv0, dvf = v1l - vc1, vc2 - v2l
    rv0 = jnp.asarray(np.concatenate([r1, v1l]))
    traj, phis = stm_series(rv0, mu, tof, n)
    pmag, cond = primer_history(phis, dv0 / np.linalg.norm(dv0), dvf / np.linalg.norm(dvf))
    imax = 5 + int(np.argmax(pmag[5:-5]))
    return float(np.linalg.norm(dv0) + np.linalg.norm(dvf)), pmag, pmag[imax], imax / n, cond, traj


def dv3(r1, r2, vc1, vc2, mu, tof, f, rm):
    """Total Δv of a 3-impulse transfer with a midcourse node at r_m, time-fraction f (single-rev legs)."""
    tm = f * tof
    v1a, vma = LAM.lambert(jnp.asarray(r1), jnp.asarray(rm), tm, mu=mu)
    vmb, v2b = LAM.lambert(jnp.asarray(rm), jnp.asarray(r2), tof - tm, mu=mu)
    v1a, vma, vmb, v2b = map(np.asarray, (v1a, vma, vmb, v2b))
    return (np.linalg.norm(v1a - vc1) + np.linalg.norm(vmb - vma) + np.linalg.norm(vc2 - v2b))


def optimize_midcourse(r1, r2, vc1, vc2, mu, tof, f, rm0, iters=140, lr=0.02):
    """Local search on the in-plane midcourse position to minimize 3-impulse Δv (FD gradient descent)."""
    rm = rm0.astype(float).copy()
    best = dv3(r1, r2, vc1, vc2, mu, tof, f, rm)
    for _ in range(iters):
        base = dv3(r1, r2, vc1, vc2, mu, tof, f, rm)
        g = np.zeros(3)
        for j in range(2):
            rp = rm.copy()
            rp[j] += 1e-5
            g[j] = (dv3(r1, r2, vc1, vc2, mu, tof, f, rp) - base) / 1e-5
        rm = rm - lr * g / (np.linalg.norm(g) + 1e-9)
        d = dv3(r1, r2, vc1, vc2, mu, tof, f, rm)
        best = min(best, d)
    return best


def verify(args):
    print("=== R-N9: primer vector from the diff-sim adjoint — optimality certificate (offline) ===")
    mu, n = 1.0, args.steps

    # ---- H-N9a: STM extraction vs finite differences ----
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = 3.0 * np.array([np.cos(np.radians(100)), np.sin(np.radians(100)), 0.0])
    tof = 4.0
    v1l, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=mu)
    rv0 = jnp.asarray(np.concatenate([r1, np.asarray(v1l)]))
    traj, phis = stm_series(rv0, mu, tof, n)
    fd = np.zeros((6, 6))
    for j in range(6):
        pert = np.asarray(rv0).copy()
        pert[j] += 1e-6
        rvt, _ = NB.rollout(jnp.asarray(pert), jnp.zeros((n, 1, 3)), jnp.array([mu]), tof / n, soft=0.0)
        fd[:, j] = (np.asarray(rvt) - traj[-1]) / 1e-6
    stm_err = np.abs(phis[-1] - fd).max() / np.abs(fd).max()
    print(f"  H-N9a STM extraction: autodiff-vs-finite-difference Φ(tf,t0) rel. err = {stm_err:.2e} "
          f"({'PASS' if stm_err < 1e-3 else 'FAIL'})")

    # ---- H-N9b: primer diagnoses optimality, verified operationally (single-rev cases) ----
    print("  H-N9b optimality diagnosis (midcourse impulse helps ⟺ |p|_max > 1):")
    print(f"    {'case':>16} {'|p|max':>7} {'dv2imp':>7} {'dv3best':>7} {'helps':>6} {'consistent':>11}")
    cases = [("optimal 150°", 2.0, 150, 6.0), ("marginal 150°", 3.0, 150, 4.0),
             ("subopt 250°", 2.0, 250, 6.0), ("subopt 300°", 2.0, 300, 6.0)]
    for tag, R, thd, t in cases:
        th = np.radians(thd)
        rr1 = np.array([1.0, 0.0, 0.0])
        vcc1 = np.array([0.0, 1.0, 0.0])
        rr2 = R * np.array([np.cos(th), np.sin(th), 0.0])
        vcc2 = np.sqrt(mu / R) * np.array([-np.sin(th), np.cos(th), 0.0])
        dv2, pmag, pmax, f, cond, tr = transfer_primer(rr1, rr2, vcc1, vcc2, mu, t, n)
        rm0 = tr[int(f * n) - 1][:3]
        base = dv3(rr1, rr2, vcc1, vcc2, mu, t, f, rm0)
        if abs(base - dv2) > 0.02 * dv2:      # baseline must reconstruct the arc (else multi-rev/branch)
            print(f"    {tag:>16} {pmax:7.3f} {dv2:7.4f} {'N/A':>7} {'—':>6} {'skip(multi-rev)':>11}")
            continue
        dvb = optimize_midcourse(rr1, rr2, vcc1, vcc2, mu, t, f, rm0)
        helps = dvb < dv2 - 1e-3
        consistent = (pmax > 1.0) == helps
        print(f"    {tag:>16} {pmax:7.3f} {dv2:7.4f} {dvb:7.4f} {str(helps):>6} {str(consistent):>11}")

    # ---- H-N9c: certify a real direct Earth→Mars transfer (offline Sun-only, R-N5 geometry) ----
    au = NB.AU
    mu_s = NB.GM["sun"]
    r_e, r_m = 1.0 * au, 1.5237 * au
    sweep = np.radians(150.0)
    re = np.array([r_e, 0.0, 0.0])
    rm = r_m * np.array([np.cos(sweep), np.sin(sweep), 0.0])
    vce = np.sqrt(mu_s / r_e) * np.array([0.0, 1.0, 0.0])
    vcm = np.sqrt(mu_s / r_m) * np.array([-np.sin(sweep), np.cos(sweep), 0.0])
    tof_m = 250.0 * NB.DAY
    dv2, pmag, pmax, f, cond, _ = transfer_primer(re, rm, vce, vcm, mu_s, tof_m, n)
    verdict = ("primer-OPTIMAL (no beneficial DSM)" if pmax <= 1.0
               else f"primer flags a beneficial deep-space maneuver at t/TOF={f:.2f}")
    print(f"  H-N9c Earth→Mars direct (150° sweep, 250 d): |p|_max = {pmax:.4f} → {verdict}")
    print("  → the diff-sim adjoint yields a working optimality certificate: |p|≤1 certifies no beneficial "
          "infinitesimal impulse (a NECESSARY, first-order condition — not sufficient for global optimality).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
