#!/usr/bin/env python3
"""Manifold-seeded differentiable-sim POLICY for Earth-Moon capture (Build J).

Return to the PRIMARY project method (the diff-sim policy gradient) in the multi-body
regime. Build G optimized a single departure burn by backprop through the CR3BP and
STALLED (no ballistic capture — chaotic long-arc gradients). Build H built the stable
manifold; I hand-computed a ~5% patched-transfer beat. This module tests: can diff-sim,
SEEDED by the manifold, find a verified ballistic capture where the raw search could not?

Staged:
  R-J1  differentiable 2-burn Earth-Moon env; verify grads + Hohmann reaches the Moon.
  R-J2  raw diff-sim (Hohmann init) — reproduce/quantify the stall.
  R-J3  manifold-seeded diff-sim — BC/warm-start from the manifold reference, refine.

Reuses the VERIFIED JAX CR3BP engine (cr3bp_sim.accel/rk4) + G's capture criterion.

    uv run --with jax python scripts/cr3bp_policy.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402
import cr3bp_transfer as T  # noqa: E402  (leo_state, hohmann_tli, V_UNIT)
import cr3bp_capture as Cap  # noqa: E402  (moon_rel, R_HILL)

MU = C.MU
V_UNIT = T.V_UNIT_KMS
R_HILL = (MU / 3.0) ** (1.0 / 3.0)


def rollout2(params, s0, n, dt, mc_step):
    """Two planar burns: departure dv0 at t=0, mid-course dv1 at step mc_step.
    params = [dv0x, dv0y, dv1x, dv1y]. Returns the (n,6) trajectory."""
    dv0 = params[0:2]
    dv1 = params[2:4]
    s = s0.at[3:5].add(dv0)

    def step(carry, i):
        s = carry
        s = lax.cond(i == mc_step, lambda z: z.at[3:5].add(dv1), lambda z: z, s)
        s = C.rk4(s, dt)
        return s, s

    _, traj = lax.scan(step, s, jnp.arange(n))
    return traj


def capture_metrics(traj, tau=0.005):
    """Softmin-weighted closest lunar approach: distance, Moon-relative speed, energy."""
    d, speed, E = Cap.moon_rel(traj)
    w = jax.nn.softmax(-d / tau)
    d_ca = (w * d).sum()
    speed_ca = (w * speed).sum()
    E_ca = (w * E).sum()
    return d_ca, speed_ca, E_ca


def objective(params, s0, n, dt, mc_step, w_v, w_reach, box, _unused):
    """Capture = reach the Moon AND arrive SLOW (low Moon-relative speed → E<0). Rewarding
    approach speed (not distance) avoids the deep-fast-plunge reward-hack and pushes
    toward the low-energy gateway approach; the mid-course burn pays to slow down."""
    traj = rollout2(params, s0, n, dt, mc_step)
    d_ca, speed_ca, E_ca = capture_metrics(traj)
    dv = jnp.sqrt((params[0:2] ** 2).sum() + 1e-12) + jnp.sqrt((params[2:4] ** 2).sum() + 1e-12)
    reach = w_reach * jnp.maximum(d_ca - box, 0.0)      # get near the Moon
    slow = w_v * speed_ca                               # arrive slow → captured
    return dv + reach + slow, (dv, d_ca, E_ca)


def true_closest(dv0, s0, n, dt):
    """True (argmin) closest lunar distance and its step for a departure-only rollout."""
    traj = rollout2(jnp.concatenate([dv0, jnp.zeros(2)]), s0, n, dt, -1)
    d, _, _ = Cap.moon_rel(traj)
    d = np.asarray(d)
    j = int(np.argmin(d))
    return float(d[j]), j


def adam(p, grad_fn, iters, lr):
    m = jnp.zeros_like(p); v = jnp.zeros_like(p); t = 0
    for _ in range(iters):
        (val, aux), g = grad_fn(p)
        gn = jnp.sqrt((g ** 2).sum())
        g = g * jnp.minimum(1.0, 10.0 / jnp.maximum(gn, 1e-9))   # clip
        t += 1; m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        p = p - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return p


def verify_capture_jax(state, dt, steps):
    """From a closest-approach state, coast (no thrust); revs inside the Hill sphere."""
    s = state; xs = [np.asarray(s)]
    for _ in range(steps):
        s = C.rk4(s, dt); xs.append(np.asarray(s))
    xs = np.array(xs)
    rmoon = np.array([1.0 - MU, 0.0, 0.0])
    d = np.linalg.norm(xs[:, 0:3] - rmoon, axis=1)
    out = np.where(d > R_HILL)[0]
    left = out[0] if len(out) else len(d)
    ang = np.unwrap(np.arctan2(xs[:left, 1] - rmoon[1], xs[:left, 0] - rmoon[0]))
    revs = abs(ang[-1] - ang[0]) / (2 * np.pi) if left > 1 else 0.0
    return left * dt, revs


def optimize(args, mode):
    s0 = T.leo_state(args.r_leo)
    tli = T.hohmann_tli(args.r_leo)
    n = args.n
    if mode == "raw":
        p = jnp.array([0.0, tli, 0.0, 0.0])            # naive Hohmann-direction init (G-like)
        mc = n // 2
    else:                                              # manifold/scan-informed seed:
        # find (by TRUE-distance mini-scan) a departure that actually reaches the Moon,
        # then a retrograde insertion burn timed at that closest approach — the
        # manifold/I 2-burn structure (depart-to-Moon + slow-down-to-capture).
        best = None
        for an in np.linspace(0, 2 * np.pi, 72, endpoint=False):
            dv0c = jnp.array([tli * np.cos(an), tli * np.sin(an)])
            dmin, j = true_closest(dv0c, s0, n, args.dt)
            if best is None or dmin < best[0]:
                best = (dmin, j, dv0c)
        _, mc, dv0 = best
        traj0 = rollout2(jnp.concatenate([dv0, jnp.zeros(2)]), s0, n, args.dt, -1)
        v_mc = np.asarray(traj0[mc, 3:5])
        dv1 = jnp.asarray(-args.seed_ins * v_mc / (np.linalg.norm(v_mc) + 1e-9))
        p = jnp.concatenate([dv0, dv1])
        print(f"       seed: departure reaches {best[0]*C.L_UNIT_KM:.0f} km at step {mc}")
    grad_fn = jax.jit(jax.value_and_grad(
        lambda q: objective(q, s0, n, args.dt, mc, args.w_v, args.w_reach, args.box, 0.0),
        has_aux=True))
    p = adam(p, grad_fn, args.iters, args.lr)
    (val, (dv, d_ca, E_ca)), _ = grad_fn(p)
    # verify at the true closest-approach state of the optimized trajectory
    traj = rollout2(p, s0, n, args.dt, mc)
    d, _, _ = Cap.moon_rel(traj)
    j = int(np.argmin(np.asarray(d)))
    bt, revs = verify_capture_jax(jnp.asarray(traj[j]), args.dt, args.verify_steps)
    dvk = float(dv) * V_UNIT
    print(f"  [{mode}] Δv={float(dv):.4f} ({dvk:.3f} km/s)  closest={float(d[j])*C.L_UNIT_KM:.0f} km  "
          f"E_moon={float(E_ca):+.4f}  mc_step={mc}")
    print(f"       bounded-prop: {revs:.1f} Moon revs, ~{bt*C.T_UNIT_S/86400:.1f} d  → "
          f"{'VERIFIED capture' if revs>=2 else 'no verified capture (flyby/hyperbolic)'}")
    return dict(mode=mode, dv=float(dv), dvk=dvk, d_ca=float(d[j]), E=float(E_ca), revs=revs)


def verify(args):
    print("=== R-J1: differentiable 2-burn Earth-Moon env — verification ===")
    s0 = T.leo_state(args.r_leo)
    tli = T.hohmann_tli(args.r_leo)
    n = args.n
    mc = n // 2
    # Hohmann departure, no mid-course burn
    p0 = jnp.array([0.0, tli, 0.0, 0.0])
    val_grad = jax.jit(jax.value_and_grad(
        lambda p: objective(p, s0, n, args.dt, mc, args.w_v, args.w_reach, args.box, 0.0),
        has_aux=True))
    (val, (dv, d_ca, E_ca)), g = val_grad(p0)
    gfin = bool(np.all(np.isfinite(np.asarray(g))))
    print(f"  Hohmann init (TLI={tli:.4f}={tli*V_UNIT:.3f} km/s), n={n}, dt={args.dt}:")
    print(f"    loss={float(val):.4f}  Δv={float(dv):.4f} ({float(dv)*V_UNIT:.3f} km/s)  "
          f"closest={float(d_ca)*C.L_UNIT_KM:.0f} km  E_moon={float(E_ca):+.4f}")
    print(f"    gradient finite: {gfin}  |grad|={float(jnp.sqrt((g**2).sum())):.3e}")
    print(f"    → engine reuse OK; rollout differentiable; "
          f"{'reaches Moon vicinity' if float(d_ca)<0.1 else 'does NOT reach Moon (tune n/dt)'}.")


def scan(args):
    """Solvability + raw single-burn baseline: sweep departure magnitude×angle, report
    the closest lunar approach and best Moon-energy any single departure burn achieves."""
    print("=== R-J1b: departure-burn scan (env solvability + raw baseline) ===")
    s0 = T.leo_state(args.r_leo)
    tli = T.hohmann_tli(args.r_leo)
    n, mc = args.n, args.n // 2
    roll = jax.jit(lambda p: capture_metrics(rollout2(p, s0, n, args.dt, mc)))
    best = None
    mags = np.linspace(0.85 * tli, 1.15 * tli, 13)
    angs = np.linspace(0.0, 2.0 * np.pi, 49, endpoint=False)
    for mg in mags:
        for an in angs:
            p = jnp.array([mg * np.cos(an), mg * np.sin(an), 0.0, 0.0])
            d_ca, sp_ca, E_ca = (float(x) for x in roll(p))
            if best is None or d_ca < best[0]:
                best = (d_ca, E_ca, mg, an)
    d_ca, E_ca, mg, an = best
    print(f"  best single departure: |Δv|={mg:.4f} ({mg*V_UNIT:.3f} km/s) ang={np.degrees(an):.0f}°")
    print(f"    closest {d_ca*C.L_UNIT_KM:.0f} km, E_moon={E_ca:+.4f} "
          f"({'BOUND' if E_ca<0 else 'hyperbolic'})")
    print(f"    → env is {'SOLVABLE (a burn reaches the Moon)' if d_ca<0.05 else 'not reaching Moon in this grid'}; "
          f"single-burn capture: {'YES' if (E_ca<0 and d_ca<R_HILL) else 'NO (matches G — needs 2 burns/manifold)'}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--optimize", action="store_true", help="R-J2/J3: raw vs seeded diff-sim")
    ap.add_argument("--r-leo", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--w-v", type=float, default=3.0)     # penalty on Moon-relative approach speed
    ap.add_argument("--w-reach", type=float, default=20.0)
    ap.add_argument("--box", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed-ang", type=float, default=220.0)   # scan-best departure angle
    ap.add_argument("--seed-ins", type=float, default=0.5)     # insertion-burn seed magnitude
    ap.add_argument("--verify-steps", type=int, default=8000)
    args = ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.verify:
        verify(args)
    if args.scan:
        scan(args)
    if args.optimize:
        print("=== R-J2/J3: raw vs manifold-seeded diff-sim (2-burn capture) ===")
        optimize(args, "raw")
        optimize(args, "seeded")


if __name__ == "__main__":
    main()
