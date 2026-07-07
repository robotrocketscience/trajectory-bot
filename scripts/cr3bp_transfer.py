#!/usr/bin/env python3
"""Differentiable Earth->Moon departure-burn transfer in the CR3BP (Build F, F3).

The tractable first cut at the Tier-3 discovery question: rather than a full policy
through the chaotic dynamics, optimize the DEPARTURE Δv from a circular LEO — a
low-dimensional, differentiable problem — to bring the ballistic trajectory to the
Moon's vicinity, and ask whether the optimal burn is BELOW the two-body Hohmann
trans-lunar injection (TLI). A sub-Hohmann burn that still reaches the Moon is the
multi-body signature: the Moon's gravity meets you (the low-energy insight), which
patched-conic Hohmann can't see.

Baseline: two-body Hohmann from LEO (r≈0.017 nondim) to the Moon's orbit (r=1):
ΔV_TLI = v_circ(r_leo)·(√(2/(1+r_leo)) − 1) ≈ 3.06 nondim ≈ 3.13 km/s.

    uv run --with jax python scripts/cr3bp_transfer.py --optimize
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402

V_UNIT_KMS = C.L_UNIT_KM / C.T_UNIT_S      # nondim velocity -> km/s (~1.0245)


def leo_state(r_leo):
    """Circular prograde LEO around Earth, expressed in the rotating frame."""
    x = -C.MU + r_leo
    v_in = np.sqrt((1.0 - C.MU) / r_leo)   # inertial circular speed
    v_rot = v_in - (r_leo - C.MU)          # subtract frame rotation ω×r (planar, +y)
    return jnp.array([x, 0.0, 0.0, 0.0, v_rot, 0.0])


def hohmann_tli(r_leo):
    v_c = np.sqrt((1.0 - C.MU) / r_leo)
    return v_c * (np.sqrt(2.0 / (1.0 + r_leo)) - 1.0)     # to r=1 (Moon orbit)


def rollout_traj(s0, dv, n, dt):
    s = s0.at[3:5].add(dv)                 # apply planar departure burn to (vx,vy)
    def step(s, _):
        return C.rk4(s, dt), s
    _, traj = lax.scan(step, s, None, length=n)
    return traj                             # (n, 6)


def moon_dist(traj):
    d = traj[:, 0:3] - jnp.array([1.0 - C.MU, 0.0, 0.0])
    return jnp.sqrt((d ** 2).sum(axis=1))


def objective(dv, s0, n, dt, w_reach):
    traj = rollout_traj(s0, dv, n, dt)
    dmin = jnp.min(moon_dist(traj))        # TRUE closest approach (subgradient through argmin)
    return jnp.sqrt((dv ** 2).sum() + 1e-9) + w_reach * dmin, dmin


def optimize(args):
    r_leo = args.r_leo
    s0 = leo_state(r_leo)
    tli = hohmann_tli(r_leo)
    print(f"CR3BP Earth->Moon departure-burn optimization")
    print(f"  r_leo={r_leo:.4f} nondim  Hohmann TLI={tli:.4f} nondim "
          f"({tli*V_UNIT_KMS:.3f} km/s)  [reach r=1]")
    dv = jnp.array([0.0, tli])             # init at the Hohmann TLI (prograde)
    opt = (jnp.zeros(2), jnp.zeros(2), 0)
    vg = jax.jit(jax.value_and_grad(lambda d: objective(d, s0, args.n, args.dt, args.w_reach),
                                    has_aux=True))
    best_dv = None; best_dmin = 1e9; best_mag = 1e9
    for it in range(args.iters):
        (val, dmin), g = vg(dv)
        gn = jnp.sqrt((g ** 2).sum())
        g = g * jnp.minimum(1.0, 1.0 / jnp.maximum(gn, 1e-9))
        m, v, t = opt; t = t + 1
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
        dv = dv - args.lr * mh / (jnp.sqrt(vh) + 1e-8)
        mag = float(jnp.sqrt((dv ** 2).sum())); dm = float(dmin)
        # track the cheapest burn that still reaches within the Moon capture box
        if dm < args.reach_box and mag < best_mag:
            best_mag = mag; best_dmin = dm; best_dv = np.asarray(dv)
        if (it + 1) % args.eval_every == 0 or it == 0:
            print(f"  it={it:4d} |Δv|={mag:.4f} ({mag*V_UNIT_KMS:.3f} km/s) "
                  f"min-dist-to-Moon={dm:.4f} ({dm*C.L_UNIT_KM:.0f} km)", flush=True)
    print("  ---")
    if best_dv is not None:
        r_moon_km = best_dmin * C.L_UNIT_KM
        print(f"  BEST reaching burn (<{args.reach_box:.3f} of Moon): "
              f"|Δv|={best_mag:.4f} nondim = {best_mag*V_UNIT_KMS:.3f} km/s, "
              f"closest {r_moon_km:.0f} km")
        print(f"  vs Hohmann TLI {tli:.4f} ({tli*V_UNIT_KMS:.3f} km/s): "
              f"ratio {best_mag/tli:.3f}  "
              f"{'BELOW Hohmann (low-energy signature)' if best_mag < tli*0.999 else 'not below Hohmann'}")
    else:
        print(f"  no burn brought the trajectory within {args.reach_box:.3f} of the Moon "
              f"(engine ok, but this init/window did not reach — honest null)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--r-leo", type=float, default=0.03)     # ~11500 km (high LEO/MEO)
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--w-reach", type=float, default=20.0)
    ap.add_argument("--reach-box", type=float, default=0.05)  # ~19000 km of Moon (loose capture)
    ap.add_argument("--eval-every", type=int, default=50)
    args = ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.optimize:
        optimize(args)


if __name__ == "__main__":
    main()
