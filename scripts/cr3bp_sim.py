#!/usr/bin/env python3
"""Circular Restricted 3-Body Problem (Build F) — a differentiable Earth-Moon testbed.

Tier-3's discovery question: in the multi-body regime Hohmann is NOT optimal —
low-energy / weak-stability-boundary transfers use the Moon's gravity to capture
with less Δv than a patched-conic Hohmann-plus-capture burn (Belbruno; flown by
Hiten). The ROADMAP locks Tier-3 to ephemeris N-body eventually, but the CLEAN,
differentiable first testbed is the CR3BP (an explicitly-sanctioned optional
model). This module is the verified dynamics engine; the transfer/diff-sim search
builds on it.

Rotating-frame nondimensional CR3BP (Earth at (-μ,0,0), Moon at (1-μ,0,0)):
    ẍ = 2ẏ + x - (1-μ)(x+μ)/r1³ - μ(x-1+μ)/r2³
    ÿ = -2ẋ + y - (1-μ)y/r1³   - μy/r2³
    z̈ =            -(1-μ)z/r1³ - μz/r2³
Jacobi constant C = 2Ω - v²,  Ω = (x²+y²)/2 + (1-μ)/r1 + μ/r2  (conserved ballistic).

    uv run --with jax python scripts/cr3bp_sim.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

MU = 0.012150585609624             # Earth-Moon mass parameter
# nondimensional -> physical: length unit = mean Earth-Moon distance, time unit
# such that the mean motion is 1. For reference only (dynamics are nondimensional).
L_UNIT_KM = 384400.0
T_UNIT_S = 375190.0                # ~4.342 days (1/mean-motion)


def _r1r2(r):
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r1 = jnp.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)          # to Earth
    r2 = jnp.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)    # to Moon
    return r1, r2


def accel(state, thrust=None):
    """Rotating-frame acceleration (+ optional thrust [ax,ay,az] nondim)."""
    r = state[..., 0:3]; v = state[..., 3:6]
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    vx, vy = v[..., 0], v[..., 1]
    r1, r2 = _r1r2(r)
    r1c = jnp.maximum(r1, 1e-4) ** 3
    r2c = jnp.maximum(r2, 1e-4) ** 3
    om = 1.0 - MU
    ax = 2.0 * vy + x - om * (x + MU) / r1c - MU * (x - 1.0 + MU) / r2c
    ay = -2.0 * vx + y - om * y / r1c - MU * y / r2c
    az = -om * z / r1c - MU * z / r2c
    a = jnp.stack([ax, ay, az], axis=-1)
    if thrust is not None:
        a = a + thrust
    return a


def deriv(state, thrust=None):
    v = state[..., 3:6]
    a = accel(state, thrust)
    return jnp.concatenate([v, a], axis=-1)


def rk4(state, dt, thrust=None):
    k1 = deriv(state, thrust)
    k2 = deriv(state + 0.5 * dt * k1, thrust)
    k3 = deriv(state + 0.5 * dt * k2, thrust)
    k4 = deriv(state + dt * k3, thrust)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobi(state):
    r = state[..., 0:3]; v = state[..., 3:6]
    x, y = r[..., 0], r[..., 1]
    r1, r2 = _r1r2(r)
    Om = 0.5 * (x ** 2 + y ** 2) + (1.0 - MU) / r1 + MU / r2
    return 2.0 * Om - (v ** 2).sum(axis=-1)


def propagate(state, dt, n, thrust=None):
    def step(s, _):
        return rk4(s, dt, thrust), None
    out, _ = lax.scan(step, state, None, length=n)
    return out


def lagrange_points():
    """Collinear L1,L2,L3 (Newton on Ω_x=0, y=z=0) and triangular L4,L5 (exact)."""
    def omega_x(x):
        r1 = abs(x + MU); r2 = abs(x - 1.0 + MU)
        return x - (1.0 - MU) * (x + MU) / r1 ** 3 - MU * (x - 1.0 + MU) / r2 ** 3

    def newton(x0):
        x = x0
        for _ in range(100):
            f = omega_x(x)
            df = (omega_x(x + 1e-7) - omega_x(x - 1e-7)) / 2e-7
            x = x - f / df
        return x
    L1 = newton(1.0 - MU - 0.15)     # between Earth and Moon
    L2 = newton(1.0 - MU + 0.15)     # beyond Moon
    L3 = newton(-1.0 - MU * 0.5)     # opposite side
    L4 = (0.5 - MU, np.sqrt(3.0) / 2.0)
    L5 = (0.5 - MU, -np.sqrt(3.0) / 2.0)
    return {"L1": L1, "L2": L2, "L3": L3, "L4": L4, "L5": L5}


def verify():
    print("=== CR3BP verification (Earth-Moon, μ=%.6g) ===" % MU)
    lp = lagrange_points()
    # L-point residual: Ω_x should be ~0 at collinear points; L4/L5 exact triangular
    for name in ("L1", "L2", "L3"):
        x = lp[name]
        s = jnp.array([x, 0.0, 0.0, 0.0, 0.0, 0.0])
        a = np.asarray(accel(s))            # at rest in rotating frame -> accel is Ω-gradient
        print(f"  {name}: x={x:+.6f}  |accel|={np.linalg.norm(a):.2e} (want ~0)")
    print(f"  L4: ({lp['L4'][0]:+.4f},{lp['L4'][1]:+.4f})  L5: ({lp['L5'][0]:+.4f},{lp['L5'][1]:+.4f})"
          f"  (exact triangular: x=0.5-μ={0.5-MU:.4f}, y=±√3/2={np.sqrt(3)/2:.4f})")
    # Jacobi constant conservation on a BOUNDED orbit far from the primaries (an L4
    # libration; μ<0.0385 so L4 is linearly stable) — no close approach, no r-clamp,
    # so a correct RK4 must conserve C. (A near-singular plunge would break it via
    # the clamp, not a dynamics error.)
    s0 = jnp.array([0.49, 0.87, 0.0, 0.0, 0.0, 0.0])        # small perturbation off L4
    C0 = float(jacobi(s0))
    sT = propagate(s0, 1e-3, 20000)
    CT = float(jacobi(sT))
    print(f"  Jacobi C (L4 libration): start {C0:.8f} -> after 20k steps {CT:.8f}  "
          f"(drift {abs(CT-C0):.2e}; correct RK4 conserves it)")
    print(f"  final state finite: {bool(np.all(np.isfinite(np.asarray(sT))))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    verify()


if __name__ == "__main__":
    main()
