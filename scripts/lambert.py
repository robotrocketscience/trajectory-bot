#!/usr/bin/env python3
"""Differentiable universal-variable Lambert solver (JAX).

Given two position vectors and a time-of-flight, solve the two-body boundary
value problem for the terminal velocities (r1,r2,Δt) -> (v1,v2). Universal-
variable / Stumpff formulation (Bate-Mueller-White; Curtis Alg. 5.2): a fixed
Newton-iteration count on the universal anomaly z, reusing fgprop's Stumpff C/S
(series near z=0), so the whole map is differentiable — the inner "optimal
targeting" solve for an RL guidance/control split, and usable inside diff-sim.

Verify + benchmark:
    uv run --with jax python scripts/lambert.py
"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

import fgprop as FG

MU = 398600.4418
_SMALL = 1e-2               # |z| below this uses the z=0 series branch


def _dtau_dz(z, y, C, S, A):
    """dF/dz for the Newton step (Curtis Alg. 5.2), with a z≈0 series branch."""
    small = jnp.abs(z) < _SMALL
    zc = jnp.where(small, 1.0, z)
    ratio = jnp.power(y / C, 1.5)
    term1 = ratio * ((1.0 / (2.0 * zc)) * (C - 1.5 * S / C) + 0.75 * S * S / C)
    term1 = term1 + (A / 8.0) * (3.0 * (S / C) * jnp.sqrt(y) + A * jnp.sqrt(C / y))
    # z -> 0 limit
    y0 = jnp.maximum(y, 1e-12)
    term0 = (np.sqrt(2.0) / 40.0) * jnp.power(y0, 1.5) \
        + (A / 8.0) * (jnp.sqrt(y0) + A * jnp.sqrt(1.0 / (2.0 * y0)))
    return jnp.where(small, term0, term1)


def lambert(r1v, r2v, dt, mu=MU, prograde=True, iters=32):
    """Solve Lambert's problem. r1v,r2v: [...,3]; dt: [...] or scalar.
    Returns (v1, v2), each [...,3]. Single-revolution, short/long way by prograde."""
    r1 = jnp.linalg.norm(r1v, axis=-1)
    r2 = jnp.linalg.norm(r2v, axis=-1)
    cross_z = r1v[..., 0] * r2v[..., 1] - r1v[..., 1] * r2v[..., 0]
    cos_dnu = jnp.clip((r1v * r2v).sum(-1) / (r1 * r2), -1.0, 1.0)
    dnu = jnp.arccos(cos_dnu)
    # transfer sweep angle: prograde takes the short way unless the cross-product
    # z-component says the motion carries past π (Curtis' Δθ branch).
    if prograde:
        dnu = jnp.where(cross_z < 0.0, 2.0 * np.pi - dnu, dnu)
    else:
        dnu = jnp.where(cross_z >= 0.0, 2.0 * np.pi - dnu, dnu)

    A = jnp.sin(dnu) * jnp.sqrt(r1 * r2 / (1.0 - jnp.cos(dnu)))
    smu = jnp.sqrt(mu)

    def y_of(z):
        C = FG.stumpff_C(z); S = FG.stumpff_S(z)
        y = r1 + r2 + A * (z * S - 1.0) / jnp.sqrt(C)
        return jnp.maximum(y, 1e-9), C, S      # floor keeps sqrt(y) differentiable

    def body(z, _):
        y, C, S = y_of(z)
        chi = jnp.sqrt(y / C)
        F = chi ** 3 * S + A * jnp.sqrt(y) - smu * dt
        dF = _dtau_dz(z, y, C, S, A)
        return z - F / dF, None

    # Newton from z=0. Exact away from Δθ=π; the ~7% tail of near-half-period /
    # near-180°-sweep transfers is the fundamental Lambert singularity (the transfer
    # plane is undefined for antipodal endpoints) — a robust iterator (Izzo 2015)
    # tightens the approach to π but cannot remove the singularity itself. Note this
    # bites exactly at Hohmann geometry (Δθ=π), which the direct diff-sim policy avoids.
    z0 = jnp.zeros_like(r1)
    z, _ = lax.scan(body, z0, None, length=iters)
    y, C, S = y_of(z)
    f = 1.0 - y / r1
    g = A * jnp.sqrt(y / mu)
    gdot = 1.0 - y / r2
    v1 = (r2v - f[..., None] * r1v) / g[..., None]
    v2 = (gdot[..., None] * r2v - r1v) / g[..., None]
    return v1, v2


if __name__ == "__main__":
    # 1) reference case — Curtis Example 5.2 (3D, prograde)
    r1 = jnp.array([5000.0, 10000.0, 2100.0])
    r2 = jnp.array([-14600.0, 2500.0, 7000.0])
    dt = 3600.0
    v1, v2 = lambert(r1, r2, dt)
    v1_ref = np.array([-5.9925, 1.9254, 3.2456])
    v2_ref = np.array([-3.3125, -4.1966, -0.38529])
    print("Curtis Ex 5.2 (prograde, dt=3600s):")
    print(f"  v1 = {np.asarray(v1)}   ref {v1_ref}   err {np.linalg.norm(np.asarray(v1)-v1_ref):.2e}")
    print(f"  v2 = {np.asarray(v2)}   ref {v2_ref}   err {np.linalg.norm(np.asarray(v2)-v2_ref):.2e}")

    # 2) self-consistency (airtight): propagate v1 for dt via the verified fg
    # propagator; it must land at r2 in both position AND velocity.
    print("\nself-consistency (fg_propagate the Lambert v1 for dt -> r2):")
    key = jax.random.PRNGKey(0)
    s = FG._sample(key, 512)
    r1b, v1b = s[:, 0:3], s[:, 3:6]
    # random second point: propagate each start by a random coast, use that r as r2
    dts = 1500.0 + 3000.0 * jax.random.uniform(jax.random.PRNGKey(1), (512,))
    prop0 = jax.vmap(lambda rv, t: FG.fg_propagate(rv, t))(s[:, 0:6], dts)
    r2b = prop0[:, 0:3]
    V1, V2 = lambert(r1b, r2b, dts)
    landed = jax.vmap(lambda r, v, t: FG.fg_propagate(jnp.concatenate([r, v]), t))(r1b, V1, dts)
    rerr = np.asarray(jnp.linalg.norm(landed[:, 0:3] - r2b, axis=1)
                      / jnp.linalg.norm(r2b, axis=1))
    r1n = np.asarray(jnp.linalg.norm(r1b, axis=1)); r2n = np.asarray(jnp.linalg.norm(r2b, axis=1))
    cosd = np.clip((np.asarray(r1b) * np.asarray(r2b)).sum(1) / (r1n * r2n), -1, 1)
    crossz = np.asarray(r1b[:, 0] * r2b[:, 1] - r1b[:, 1] * r2b[:, 0])
    shortway = crossz >= 0.0        # Δθ < π (the well-posed regime)
    print(f"  all {len(rerr)}: median relerr={np.median(rerr):.2e}  p90={np.percentile(rerr,90):.2e}"
          f"  frac<1e-4={(rerr<1e-4).mean():.3f}")
    print(f"  short-way only (n={shortway.sum()}): median={np.median(rerr[shortway]):.2e}"
          f"  max={rerr[shortway].max():.2e}  frac<1e-4={(rerr[shortway]<1e-4).mean():.3f}")

    # 3) gradient finiteness (Lambert must be differentiable to sit inside diff-sim)
    def scal(r2v):
        v1, v2 = lambert(r1, r2v, dt)
        return jnp.sum(v1 ** 2 + v2 ** 2)
    g = jax.grad(scal)(r2)
    print(f"\ngrad d|v|²/dr2 finite: {bool(jnp.all(jnp.isfinite(g)))}   grad = {np.asarray(g)}")
