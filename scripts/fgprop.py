#!/usr/bin/env python3
"""Differentiable universal-variable Kepler f&g propagator (JAX).

Analytically advances a two-body state (r,v) by dt in one step (Bate-Mueller-
White / Vallado universal variables + Stumpff functions), replacing RK4
integration of ballistic coast arcs. Fully differentiable: fixed Newton-
iteration count, and Stumpff C/S use a series near z=0 (whose closed forms
have an infinite sqrt-gradient at 0) so grads stay finite everywhere.

Verify + benchmark:
    uv run --with jax python scripts/fgprop.py
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax, jax.numpy as jnp
from jax import lax, random

MU = 398600.4418
_SMALL = 1e-2


def stumpff_C(z):
    small = jnp.abs(z) < _SMALL
    zc = jnp.where(small, 1.0, z)            # safe z for the closed form
    sz = jnp.sqrt(jnp.abs(zc))
    pos = (1.0 - jnp.cos(sz)) / zc
    neg = (jnp.cosh(sz) - 1.0) / (-zc)
    closed = jnp.where(zc > 0, pos, neg)
    ser = 0.5 - z / 24.0 + z * z / 720.0 - z ** 3 / 40320.0
    return jnp.where(small, ser, closed)


def stumpff_S(z):
    small = jnp.abs(z) < _SMALL
    zc = jnp.where(small, 1.0, z)
    sz = jnp.sqrt(jnp.abs(zc))
    pos = (sz - jnp.sin(sz)) / sz ** 3
    neg = (jnp.sinh(sz) - sz) / sz ** 3
    closed = jnp.where(zc > 0, pos, neg)
    ser = 1.0 / 6.0 - z / 120.0 + z * z / 5040.0 - z ** 3 / 362880.0
    return jnp.where(small, ser, closed)


def fg_propagate(rv, dt, mu=MU, iters=8):
    """rv: [..., 6] (r,v stacked). Returns state after dt (same shape)."""
    r0v = rv[..., 0:3]; v0v = rv[..., 3:6]
    r0 = jnp.linalg.norm(r0v, axis=-1)
    v0 = jnp.linalg.norm(v0v, axis=-1)
    vr0 = (r0v * v0v).sum(-1) / r0
    alpha = 2.0 / r0 - v0 * v0 / mu           # 1/a
    smu = jnp.sqrt(mu)
    chi = smu * jnp.abs(alpha) * dt            # elliptic initial guess

    def body(chi, _):
        z = alpha * chi * chi
        C = stumpff_C(z); S = stumpff_S(z)
        r = (chi * chi * C + (vr0 / smu) * chi * (1.0 - z * S) + r0 * (1.0 - z * C))
        F = ((r0 * vr0 / smu) * chi * chi * C + (1.0 - alpha * r0) * chi ** 3 * S
             + r0 * chi - smu * dt)
        return chi - F / r, None

    chi, _ = lax.scan(body, chi, None, length=iters)
    z = alpha * chi * chi
    C = stumpff_C(z); S = stumpff_S(z)
    f = 1.0 - chi * chi / r0 * C
    g = dt - chi ** 3 / smu * S
    rvec = f[..., None] * r0v + g[..., None] * v0v
    rmag = jnp.linalg.norm(rvec, axis=-1)
    fdot = smu / (rmag * r0) * (z * S - 1.0) * chi
    gdot = 1.0 - chi * chi / rmag * C
    vvec = fdot[..., None] * r0v + gdot[..., None] * v0v
    return jnp.concatenate([rvec, vvec], axis=-1)


# ---- reference: pure two-body RK4 coast ------------------------------------
def _rk4_coast(rv, dt_sub, n):
    def deriv(s):
        r = s[..., 0:3]; v = s[..., 3:6]
        rmag = jnp.linalg.norm(r, axis=-1, keepdims=True)
        acc = -MU * r / rmag ** 3
        return jnp.concatenate([v, acc], axis=-1)

    def step(s, _):
        k1 = deriv(s); k2 = deriv(s + 0.5 * dt_sub * k1)
        k3 = deriv(s + 0.5 * dt_sub * k2); k4 = deriv(s + dt_sub * k3)
        return s + (dt_sub / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None

    s, _ = lax.scan(step, rv, None, length=n)
    return s


def _sample(key, B):
    import jaxsim as J
    s, _ = J.sample_orbits(key, B)
    return s[:, 0:6]      # r,v only


if __name__ == "__main__":
    key = random.PRNGKey(0)
    rv0 = _sample(key, 2048)
    print("parity: f&g vs RK4 coast (max relative error in r and v)")
    for secs, nsub in [(200, 20), (2000, 200), (20000, 2000), (60000, 6000)]:
        ref = _rk4_coast(rv0, 10.0, nsub)
        fg = fg_propagate(rv0, float(secs))
        er = jnp.max(jnp.linalg.norm(fg[:, 0:3] - ref[:, 0:3], axis=-1)
                     / jnp.linalg.norm(ref[:, 0:3], axis=-1))
        ev = jnp.max(jnp.linalg.norm(fg[:, 3:6] - ref[:, 3:6], axis=-1)
                     / jnp.linalg.norm(ref[:, 3:6], axis=-1))
        print(f"  dt={secs:6d}s ({nsub:5d} substeps)   r_relerr={float(er):.2e}  v_relerr={float(ev):.2e}")

    # gradient finiteness (grad of a scalar of the propagated state wrt rv0)
    def scal(rv):
        return jnp.sum(fg_propagate(rv, 2000.0)[:, 0:3] ** 2)
    g = jax.grad(scal)(rv0)
    print(f"grad finite: {bool(jnp.all(jnp.isfinite(g)))}   max|grad|={float(jnp.max(jnp.abs(g))):.3e}")
