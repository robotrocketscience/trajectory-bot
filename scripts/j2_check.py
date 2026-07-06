#!/usr/bin/env python3
"""Verify the J2 oblateness term in jaxsim.deriv by its secular nodal regression.

J2's signature is that the ascending node regresses at the analytic secular rate
Ω̇ = -1.5 n J2 (R_E/p)² cos(i). Propagate a coasting orbit with J2 on and fit the
RAAN drift; it must match. Also confirm J2_COEF=0 is bit-exact two-body (no drift,
energy conserved).

    uv run --with jax python scripts/j2_check.py
"""
import sys

sys.path.insert(0, "scripts")
import numpy as np
import jax.numpy as jnp

import jaxsim as J


def raan(state):
    r = np.asarray(state[0, 0:3]); v = np.asarray(state[0, 3:6])
    h = np.cross(r, v)
    return np.arctan2(h[0], -h[1])          # Ω = atan2(h_x, -h_y)


def propagate(a, inc, orbits, j2_on):
    J.J2_COEF = J.J2_EARTH if j2_on else 0.0
    vc = np.sqrt(J.MU / a)
    r0 = np.array([a, 0.0, 0.0])
    v0 = vc * np.array([0.0, np.cos(inc), np.sin(inc)])       # start at ascending node
    state = jnp.array(np.concatenate([r0, v0, [1.0, 0, 0, 0], np.zeros(3)])[None])
    period = 2 * np.pi * np.sqrt(a ** 3 / J.MU)
    nsteps = int(orbits * period / J.DT)
    zc = jnp.zeros((1, 3)); zt = jnp.zeros((1,))
    ts = []; oms = []; e0 = None
    for k in range(nsteps):
        state = J.rk4(state, zc, zt)
        r = np.asarray(state[0, 0:3]); v = np.asarray(state[0, 3:6])
        en = 0.5 * (v @ v) - J.MU / np.linalg.norm(r)
        if e0 is None:
            e0 = en
        ts.append(k * J.DT); oms.append(raan(state))
    om = np.unwrap(np.array(oms))
    slope = np.polyfit(np.array(ts), om, 1)[0]                # rad/s, secular fit
    return slope, (en - e0) / abs(e0)


def main():
    a = J.R_BODY + 500.0        # ~500 km LEO
    inc = np.radians(51.6)      # ISS-like
    n = np.sqrt(J.MU / a ** 3)
    p = a                       # circular
    analytic = -1.5 * n * J.J2_EARTH * (J.R_BODY / p) ** 2 * np.cos(inc)
    print(f"orbit a={a:.0f} km  i=51.6°  (n={n:.3e} rad/s)")

    slope_on, dE_on = propagate(a, inc, orbits=8, j2_on=True)
    slope_off, dE_off = propagate(a, inc, orbits=8, j2_on=False)

    day = 86400.0
    print("J2 ON:")
    print(f"  measured Ω̇ = {slope_on:.4e} rad/s = {np.degrees(slope_on)*day:+.3f} °/day")
    print(f"  analytic Ω̇ = {analytic:.4e} rad/s = {np.degrees(analytic)*day:+.3f} °/day")
    print(f"  rel error  = {abs(slope_on-analytic)/abs(analytic):.2%}")
    print(f"  energy drift over 8 orbits = {dE_on:.2e}")
    print("J2 OFF (must be ~0 drift = pure two-body):")
    print(f"  measured Ω̇ = {np.degrees(slope_off)*day:+.4f} °/day   energy drift = {dE_off:.2e}")


if __name__ == "__main__":
    main()
