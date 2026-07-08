#!/usr/bin/env python3
"""Differentiable N-body engine on real JPL ephemerides (Build N, R-N1).

The locked Tier-3 dynamics: a spacecraft under K point-mass bodies (Sun, Earth, Moon,
planets) whose positions come from real JPL Horizons ephemerides (`ephemeris.py`). The
bodies are an EXOGENOUS time series — their motion does not depend on the spacecraft —
so they are pre-sampled onto the integration grid and held constant within each step;
the JAX rollout is differentiable in the SPACECRAFT trajectory w.r.t. its controls.

R-N1 verifies the ENGINE with NO network (CI-safe): the two-body Kepler limit (energy
conserved, orbit closes after one period) and finite gradients of a terminal miss w.r.t.
a departure Δv. The real-ephemeris propagation is gated behind --fetch.

    uv run --with jax python scripts/nbody_sim.py --verify           # analytic, offline
    uv run --with "jax" python scripts/nbody_sim.py --fetch          # real Sun+Earth+Moon
"""
from __future__ import annotations

import argparse

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

# GM [km^3/s^2] — IAU 2015 / JPL DE440 nominal values.
GM = {
    "sun": 1.32712440018e11,
    "mercury": 2.2032e4, "venus": 3.24859e5, "earth": 3.986004418e5,
    "moon": 4.9028000661e3, "mars": 4.282837e4, "jupiter": 1.26686534e8,
    "saturn": 3.7931187e7, "uranus": 5.793939e6, "neptune": 6.836529e6,
}
HORIZONS_ID = {
    "sun": "10", "mercury": "199", "venus": "299", "earth": "399",
    "moon": "301", "mars": "499", "jupiter": "599", "saturn": "699",
    "uranus": "799", "neptune": "899",
}
AU = 1.495978707e8            # km
DAY = 86400.0


# ---------- dynamics (JAX; bodies exogenous) ----------
def accel(r_sc, body_pos, body_gm, soft=1.0):
    """Point-mass N-body acceleration on the spacecraft (km/s^2).
    r_sc: (3,); body_pos: (K,3) body positions this step; body_gm: (K,). soft: softening
    length (km) to keep the gradient finite through a body flyby (Plummer)."""
    d = body_pos - r_sc                                   # (K,3) sc→body
    r2 = jnp.sum(d * d, axis=1) + soft ** 2               # (K,)
    return jnp.sum((body_gm / r2 ** 1.5)[:, None] * d, axis=0)


def rk4_step(rv, body_pos, body_gm, dt, thrust, soft):
    def deriv(s):
        return jnp.concatenate([s[3:], accel(s[:3], body_pos, body_gm, soft) + thrust])
    k1 = deriv(rv); k2 = deriv(rv + 0.5 * dt * k1)
    k3 = deriv(rv + 0.5 * dt * k2); k4 = deriv(rv + dt * k3)
    return rv + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(rv0, body_seq, body_gm, dt, thrust_seq=None, soft=1.0):
    """Propagate the spacecraft over the pre-sampled body positions body_seq (n,K,3).
    thrust_seq: (n,3) per-step applied acceleration (km/s^2), or None. Bodies are held
    constant within each RK4 step (valid when dt ≪ body-motion timescale)."""
    n = body_seq.shape[0]
    if thrust_seq is None:
        thrust_seq = jnp.zeros((n, 3))

    def step(rv, inp):
        bpos, thr = inp
        rv2 = rk4_step(rv, bpos, body_gm, dt, thr, soft)
        return rv2, rv2
    rvT, traj = lax.scan(step, rv0, (body_seq, thrust_seq))
    return rvT, traj


def energy(rv, body_pos, body_gm, soft=1.0):
    """Specific orbital energy in the body frame: ½v² − Σ GM/|r−r_body| (km²/s²)."""
    d = body_pos - rv[:3]
    dist = jnp.sqrt(jnp.sum(d * d, axis=1) + soft ** 2)
    return 0.5 * jnp.sum(rv[3:] ** 2) - jnp.sum(body_gm / dist)


# ---------- R-N1 verification (analytic, no network) ----------
def verify(args):
    print("=== R-N1: differentiable N-body engine verification (offline) ===")
    mu = GM["sun"]
    a = 1.0 * AU                                   # circular heliocentric orbit at 1 AU
    v_c = np.sqrt(mu / a)
    period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    rv0 = jnp.array([a, 0.0, 0.0, 0.0, v_c, 0.0])
    # one body (Sun) fixed at the origin for the whole span → pure two-body Kepler
    n = args.steps
    dt = period / n
    body_seq = jnp.zeros((n, 1, 3))                # Sun at origin every step
    gm = jnp.array([mu])
    soft = 0.0                                     # exact point mass for the clean test

    rvT, traj = rollout(rv0, body_seq, gm, dt, soft=soft)
    # (1) orbit closes after one period
    pos_err = float(jnp.linalg.norm(rvT[:3] - rv0[:3])) / a
    vel_err = float(jnp.linalg.norm(rvT[3:] - rv0[3:])) / v_c
    print(f"  Kepler closure after 1 period ({period/DAY:.1f} d, {n} steps): "
          f"|Δr|/a={pos_err:.2e}, |Δv|/v_c={vel_err:.2e}")
    # (2) energy conservation along the orbit
    E = jax.vmap(lambda s: energy(s, jnp.zeros((1, 3)), gm, soft))(traj)
    E0 = float(energy(rv0, jnp.zeros((1, 3)), gm, soft))
    drift = float(jnp.max(jnp.abs(E - E0))) / abs(E0)
    print(f"  energy: E0={E0:.4g} km²/s², max relative drift over orbit={drift:.2e} "
          f"(expected −μ/2a={-mu/(2*a):.4g})")
    # (3) Kepler third law sanity: v_c matches √(μ/a)
    print(f"  circular speed √(μ/a)={v_c:.4f} km/s (Earth ~29.78) ✓")
    # (4) differentiability: gradient of terminal miss w.r.t. a departure Δv
    target = np.array([a * np.cos(1.0), a * np.sin(1.0), 0.0])   # 1 rad downrange on ring

    def miss(dv):
        rv = rv0.at[3:].add(dv)
        rvT2, _ = rollout(rv, body_seq[: n // 6], gm, dt, soft=soft)
        return jnp.sum((rvT2[:3] - target) ** 2)
    g = jax.grad(miss)(jnp.zeros(3))
    gn = float(jnp.linalg.norm(g))
    print(f"  ∂(miss)/∂(departure Δv): |grad|={gn:.4e}  finite={np.isfinite(gn)}")
    ok = pos_err < 1e-3 and drift < 1e-4 and np.isfinite(gn) and gn > 0
    print(f"  → engine {'VERIFIED' if ok else 'FAILED'} "
          f"(Kepler closure + energy conservation + differentiable)")


# ---------- real-ephemeris propagation (network; not in CI) ----------
def sample_bodies(bodies, start_iso, days, dt, location="@sun"):
    """Pre-sample body positions (n,K,3) km on the integration grid from cached Horizons."""
    import ephemeris as E
    from astropy.time import Time, TimeDelta
    t0 = Time(start_iso, format="isot", scale="utc")
    stop_iso = str((t0 + TimeDelta(days * DAY, format="sec")).isot)
    n = int(round(days * DAY / dt))
    jds = float(t0.jd) + (np.arange(n) * dt) / DAY
    cols = []
    for b in bodies:
        spec = E.SpanSpec(HORIZONS_ID[b], start_iso, stop_iso,
                          step_s=max(3600, int(dt)), location=location)
        eph = E.get_span(spec)
        cols.append(np.array([eph.state_at_jd(float(jd))[0] for jd in jds]))   # (n,3)
    return np.stack(cols, axis=1), n                                           # (n,K,3)


def fetch_demo(args):
    print("=== R-N1 real-ephemeris demo: heliocentric orbit, Sun+Earth+Moon ===")
    # NOTE on regime: bodies are held constant within each RK4 step, which is valid only
    # when a body moves ≪ the spacecraft's distance to it per step. That holds at
    # HELIOCENTRIC scale (this demo) but BREAKS for a close geocentric orbit (Earth moves
    # ~9000 km per 300 s step, ≫ a 6778 km LEO radius → the orbit desyncs). Geocentric-scale
    # integration needs body positions evaluated at the RK4 substages (continuous), which
    # R-N2 adds when a mission needs it. Here the spacecraft is heliocentric, far from Earth.
    bodies = ["sun", "earth", "moon"]
    gm = jnp.array([GM[b] for b in bodies])
    dt = args.dt
    body_np, n = sample_bodies(bodies, args.start, args.days, dt)
    body_seq = jnp.asarray(body_np)
    # a circular heliocentric orbit at 1 AU under the REAL Sun+Earth+Moon (Sun-dominated,
    # small Earth/Moon perturbations) — should stay bounded near 1 AU.
    v_c = np.sqrt(GM["sun"] / AU)
    rv0 = jnp.asarray(np.array([AU, 0.0, 0.0, 0.0, v_c, 0.0]))
    _, traj = rollout(rv0, body_seq, gm, dt, soft=10.0)
    r_helio = np.linalg.norm(np.asarray(traj)[:, :3], axis=1)
    d_earth = np.linalg.norm(np.asarray(traj)[:, :3] - body_np[:, 1], axis=1)
    print(f"  propagated {args.days:.1f} d ({n} steps, dt={dt:.0f}s): heliocentric range "
          f"{r_helio.min()/AU:.4f}–{r_helio.max()/AU:.4f} AU (should stay ~1.0, bounded); "
          f"closest Earth approach {d_earth.min()/AU:.3f} AU. Sun/Earth/Moon from Horizons.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--start", type=str, default="2024-01-01T00:00:00")
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=60.0)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    if args.fetch:
        fetch_demo(args)


if __name__ == "__main__":
    main()
