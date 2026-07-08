#!/usr/bin/env python3
"""Gravity-assist forward-model fidelity — can the R-N5 engine reproduce a flyby? (Build N, R-N6).

Before any "can the agent DISCOVER an assist" claim, one assumption is load-bearing: the R-N5 engine
integrates with a FIXED RK4 step (dt_sub ≈ 14 h on a real Mars transfer) and Plummer softening. A gravity
assist is a close hyperbolic approach resolved over HOURS. If the fixed step steps over the close approach,
the engine cannot carry an assist and "discovery" is meaningless. This round characterizes the (softening,
step-size) regime in which the engine's velocity deflection matches the EXACT hyperbolic turning angle.

Rig (offline, CI-safe, no ephemeris, no optimizer): a two-body flyby — one fixed massive body (Jupiter μ),
a spacecraft on an EXACT hyperbola with chosen (V∞, periapsis r_p). Ground truth from the two-body
invariants, which the true dynamics conserve exactly:
    specific energy   ε = v²/2 − μ/r          → V∞ = √(2ε)
    eccentricity vec  e = ((v²−μ/r)r − (r·v)v)/μ
    turn (deflection) δ = 2·arcsin(1/|e|)      (asymptotic velocity rotation)
A perfect integrator conserves ε and e, so the deflection reconstructed from the INTEGRATED end state
equals the analytic δ. A step too coarse near periapsis drifts ε and e → the reconstructed δ, V∞ are wrong.
That drift IS the fidelity error.

    uv run --with jax python scripts/nbody_flyby.py --verify    # offline, prints the (soft,dt) regime table
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

MU_J = NB.GM["jupiter"]              # a representative massive perturber (km^3/s^2)


# ---------- exact hyperbola helpers (planet fixed at origin, orbit in z=0 plane) ----------
def hyperbola_state(mu, vinf, r_p, nu):
    """Exact state (r_vec, v_vec) on the flyby hyperbola at true anomaly nu (rad).
    Perifocal frame == inertial (planet fixed, periapsis along +x)."""
    e = 1.0 + r_p * vinf ** 2 / mu               # eccentricity (r_p = a(e-1), a = -mu/vinf^2)
    p = r_p * (1.0 + e)                           # semi-latus rectum
    r = p / (1.0 + e * np.cos(nu))
    r_vec = r * np.array([np.cos(nu), np.sin(nu), 0.0])
    s = np.sqrt(mu / p)
    v_vec = s * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    return r_vec, v_vec, e, p


def tof_symmetric(mu, vinf, e, nu_start):
    """Time of flight from nu=-nu_start (incoming) to nu=+nu_start (outgoing) on the hyperbola."""
    # hyperbolic anomaly F from true anomaly, then Kepler's eqn M = e sinh F - F
    F = 2.0 * np.arctanh(np.sqrt((e - 1.0) / (e + 1.0)) * np.tan(nu_start / 2.0))
    M = e * np.sinh(F) - F
    n = vinf ** 3 / mu                            # mean motion = sqrt(mu/|a|^3), |a| = mu/vinf^2
    return 2.0 * M / n                            # symmetric: t(-nu_start)->t(+nu_start)


def elements_from_state(mu, r_vec, v_vec):
    """Reconstruct (V∞, eccentricity, deflection δ) from a state via the two-body invariants."""
    r = np.linalg.norm(r_vec)
    v2 = float(v_vec @ v_vec)
    eps = 0.5 * v2 - mu / r                       # specific energy
    vinf = np.sqrt(max(2.0 * eps, 0.0))
    e_vec = ((v2 - mu / r) * r_vec - float(r_vec @ v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    delta = 2.0 * np.arcsin(1.0 / e) if e > 1.0 else np.nan
    return vinf, e, delta, eps


def analytic(mu, vinf, r_p):
    e = 1.0 + r_p * vinf ** 2 / mu
    delta = 2.0 * np.arcsin(1.0 / e)
    v_peri = np.sqrt(vinf ** 2 + 2.0 * mu / r_p)
    dv_helio = 2.0 * vinf * np.sin(delta / 2.0)   # heliocentric-frame speed change from the assist
    return e, delta, v_peri, dv_helio


# ---------- integrate one flyby with the R-N5 engine ----------
def integrate_flyby(mu, vinf, r_p, dt, soft, nu_frac=0.99):
    """Integrate the exact hyperbola from -nu_start to +nu_start with NB.rollout (fixed dt, softening).
    Returns (measured deflection δ, V∞ from end-state invariants, ε drift, raw velocity-turn angle, nsteps).
    """
    e = 1.0 + r_p * vinf ** 2 / mu
    nu_inf = np.arccos(-1.0 / e)
    nu_start = nu_frac * nu_inf
    r0, v0, _, _ = hyperbola_state(mu, vinf, r_p, -nu_start)
    tof = tof_symmetric(mu, vinf, e, nu_start)
    n = max(2, int(round(tof / dt)))
    rv0 = jnp.asarray(np.concatenate([r0, v0]))
    body_seq = jnp.zeros((n, 1, 3))               # single body fixed at origin every step
    gm = jnp.array([mu])
    rvT, _ = NB.rollout(rv0, body_seq, gm, tof / n, soft=soft)
    rvT = np.asarray(rvT)
    vinf_end, e_end, delta_end, eps_end = elements_from_state(mu, rvT[:3], rvT[3:])
    _, _, _, eps0 = elements_from_state(mu, r0, v0)
    eps_drift = abs(eps_end - eps0) / abs(eps0)
    # raw physical turn: angle between start and end velocity vectors (approaches δ as nu_frac->1)
    cang = float(v0 @ rvT[3:]) / (np.linalg.norm(v0) * np.linalg.norm(rvT[3:]))
    raw_turn = np.arccos(np.clip(cang, -1.0, 1.0))
    return delta_end, vinf_end, eps_drift, raw_turn, n


def verify(args):
    mu, vinf = MU_J, args.vinf
    print("=== R-N6: gravity-assist forward-model fidelity — two-body flyby vs exact hyperbola ===")
    print(f"  body μ = {mu:.4e} km³/s² (Jupiter), V∞ = {vinf:.1f} km/s\n")

    # two flyby geometries: a WIDE graze (forgiving) and a TIGHT pass (demanding)
    for tag, r_p in (("wide", args.rp_wide), ("tight", args.rp_tight)):
        e, delta, v_peri, dv_helio = analytic(mu, vinf, r_p)
        t_peri = r_p / v_peri                       # periapsis-region timescale (s)
        print(f"--- {tag} flyby: r_p={r_p:.3e} km ({r_p/71492:.1f} R_J), e={e:.4f}, "
              f"analytic δ={np.degrees(delta):.2f}°, |Δv_helio|={dv_helio:.3f} km/s")
        print(f"    v_peri={v_peri:.2f} km/s, periapsis timescale r_p/v_peri={t_peri:.3e} s "
              f"({t_peri/3600:.2f} h); R-N5 step dt_sub≈14 h={14*3600}s")
        print(f"    {'dt(s)':>9} {'dt/t_peri':>10} {'nsteps':>8} {'δ_meas(°)':>10} "
              f"{'δ_err(%)':>9} {'V∞_err(%)':>10} {'ε_drift':>10}")
        # H-N6c: sweep the step size at negligible softening
        for dt in args.dts:
            dmeas, vend, epsd, raw, n = integrate_flyby(mu, vinf, r_p, dt, soft=10.0)
            derr = abs(np.degrees(dmeas) - np.degrees(delta)) / np.degrees(delta) * 100
            verr = abs(vend - vinf) / vinf * 100
            print(f"    {dt:9.0f} {dt/t_peri:10.3f} {n:8d} {np.degrees(dmeas):10.3f} "
                  f"{derr:9.3f} {verr:10.4f} {epsd:10.2e}")
        print()

    # H-N6b: softening ceiling, at a well-resolved step on the tight flyby.
    # NOTE the metric: δ reconstructed from the eccentricity vector is BLIND to softening —
    # softened gravity is still a CENTRAL, conservative force, so it conserves ε and angular
    # momentum h, hence the osculating eccentricity (→ δ_recon) is invariant no matter the
    # softening. Softening changes the near-field PATH while conserving those invariants, so the
    # physically correct probe is the RAW velocity-turn angle (soft=10 is the resolved reference).
    r_p = args.rp_tight
    e, delta, v_peri, dv_helio0 = analytic(mu, vinf, r_p)
    dt_fine = args.dt_fine
    _, _, _, raw_ref, _ = integrate_flyby(mu, vinf, r_p, dt_fine, soft=10.0)   # resolved baseline
    print(f"--- softening sweep (tight flyby r_p={r_p:.2e} km, well-resolved dt={dt_fine:.0f}s) ---")
    print(f"    analytic asymptotic δ={np.degrees(delta):.3f}°; soft=10 resolved raw turn="
          f"{np.degrees(raw_ref):.3f}° (finite-distance ref)")
    print(f"    {'soft(km)':>10} {'soft/r_p':>10} {'raw_turn(°)':>12} {'suppress(%)':>12} "
          f"{'Δv_helio':>9} {'δ_recon(°)':>11}")
    for soft in args.softs:
        drecon, vend, epsd, raw, n = integrate_flyby(mu, vinf, r_p, dt_fine, soft=soft)
        suppress = (raw_ref - raw) / raw_ref * 100                    # turn lost vs resolved baseline
        dv_helio = 2.0 * vinf * np.sin(raw / 2.0)                     # actual free Δv the assist delivers
        print(f"    {soft:10.1f} {soft/r_p:10.2e} {np.degrees(raw):12.3f} {suppress:12.3f} "
              f"{dv_helio:9.3f} {np.degrees(drecon):11.3f}")
    print("\n  H-N6a: a feasible (soft,dt) reaches δ_err<2% & V∞_err<1% (see step-size tables). "
          "H-N6b: raw turn suppressed as soft→r_p (δ_recon blind — conserved invariants). "
          "H-N6c: δ_recon/ε blow up for dt≳t_peri (R-N5's 14h step fails the tight pass).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--vinf", type=float, default=6.0)
    ap.add_argument("--rp-wide", type=float, default=2.0e6)
    ap.add_argument("--rp-tight", type=float, default=2.0e5)
    ap.add_argument("--dts", type=float, nargs="+",
                    default=[100.0, 500.0, 2000.0, 10000.0, 50400.0, 100000.0])
    ap.add_argument("--dt-fine", type=float, default=200.0)
    ap.add_argument("--softs", type=float, nargs="+",
                    default=[10.0, 1e3, 1e4, 5e4, 1e5, 2e5, 5e5])
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
