#!/usr/bin/env python3
"""J2-assisted node change — physics + baseline (Build K, R-K1).

The project thesis: diff-sim can BEAT an analytic optimum where a fidelity term makes
the analytic sub-optimal. J2 is the cleanest such term — the impulsive RAAN-change Δv
is J2-BLIND, but J2 precesses the node for free. R-K1 verifies the J2 nodal-precession
rate (independent float64 numpy propagation vs Vallado's secular formula) and the fair
impulsive RAAN-change baseline, before any policy is built.

    uv run python scripts/j2_node.py --verify
"""
from __future__ import annotations

import argparse

import numpy as np

MU = 398600.4418          # km^3/s^2 (Earth), matches jaxsim
R_BODY = 6378.137         # km
J2 = 1.08262668e-3


def accel(rv):
    """Two-body + Vallado secular-J2 acceleration (ECI, equatorial z-axis)."""
    r = rv[:3]
    rn = np.sqrt(r @ r)
    a_kep = -MU * r / rn ** 3
    zr = r[2] / rn
    pre = -1.5 * J2 * MU * R_BODY ** 2 / rn ** 4
    a_j2 = pre * np.array([(1 - 5 * zr ** 2) * r[0] / rn,
                           (1 - 5 * zr ** 2) * r[1] / rn,
                           (3 - 5 * zr ** 2) * r[2] / rn])
    return np.concatenate([rv[3:], a_kep + a_j2])


def rk4(rv, dt):
    k1 = accel(rv); k2 = accel(rv + 0.5 * dt * k1)
    k3 = accel(rv + 0.5 * dt * k2); k4 = accel(rv + dt * k3)
    return rv + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def raan_of(rv):
    """Right ascension of ascending node from a state vector."""
    r, v = rv[:3], rv[3:]
    h = np.cross(r, v)
    n = np.cross([0, 0, 1.0], h)         # node vector
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return 0.0
    om = np.arctan2(n[1], n[0])
    return om % (2 * np.pi)


def circular_state(alt, inc_deg, raan_deg=0.0):
    """Circular orbit at altitude alt (km), inclination inc, RAAN — start at the node."""
    a = R_BODY + alt
    v = np.sqrt(MU / a)
    inc, raan = np.radians(inc_deg), np.radians(raan_deg)
    # position at the ascending node (in the orbit plane, along the node line)
    r = a * np.array([np.cos(raan), np.sin(raan), 0.0])
    # velocity: prograde, inclined — rotate (0,v,0)-ish by raan then tilt by inc
    vdir = np.array([-np.sin(raan) * np.cos(inc), np.cos(raan) * np.cos(inc), np.sin(inc)])
    return np.concatenate([r, v * vdir])


def vallado_raan_rate(alt, inc_deg):
    """Secular nodal regression dΩ/dt (rad/s): −1.5 n J2 (R/p)^2 cos i (e=0 → p=a)."""
    a = R_BODY + alt
    n = np.sqrt(MU / a ** 3)
    inc = np.radians(inc_deg)
    return -1.5 * n * J2 * (R_BODY / a) ** 2 * np.cos(inc)


def raan_change_dv(alt, inc_deg, draan_deg):
    """Fair impulsive RAAN-change Δv: rotate the plane by θ where
    cos θ = cos^2 i + sin^2 i cos ΔΩ; Δv = 2 v sin(θ/2) (circular, single burn at node)."""
    a = R_BODY + alt
    v = np.sqrt(MU / a)
    inc, dO = np.radians(inc_deg), np.radians(draan_deg)
    cth = np.cos(inc) ** 2 + np.sin(inc) ** 2 * np.cos(dO)
    theta = np.arccos(np.clip(cth, -1, 1))
    return 2 * v * np.sin(theta / 2), np.degrees(theta)


def verify(args):
    print("=== R-K1: J2 nodal precession + RAAN-change baseline ===")
    for alt, inc in ((700.0, 51.6), (700.0, 98.0), (2000.0, 51.6)):
        rv = circular_state(alt, inc)
        n_orbit = 2 * np.pi * np.sqrt((R_BODY + alt) ** 3 / MU)
        dt = 1.0
        steps = int(round(args.revs * n_orbit / dt))
        om0 = raan_of(rv)
        oms = [om0]
        for i in range(steps):
            rv = rk4(rv, dt)
            oms.append(raan_of(rv))
        oms = np.unwrap(np.array(oms))
        t = np.arange(len(oms)) * dt
        rate_num = np.polyfit(t, oms, 1)[0]              # rad/s
        rate_ana = vallado_raan_rate(alt, inc)
        deg_day = np.degrees(rate_num) * 86400.0
        deg_day_a = np.degrees(rate_ana) * 86400.0
        err = abs(rate_num - rate_ana) / abs(rate_ana) * 100
        print(f"  alt={alt:.0f} km i={inc:.1f}°: dΩ/dt numeric {deg_day:+.3f}°/day  "
              f"Vallado {deg_day_a:+.3f}°/day  (err {err:.1f}%)")
    print("  --- fair impulsive RAAN-change baseline (700 km, i=51.6°) ---")
    for dO in (10.0, 30.0, 60.0):
        dv, th = raan_change_dv(700.0, 51.6, dO)
        # free-drift time to accrue this ΔΩ via J2 alone
        rate = vallado_raan_rate(700.0, 51.6)
        t_days = abs(np.radians(dO) / rate) / 86400.0
        print(f"  ΔΩ={dO:.0f}°: plane-rot θ={th:.1f}°, impulsive Δv={dv:.4f} km/s; "
              f"J2 free-drift time {t_days:.1f} d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--revs", type=float, default=15.0)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
