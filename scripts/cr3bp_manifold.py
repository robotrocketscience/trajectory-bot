#!/usr/bin/env python3
"""Stable-manifold ballistic capture in the CR3BP (Build H).

G's naive departure-burn grid could not find a ballistic capture because the capture
set is the STABLE MANIFOLD of an L1/L2 periodic (Lyapunov) orbit — a measure-zero
tube you must seed onto deliberately (Conley-McGehee; Koon-Lo-Marsden-Ross 2011). H
builds that structure:

  R-H1  Lyapunov orbit about L1/L2 via differential correction   (--orbit)
  R-H2  monodromy eigenstructure + stable manifold               (--manifold)
  R-H3  sweep manifold trajectories for VERIFIED ballistic capture (--capture)
  R-H4  Earth-connection + honest Δv vs Hohmann+capture           (--transfer)

Dynamics here are an INDEPENDENT float64 numpy reimplementation of cr3bp_sim's
rotating-frame CR3BP (differential correction needs float64, and a second
implementation is a bug-catch); cross-checked against cr3bp_sim.accel in --check.

    uv run --with jax python scripts/cr3bp_manifold.py --check --orbit
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402  (for MU, geometry, cross-check, physical units)

MU = C.MU
OM = 1.0 - MU
R_MOON = np.array([1.0 - MU, 0.0, 0.0])
R_HILL = (MU / 3.0) ** (1.0 / 3.0)


# ----- independent float64 dynamics -------------------------------------------------
def accel_np(s):
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    ax = 2.0 * vy + x - OM * (x + MU) / r1 ** 3 - MU * (x - 1.0 + MU) / r2 ** 3
    ay = -2.0 * vx + y - OM * y / r1 ** 3 - MU * y / r2 ** 3
    az = -OM * z / r1 ** 3 - MU * z / r2 ** 3
    return np.array([vx, vy, vz, ax, ay, az])


def rk4_np(s, dt):
    k1 = accel_np(s)
    k2 = accel_np(s + 0.5 * dt * k1)
    k3 = accel_np(s + 0.5 * dt * k2)
    k4 = accel_np(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobi_np(s):
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    Om = 0.5 * (x ** 2 + y ** 2) + OM / r1 + MU / r2
    return 2.0 * Om - (vx * vx + vy * vy + vz * vz)


def omega_second(xL):
    """Ω_xx, Ω_yy at a collinear point (y=z=0). c = (1-μ)/|x+μ|³ + μ/|x-1+μ|³."""
    c = OM / abs(xL + MU) ** 3 + MU / abs(xL - 1.0 + MU) ** 3
    return 1.0 + 2.0 * c, 1.0 - c, c


def linear_seed(xL, Ax):
    """In-plane center-mode seed: x = xL - Ax cos(ω t), y = κ Ax sin(ω t)."""
    Uxx, Uyy, _ = omega_second(xL)
    # Λ⁴ + (4 - Uxx - Uyy)Λ² + Uxx·Uyy = 0; center root β = Λ² < 0 → ω = √(-β)
    b = 4.0 - Uxx - Uyy
    disc = np.sqrt(b * b - 4.0 * Uxx * Uyy)
    beta_center = 0.5 * (-b - disc)           # the negative root
    wp = np.sqrt(-beta_center)
    kappa = (wp * wp + Uxx) / (2.0 * wp)
    return np.array([xL - Ax, 0.0, 0.0, 0.0, kappa * Ax * wp, 0.0]), wp


# ----- differential correction (R-H1) -----------------------------------------------
def half_period_crossing(s0, dt):
    """Integrate until y returns to 0 (from +); linear-interp the crossing state & time."""
    s = s0.copy()
    t = 0.0
    y_prev = s[1]
    s_prev = s.copy()
    # step off the y=0 axis first
    for _ in range(20_000_000):
        s_new = rk4_np(s, dt)
        t_new = t + dt
        if s[1] >= 0.0 and s_new[1] < 0.0 and t_new > dt * 5:
            frac = s[1] / (s[1] - s_new[1])       # fraction to y=0
            s_cross = s + frac * (s_new - s)
            return s_cross, t + frac * dt
        s_prev, y_prev = s, s[1]
        s, t = s_new, t_new
    raise RuntimeError("no y=0 crossing found")


def integrate_fixed_time(s0, dt, T):
    """Integrate for exactly time T (floor steps + one remainder step)."""
    n = int(np.floor(T / dt))
    s = s0.copy()
    for _ in range(n):
        s = rk4_np(s, dt)
    rem = T - n * dt
    if rem > 1e-15:
        s = rk4_np(s, rem)
    return s


def differential_correct(xL, Ax, dt=1e-4, tol=1e-11, maxit=40):
    """STM-based correction: hold x0 fixed, correct vy0 so the return y=0 crossing is
    perpendicular (vx=0). Accounts for crossing-time variation via the
    a_x/vy term (Koon-Lo-Marsden-Ross §4): δvy0 = -vx_c / (Φ34 − (a_xc/vy_c)Φ24)."""
    s0, wp = linear_seed(xL, Ax)
    for it in range(maxit):
        sc, thalf = half_period_crossing(s0, dt)
        vx_c = sc[3]
        if abs(vx_c) < tol:
            return s0.copy(), 2.0 * thalf, wp, it, abs(vx_c)
        a_c = accel_np(sc)                          # accel at crossing → a_xc = a_c[3]
        vy_c = sc[4]
        # STM columns for the vy0 perturbation, evaluated at the SAME crossing time
        h = 1e-7
        s0h = s0.copy(); s0h[4] += h
        sch = integrate_fixed_time(s0h, dt, thalf)
        phi34 = (sch[3] - vx_c) / h                 # ∂vx_c/∂vy0
        phi24 = (sch[1] - sc[1]) / h                # ∂y_c/∂vy0
        denom = phi34 - (a_c[3] / vy_c) * phi24
        s0[4] -= vx_c / denom
    sc, thalf = half_period_crossing(s0, dt)
    return s0.copy(), 2.0 * thalf, wp, maxit, abs(sc[3])


def propagate_np(s0, dt, n):
    traj = np.empty((n + 1, 6))
    traj[0] = s0
    s = s0.copy()
    for i in range(n):
        s = rk4_np(s, dt)
        traj[i + 1] = s
    return traj


def check_dynamics():
    """Cross-check the numpy dynamics against the verified JAX cr3bp_sim.accel."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(200):
        s = rng.normal(size=6) * np.array([1.5, 1.5, 0.3, 1.0, 1.0, 0.3]) + \
            np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_np = accel_np(s)[3:]                 # accel_np returns [v(3), a(3)]
        a_jx = np.asarray(C.accel(jnp.asarray(s)))   # C.accel returns a(3) only
        maxerr = max(maxerr, float(np.max(np.abs(a_np - a_jx))))
    print(f"  numpy-vs-JAX accel max |Δ| over 200 random states: {maxerr:.2e} "
          f"(float32 JAX vs float64 numpy; want <1e-4)")


def run_orbit(args):
    print("=== R-H1: Lyapunov orbits via differential correction ===")
    lp = C.lagrange_points()
    for name, Ax in (("L1", args.ax_l1), ("L2", args.ax_l2)):
        xL = lp[name]
        s0, T, wp, it, res = differential_correct(xL, Ax, dt=args.dt)
        traj = propagate_np(s0, args.dt, int(round(T / args.dt)))
        C0 = jacobi_np(s0); Cend = jacobi_np(traj[-1])
        close = np.linalg.norm(traj[-1] - s0)
        xamp = 0.5 * (traj[:, 0].max() - traj[:, 0].min())
        yamp = traj[:, 1].max()
        print(f"  {name} (x_L={xL:+.5f}, seed Ax={Ax:.3f}): converged in {it} Newton "
              f"steps, |vx_cross|={res:.1e}")
        print(f"     period T={T:.5f} ({T*C.T_UNIT_S/86400:.3f} d)  linear 2π/ω={2*np.pi/wp:.4f}"
              f"  Jacobi C={C0:.6f} drift={abs(Cend-C0):.1e}")
        print(f"     closure |s(T)-s0|={close:.1e}  x-amp={xamp:.4f} y-amp={yamp:.4f} "
              f"({xamp*C.L_UNIT_KM:.0f} km)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="cross-check numpy vs JAX dynamics")
    ap.add_argument("--orbit", action="store_true", help="R-H1: Lyapunov orbits")
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--ax-l1", type=float, default=0.02)
    ap.add_argument("--ax-l2", type=float, default=0.02)
    args = ap.parse_args()
    if args.check:
        check_dynamics()
    if args.orbit:
        run_orbit(args)


if __name__ == "__main__":
    main()
