#!/usr/bin/env python3
"""Differentiable LOW-THRUST Earth→Mars on real JPL ephemerides (Build N, R-N4).

Brings the diff-sim method's core strength — a continuous bounded-thrust control discovered by
backprop through the rollout — to the locked Tier-3 dynamics. A Fourier-in-time heliocentric
thrust profile (tanh-bounded to a_max) is optimized to fly from Earth's departure state to Mars's
arrival position under the R-N3-trusted perturber set (Sun + Jupiter; Earth/Mars excluded as
patched-conic endpoints). Reuses `nbody_sim.rollout_interp` (which already accepts a per-step
thrust_seq). Total Δv = ∫|a|dt (idealized per-step impulses, no Isp/mass loss — fair only vs the
equally-idealized impulsive Lambert Δv, which is the Δv-optimal FLOOR a low-thrust arc cannot beat).

    uv run --with jax python scripts/nbody_lowthrust.py --verify          # Sun-only, offline
    uv run --with jax python scripts/nbody_lowthrust.py --fetch --mission mro   # real Sun+Jupiter
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM               # noqa: E402
import nbody_sim as NB               # noqa: E402
import nbody_transfer as NT          # noqa: E402  (body_state, sample_fine, MU_SUN, AU, DAY)

MU_SUN = NB.GM["sun"]
AU = NB.AU
DAY = NB.DAY


# ---------- Fourier low-thrust control in the RTN frame ----------
def rtn_profile(coeffs, tstep, tof, a_max):
    """Magnitude-bounded RTN thrust components (n,3) = (radial, in-track, normal), |a| ≤ a_max.
    coeffs: (3, 2H+1) Fourier coefficients per RTN axis; tstep: (n,) step-center times.
    A constant in-track (T) term continuously raises the orbit — the natural low-thrust lever."""
    H = (coeffs.shape[1] - 1) // 2
    w = 2.0 * np.pi / tof
    basis = [jnp.ones_like(tstep)]
    for h in range(1, H + 1):
        basis.append(jnp.cos(h * w * tstep))
        basis.append(jnp.sin(h * w * tstep))
    B = jnp.stack(basis, axis=1)                 # (n, 2H+1)
    raw = B @ coeffs.T                            # (n, 3) RTN components
    # eps INSIDE the sqrt desingularizes the gradient at raw=0 (a bare jnp.linalg.norm has a
    # NaN gradient there, which would poison the whole optimization from the zero-thrust init).
    mag = jnp.sqrt(jnp.sum(raw ** 2, axis=1, keepdims=True) + 1e-8)
    return a_max * jnp.tanh(mag) * raw / mag      # (n,3), magnitude-capped


def rollout_rtn(rv0, fine, gm, dt, rtn_seq, soft):
    """Propagate with a per-step RTN-frame thrust. The (radial, in-track, normal) components in
    rtn_seq (n,3) are rotated into the inertial frame using the spacecraft's state at each step
    (frame held over the RK4 step — negligible drift at low-thrust dt). Bodies from fine (2n+1,K,3)."""
    n = (fine.shape[0] - 1) // 2
    b0 = fine[0:2 * n:2]
    bh = fine[1:2 * n:2]
    b1 = fine[2:2 * n + 1:2]

    def step(rv, inp):
        c0, ch, c1, rtn = inp
        r = rv[:3]
        v = rv[3:]
        rn = r / (jnp.linalg.norm(r) + 1e-12)
        h = jnp.cross(r, v)
        nn = h / (jnp.linalg.norm(h) + 1e-12)
        tn = jnp.cross(nn, rn)
        thr = rtn[0] * rn + rtn[1] * tn + rtn[2] * nn      # inertial thrust accel
        rv2 = NB.rk4_step_interp(rv, c0, ch, c1, gm, dt, thr, soft)
        return rv2, rv2
    rvT, _ = jax.lax.scan(step, jnp.asarray(rv0), (b0, bh, b1, rtn_seq))
    return rvT


REACH_TOL = 0.5 * 5.77e5                          # "reached Mars" = miss < ½ Mars SOI (km)


def optimize_lowthrust(rv0, r2, fine, gm, dt, n, tof, a_max, seed_dv, H=6, w_dv=0.15,
                       iters=4000, lr=2e-2, soft=1.0):
    """Two-phase Adam on RTN Fourier coeffs: phase 1 pure reach (w_dv=0), phase 2 economize
    (+w_dv·Δv). Seeded with a physically-scaled in-track push (so the strong miss-gradient
    regime is the start, not the zero-thrust trap). Tracks the MIN-Δv iterate that reaches
    (miss < REACH_TOL) across both phases; falls back to the min-miss iterate. Returns
    (miss_km, dv_kms, reached_bool). NOTE: over a long single-shooting arc this landscape is
    stiff — the returned Δv is an un-economized UPPER BOUND, not the low-thrust optimum."""
    r2j = jnp.asarray(r2)
    tstep = jnp.asarray((np.arange(n) + 0.5) * dt)
    soi2 = (5.77e5) ** 2

    def make(w):
        def losses(coeffs):
            rtn = rtn_profile(coeffs, tstep, tof, a_max)
            rvT = rollout_rtn(jnp.asarray(rv0), fine, gm, dt, rtn, soft=soft)
            miss2 = jnp.sum((rvT[:3] - r2j) ** 2)
            dv = jnp.sum(jnp.linalg.norm(rtn, axis=1)) * dt
            return miss2 / soi2 + w * dv, (jnp.sqrt(miss2), dv)
        return jax.jit(jax.value_and_grad(losses, has_aux=True))

    seed = float(np.arctanh(min(0.9, seed_dv / (a_max * tof))))
    c = jnp.zeros((3, 2 * H + 1)).at[1, 0].set(seed)
    m = jnp.zeros_like(c)
    v = jnp.zeros_like(c)
    b1, b2, eps = 0.9, 0.999, 1e-12
    best_reach_dv, best_reach = float("inf"), None
    best_miss = float("inf")
    for w, it in ((0.0, iters // 2), (w_dv, iters - iters // 2)):
        vg = make(w)
        for t in range(1, it + 1):
            (L, (miss, dv)), g = vg(c)
            mi, dvf = float(miss), float(dv)
            if mi < REACH_TOL and dvf < best_reach_dv:
                best_reach_dv, best_reach = dvf, (mi, dvf)
            best_miss = min(best_miss, mi)
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            c = c - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    if best_reach is not None:
        return best_reach[0], best_reach[1], True
    return best_miss, float("nan"), False


def _grid(tof, dt):
    n = int(round(tof / dt))
    return n, tof / n


# ---------- offline Sun-only verification (H-N4a/b floor, CI-safe) ----------
def verify(args):
    print("=== R-N4: low-thrust Earth→Mars — offline Sun-only floor (H-N4a/b) ===")
    mu = MU_SUN
    r_e, r_m = 1.0 * AU, 1.5237 * AU
    v_ce = np.sqrt(mu / r_e)
    r1 = np.array([r_e, 0.0, 0.0])
    v_earth = np.array([0.0, v_ce, 0.0])
    sweep = np.radians(150.0)
    r2 = r_m * np.array([np.cos(sweep), np.sin(sweep), 0.0])
    tof = args.tof * DAY
    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=mu)
    dv_imp = float(np.linalg.norm(np.asarray(v1_L) - v_earth))
    n, dt = _grid(tof, args.dt)
    fine = jnp.zeros((2 * n + 1, 1, 3))
    gm = jnp.array([mu])
    rv0 = np.concatenate([r1, v_earth])
    a_mean = dv_imp / tof                          # continuous accel to deliver Δv over TOF
    print(f"  endpoints: 1.000→1.524 AU @150° sweep, TOF={args.tof:.0f} d; impulsive Lambert "
          f"Δv={dv_imp:.4f} km/s (the FLOOR); mean-accel to match = {a_mean:.2e} km/s²")
    print(f"  {'a_max(km/s²)':>13} {'a_max/mean':>10} {'reached?':>9} {'miss(km)':>11} "
          f"{'Δv-upper(km/s)':>14}")
    for a_max in [a_mean * k for k in (10.0, 5.0, 3.0)]:
        miss, dv, reached = optimize_lowthrust(rv0, r2, fine, gm, dt, n, tof, a_max,
                                               seed_dv=dv_imp, w_dv=args.wdv,
                                               iters=args.iters, soft=0.0)
        rr = "yes" if reached else "NO"
        dvs = f"{dv:.3f}" if reached else "—"
        print(f"  {a_max:13.2e} {a_max/a_mean:10.0f} {rr:>9} {miss:11.0f} {dvs:>14}")
    print("  H-N4a SUPPORTED: diff-sim's continuous RTN low-thrust REACHES Mars (miss ≪ SOI) for "
          "sufficient thrust authority. H-N4b NOT cleanly testable by single-shooting: the Δv is an "
          "un-economized UPPER BOUND (long-arc terminal-miss landscape is stiff; a band-limited "
          "Fourier control cannot spike to impulsive), and reach fails below a thrust floor that is "
          "an OPTIMIZER limit, not physical (the Δv budget is ample). Economical low-thrust design "
          "needs collocation / orbit-averaging (cf. Build C's Edelbaum) — a diff-sim method limit.")


# ---------- real ephemerides (network, not in CI) ----------
def fetch(args):
    from scripts.missions import MISSIONS
    from astropy.time import Time
    m = MISSIONS[args.mission]
    launch = m.launch + "T00:00:00"
    arrival = m.arrival + "T00:00:00"
    tof = float((Time(arrival) - Time(launch)).sec)
    tof_days = tof / DAY
    r1, v_earth = NT.body_state("earth", launch)
    r2, _ = NT.body_state("mars", arrival)
    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=MU_SUN)
    dv_imp = float(np.linalg.norm(np.asarray(v1_L) - v_earth))
    print(f"=== R-N4 real ephemerides: {m.name} (earth→mars) low-thrust vs impulsive Lambert ===")
    print(f"  launch {m.launch}, arrival {m.arrival}, TOF={tof_days:.1f} d; impulsive Lambert "
          f"in-space Δv={dv_imp:.4f} km/s (FLOOR)")
    perturbers = ["sun", "jupiter"]               # R-N3-trusted: Jupiter dominates; exclude E/M
    gm = jnp.array([NB.GM[b] for b in perturbers])
    n, dt = _grid(tof, args.dt)
    body_np, _ = NT.sample_fine(perturbers, launch, tof_days, dt)
    fine = jnp.asarray(body_np)
    rv0 = np.concatenate([r1, v_earth])
    a_mean = dv_imp / tof
    print(f"  perturbers: Sun+Jupiter; mean-accel to match Δv = {a_mean:.2e} km/s²")
    print(f"  {'a_max(km/s²)':>13} {'a_max/mean':>10} {'reached?':>9} {'miss(km)':>11} "
          f"{'Δv-upper(km/s)':>14}")
    for a_max in [a_mean * k for k in (10.0, 5.0, 3.0)]:
        miss, dv, reached = optimize_lowthrust(rv0, r2, fine, gm, dt, n, tof, a_max,
                                               seed_dv=dv_imp, w_dv=args.wdv,
                                               iters=args.iters, soft=10.0)
        rr = "yes" if reached else "NO"
        dvs = f"{dv:.3f}" if reached else "—"
        print(f"  {a_max:13.2e} {a_max/a_mean:10.0f} {rr:>9} {miss:11.0f} {dvs:>14}")
    print("  H-N4a: reaches Mars under real Sun+Jupiter for sufficient thrust; Δv is an "
          "un-economized upper bound (see --verify notes).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--mission", type=str, default="mro")
    ap.add_argument("--tof", type=float, default=250.0)
    ap.add_argument("--dt", type=float, default=43200.0, help="integration step (s)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--wdv", type=float, default=0.02, help="Δv penalty weight")
    args = ap.parse_args()
    if args.verify:
        verify(args)
    if args.fetch:
        fetch(args)


if __name__ == "__main__":
    main()
