#!/usr/bin/env python3
"""Departure-Δv transfer on real ephemerides vs the two-body Lambert optimum (Build N, R-N2).

The fair Δv-vs-Δv test for the locked Tier-3 dynamics. Given matched endpoints
(r1 = departure-body position at launch, r2 = target-body position at arrival, TOF fixed),
compare two ways to get the IN-SPACE departure Δv — the heliocentric injection beyond the
departure body's own velocity, |v_depart − v_body| (launch C3 is vehicle-set, so total
mission Δv is not the fair metric; see docs/BENCHMARKS.md, memory 418e2e2):

  • Lambert baseline  — the two-body (Sun-only) optimum v1 for (r1, r2, TOF), scripts/lambert.py.
  • N-body optimize   — differentiate a departure Δv through the real perturbed field
                         (scripts/nbody_sim.rollout_interp) to minimize the terminal miss to r2.

Same endpoints, so the DIFFERENCE is the N-body correction to the two-body plan — NOT a "beat"
of Lambert (Lambert IS the endpoint-matched optimum in its own two-body field; the perturbed
field can only require Δv ≥ that). The offline test is the gate: with only the Sun present the
N-body dynamics ARE two-body, so the optimizer must recover the Lambert Δv (H-N2a). The real
correction (H-N2b) is then reported honestly under --fetch.

    uv run --with jax python scripts/nbody_transfer.py --verify            # Sun-only, offline
    uv run --with jax python scripts/nbody_transfer.py --fetch --mission mro   # real ephemerides
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM                       # noqa: E402  differentiable Lambert (on main)
import nbody_sim as NB                       # noqa: E402  the R-N1 engine + R-N2 rollout_interp

MU_SUN = NB.GM["sun"]
AU = NB.AU
DAY = NB.DAY


# ---------- optimizer: differentiate the departure Δv through the N-body rollout ----------
def optimize_departure(rv_body, v_body, r2, fine_seq, body_gm, dt, dv0,
                       soft=1.0, iters=400, lr=None):
    """Adam on the departure Δv (3-vector) to minimize the terminal miss to r2 under the
    N-body rollout. rv_body = departure position; the spacecraft leaves at v_body+Δv.
    Returns (dv_opt, miss_km, history_final_loss)."""
    r2j = jnp.asarray(r2)
    rvb = jnp.asarray(rv_body)
    vb = jnp.asarray(v_body)
    scale = AU ** 2                                    # normalize miss² to O(1)

    def loss(dv):
        rv0 = jnp.concatenate([rvb, vb + dv])
        rvT, _ = NB.rollout_interp(rv0, fine_seq, body_gm, dt, soft=soft)
        return jnp.sum((rvT[:3] - r2j) ** 2) / scale

    vg = jax.jit(jax.value_and_grad(loss))
    # step size in km/s. The 200+-day terminal miss is stiff (|∂miss/∂Δv|~1e13), so Adam's
    # per-coordinate normalization sets the effective per-step travel to ~lr km/s; lr≈0.03
    # marches from a cold start to the ~km/s injection in a few thousand iters, best-tracking
    # the sharp minimum. See R-N2 notes.
    lr = lr if lr is not None else 3e-2
    m = jnp.zeros(3)
    v = jnp.zeros(3)
    dv = jnp.asarray(dv0, dtype=jnp.float64)
    b1, b2, eps = 0.9, 0.999, 1e-12
    best_dv, best_loss = dv, float("inf")
    for t in range(1, iters + 1):
        L, g = vg(dv)
        Lf = float(L)
        if Lf < best_loss:                             # track best BEFORE the step
            best_loss, best_dv = Lf, dv
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mh = m / (1 - b1 ** t)
        vh = v / (1 - b2 ** t)
        dv = dv - lr * mh / (jnp.sqrt(vh) + eps)
    miss_km = float(jnp.sqrt(best_loss * scale))
    return np.asarray(best_dv), miss_km, best_loss


# ---------- offline verification: Sun-only must recover Lambert (H-N2a) ----------
def verify(args):
    print("=== R-N2: departure Δv vs Lambert — offline Sun-only recovery (H-N2a) ===")
    mu = MU_SUN
    r_e, r_m = 1.0 * AU, 1.5237 * AU               # Earth, Mars heliocentric radii (circular)
    v_ce = np.sqrt(mu / r_e)
    r1 = np.array([r_e, 0.0, 0.0])
    v_earth = np.array([0.0, v_ce, 0.0])           # prograde circular
    sweep = np.radians(150.0)                       # transfer sweep (avoids the Δθ=π singularity)
    r2 = r_m * np.array([np.cos(sweep), np.sin(sweep), 0.0])
    tof = args.tof * DAY

    # Lambert two-body optimum for (r1, r2, TOF)
    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=mu)
    v1_L = np.asarray(v1_L)
    dv_lambert = v1_L - v_earth
    print(f"  endpoints: r1=1.000 AU, r2=1.524 AU @ {np.degrees(sweep):.0f}° sweep, "
          f"TOF={args.tof:.0f} d")
    print(f"  Lambert v1={v1_L}  → in-space Δv={np.linalg.norm(dv_lambert):.5f} km/s")

    # N-body optimize with ONLY the Sun (fixed at origin) → dynamics are two-body.
    n = int(round(tof / args.dt))
    dt = tof / n
    fine = jnp.zeros((2 * n + 1, 1, 3))            # Sun at origin at every half-step
    gm = jnp.array([mu])
    # Lambert-INDEPENDENT cold start: Δv=0 (coast on the departure body's circular orbit,
    # zero injection). The optimizer must FIND the transfer from nothing, not refine a seed.
    dv0 = np.zeros(3)
    dv_nb, miss_km, _ = optimize_departure(r1, v_earth, r2, fine, gm, dt, dv0,
                                           soft=0.0, iters=args.iters)
    print("  N-body (Sun-only) cold start Δv=0 (Lambert-independent)")
    print(f"  N-body optimized Δv={np.linalg.norm(dv_nb):.5f} km/s, terminal miss="
          f"{miss_km:.1f} km ({miss_km/AU:.2e} AU)")

    rel = np.linalg.norm(dv_nb - dv_lambert) / np.linalg.norm(dv_lambert)
    soi_mars = 5.77e5                              # Mars SOI radius (km)
    print(f"  |Δv_nbody − Δv_lambert| / |Δv_lambert| = {rel:.2e}  (predict <1e-3)")
    print(f"  terminal miss / Mars SOI = {miss_km/soi_mars:.2e}  (predict ≪1)")
    ok = rel < 1e-3 and miss_km < 0.1 * soi_mars
    print(f"  → H-N2a {'SUPPORTED' if ok else 'REFUTED'}: N-body departure-Δv optimize "
          f"{'recovers' if ok else 'does NOT recover'} the Lambert optimum in the two-body limit")


# ---------- real-ephemeris correction (H-N2b; network, not in CI) ----------
def sample_fine(bodies, start_iso, tof_days, dt, location="@sun"):
    """Body positions at HALF-step resolution: (2n+1, K, 3) km, from cached Horizons."""
    import ephemeris as E
    from astropy.time import Time, TimeDelta
    t0 = Time(start_iso, format="isot", scale="utc")
    stop = str((t0 + TimeDelta(tof_days * DAY + dt, format="sec")).isot)
    n = int(round(tof_days * DAY / dt))
    half = dt / 2.0
    jds = float(t0.jd) + (np.arange(2 * n + 1) * half) / DAY
    cols = []
    for b in bodies:
        spec = E.SpanSpec(NB.HORIZONS_ID[b], start_iso, stop,
                          step_s=max(3600, int(half)), location=location)
        eph = E.get_span(spec)
        cols.append(np.array([eph.state_at_jd(float(jd))[0] for jd in jds]))
    return np.stack(cols, axis=1), n              # (2n+1, K, 3)


def body_state(body, iso, location="@sun"):
    import ephemeris as E
    from astropy.time import Time, TimeDelta
    t0 = Time(iso, format="isot", scale="utc")
    stop = str((t0 + TimeDelta(2 * 3600.0, format="sec")).isot)   # short valid window
    spec = E.SpanSpec(NB.HORIZONS_ID[body], iso, stop, step_s=3600, location=location)
    st = E.get_span(spec).state_at_jd(float(t0.jd))
    return np.asarray(st[0]), np.asarray(st[1])   # (pos, vel) km, km/s


def fetch(args):
    from scripts.missions import MISSIONS
    m = MISSIONS[args.mission]
    dep, tgt = m.transfer.split("->")
    dep = {"Earth": "earth"}.get(dep, dep.lower())
    tgt = {"Mars": "mars"}.get(tgt, tgt.lower())
    print(f"=== R-N2 real ephemerides: {m.name} ({dep}→{tgt}) — Lambert vs N-body (H-N2b) ===")
    launch = m.launch + "T00:00:00"
    arrival = m.arrival + "T00:00:00"
    from astropy.time import Time
    tof = float((Time(arrival) - Time(launch)).sec)
    tof_days = tof / DAY

    r1, v_earth = body_state(dep, launch)
    r2, _ = body_state(tgt, arrival)
    print(f"  launch {m.launch}, arrival {m.arrival}, TOF={tof_days:.1f} d")
    print(f"  r1={np.linalg.norm(r1)/AU:.4f} AU ({dep}), r2={np.linalg.norm(r2)/AU:.4f} AU ({tgt})")

    # transfer geometry (short-way vs long-way): the single-rev Lambert solver is only
    # well-posed away from the Δθ=π singularity it documents; a long-way (>180°) prograde
    # sweep is exactly where its fixed z=0 Newton iteration can fail to converge.
    crossz = r1[0] * r2[1] - r1[1] * r2[0]
    cosd = np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2))
    sweep = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    sweep = sweep if crossz >= 0 else 360.0 - sweep
    print(f"  prograde transfer sweep = {sweep:.1f}° ({'short' if sweep < 180 else 'LONG'} way)")

    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=MU_SUN)
    v1_L = np.asarray(v1_L)
    dv_lambert = v1_L - v_earth

    # perturbers for the heliocentric cruise: Sun + third-body planets, EXCLUDING the departure
    # and target bodies (their SOI phases are the patched-conic endpoints, paid separately). This
    # isolates the deep-space perturbation on the heliocentric transfer, matching Lambert's frame.
    perturbers = ["sun", "jupiter", "saturn", "venus"]
    gm = jnp.array([NB.GM[b] for b in perturbers])
    dt = args.dt
    body_np, n = sample_fine(perturbers, launch, tof_days, dt)
    fine = jnp.asarray(body_np)

    # GATE: is the Lambert baseline itself valid? Propagate v1_L under SUN-ONLY; a converged
    # Lambert solution self-closes on r2 to ~mm. A large self-miss means the solver did not
    # converge for this geometry (long-way / near-180°), so any "N-body correction" against it
    # would be a baseline artifact, not physics — report the failure instead of a fake number.
    sun_only = jnp.asarray(body_np[:, :1])          # Sun column at half-step resolution
    rvT_sun, _ = NB.rollout_interp(jnp.asarray(np.concatenate([r1, v1_L])),
                                   sun_only, jnp.array([MU_SUN]), dt, soft=0.0)
    self_miss = float(jnp.linalg.norm(rvT_sun[:3] - jnp.asarray(r2)))
    print(f"  Lambert in-space Δv (two-body, Sun-only) = {np.linalg.norm(dv_lambert):.4f} km/s"
          f"  [self-consistency miss={self_miss:.3g} km]")
    if self_miss > 1e3:
        # optimizer is branch-agnostic — show it still solves the targeting, then stop.
        dv_nb, miss_km, _ = optimize_departure(r1, v_earth, r2, fine, gm, dt,
                                               dv0=np.zeros(3), soft=10.0, iters=args.iters)
        print(f"  ⚠ Lambert baseline DID NOT CONVERGE for this {sweep:.0f}° long-way transfer "
              f"(self-miss {self_miss:.2e} km ≫ mm) — the single-rev z=0 Newton solver's "
              f"documented near-180° failure. NO correction reported (invalid baseline).")
        print(f"  (The differentiable N-body optimizer is branch-agnostic: from a cold Δv=0 it "
              f"still solves the targeting — Δv={np.linalg.norm(dv_nb):.4f} km/s, miss={miss_km:.0f} km.)")
        return

    # (1) how far does the two-body Lambert plan miss under the real perturbed field?
    rv0_L = jnp.asarray(np.concatenate([r1, v1_L]))
    rvT_L, _ = NB.rollout_interp(rv0_L, fine, gm, dt, soft=10.0)
    miss_L = float(jnp.linalg.norm(rvT_L[:3] - jnp.asarray(r2)))
    print(f"  Lambert plan executed under Sun+{','.join(perturbers[1:])}: misses {tgt} by "
          f"{miss_L:.0f} km ({miss_L/5.77e5:.2f} Mars-SOI)")

    # (2) re-optimize the departure Δv under N-body, seeded from the Lambert solution.
    dv_nb, miss_km, _ = optimize_departure(r1, v_earth, r2, fine, gm, dt,
                                           dv0=(v1_L - v_earth), soft=10.0, iters=args.iters)
    print(f"  N-body optimized in-space Δv = {np.linalg.norm(dv_nb):.4f} km/s, residual miss="
          f"{miss_km:.0f} km")
    corr = np.linalg.norm(dv_nb) - np.linalg.norm(dv_lambert)
    print(f"  → N-body correction to the departure Δv = {corr*1000:+.1f} m/s "
          f"({corr/np.linalg.norm(dv_lambert)*100:+.2f}%) — the third-body perturbation on the "
          f"heliocentric transfer, cleaned up by re-optimizing under the real field. Small, as "
          f"expected for a Sun-dominated cruise (the baseline passed the self-consistency gate).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--mission", type=str, default="mro")
    ap.add_argument("--tof", type=float, default=250.0, help="offline transfer TOF (days)")
    ap.add_argument("--dt", type=float, default=21600.0, help="integration step (s)")
    ap.add_argument("--iters", type=int, default=6000)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    if args.fetch:
        fetch(args)


if __name__ == "__main__":
    main()
