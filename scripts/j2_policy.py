#!/usr/bin/env python3
"""J2-assisted node change — diff-sim policy vs the J2-blind optimum (Build K, R-K2).

Thesis test (H-K): a differentiable-sim policy, optimizing TRUE total Δv over the real
J2-on dynamics with a fixed time budget, can BEAT the J2-blind analytic optimum for a
RAAN (node) change — by DISCOVERING that lowering the orbit accelerates J2 nodal drift
(dΩ/dt ∝ a^-7/2 cos i), buying the node change far cheaper than an impulsive plane
rotation (which is J2-blind: Δv = 2v sin(θ/2)).

Design choices for an HONEST, DISCOVERY-genuine test:
  * No baked-in "dive" structure. The control is a per-step impulsive Δv (an
    optimal-control formulation); the policy must DISCOVER that spending Δv on altitude
    (to speed drift) beats spending it on plane rotation. Total Δv = Σ|Δv_i|.
  * Steel-manned baseline: the J2-blind planner may use EITHER a single-impulse plane
    change OR a bi-elliptic (raise-apoapsis, rotate-cheap, return) plane change —
    whichever is cheaper. We take the min. (Time-feasibility of bi-elliptic is noted.)
  * The maneuver must actually REACH the target (RAAN, and return to circular at the
    same a and i) within the budget — verified by re-flying, not asserted.
  * Regime-specific: LEO/MEO where J2 matters. State the regime.

    uv run --with jax python scripts/j2_policy.py --check      # R-K2a building block
    uv run --with jax python scripts/j2_policy.py --optimize   # R-K2b beat test
"""
from __future__ import annotations

import argparse

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

MU = 398600.4418          # km^3/s^2 (Earth)
R_BODY = 6378.137         # km
J2 = 1.08262668e-3
DAY = 86400.0


# ---------- dynamics (JAX, Vallado secular-J2 ECI — matches verified j2_node.py) ----------
def accel(rv):
    r = rv[:3]
    # floor at the surface: a burn that drives r toward 0 gives huge accel -> NaN/blowup
    # gradients over the long rollout (fake sub-surface dynamics). Valid orbits unaffected.
    rn = jnp.maximum(jnp.sqrt(r @ r), R_BODY)
    a_kep = -MU * r / rn ** 3
    zr = r[2] / rn
    pre = -1.5 * J2 * MU * R_BODY ** 2 / rn ** 4
    a_j2 = pre * jnp.stack([(1 - 5 * zr ** 2) * r[0] / rn,
                            (1 - 5 * zr ** 2) * r[1] / rn,
                            (3 - 5 * zr ** 2) * r[2] / rn])
    return jnp.concatenate([rv[3:], a_kep + a_j2])


def rk4(rv, dt):
    k1 = accel(rv); k2 = accel(rv + 0.5 * dt * k1)
    k3 = accel(rv + 0.5 * dt * k2); k4 = accel(rv + dt * k3)
    return rv + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------- orbit helpers ----------
def circular_state(alt, inc_deg, raan_deg=0.0):
    return orbit_state(R_BODY + alt, 0.0, inc_deg, raan_deg)


def orbit_state(a, e, inc_deg, raan_deg=0.0):
    """State at periapsis (argument of periapsis = 0, so periapsis at the ascending node)
    for an orbit of semi-major a, eccentricity e, inclination, RAAN. e=0 → circular."""
    r_p = a * (1.0 - e)
    v_p = np.sqrt(MU * (1.0 + e) / (a * (1.0 - e)))
    inc, raan = np.radians(inc_deg), np.radians(raan_deg)
    r = r_p * np.array([np.cos(raan), np.sin(raan), 0.0])
    vdir = np.array([-np.sin(raan) * np.cos(inc), np.cos(raan) * np.cos(inc), np.sin(inc)])
    return np.concatenate([r, v_p * vdir])


def elements(rv):
    """(a, e, inc, RAAN) from a state vector — JAX, differentiable."""
    r, v = rv[:3], rv[3:]
    rn = jnp.sqrt(r @ r); vn2 = v @ v
    a = 1.0 / (2.0 / rn - vn2 / MU)
    h = jnp.cross(r, v); hn = jnp.sqrt(h @ h)
    inc = jnp.arccos(jnp.clip(h[2] / hn, -1.0, 1.0))
    n = jnp.stack([-h[1], h[0], 0.0])            # node = z x h
    nn = jnp.sqrt(n @ n)
    raan = jnp.arctan2(n[1], n[0]) % (2 * jnp.pi)
    e_vec = ((vn2 - MU / rn) * r - (r @ v) * v) / MU
    e = jnp.sqrt(e_vec @ e_vec)
    return a, e, inc, raan, nn


# ---------- low-thrust control: smooth RTN-frame Fourier profile (few DOF) ----------
# The control is a low-thrust acceleration in the radial/tangential/normal frame, each
# axis a truncated Fourier series in normalized time. Tangential thrust modulates the
# semi-major axis (→ J2 drift rate); normal thrust rotates the orbit plane. The optimizer
# DISCOVERS which is cheaper — no dive/plane-change structure is baked in. ~36 DOF total.
def rtn_profile(coeffs, n, a_max):
    """coeffs (3,2,K) → per-step RTN accel (n,3), bounded to a_max via tanh."""
    K = coeffs.shape[2]
    tau = jnp.arange(n) / n                       # (n,) in [0,1)
    k = jnp.arange(K)
    ph = 2 * jnp.pi * jnp.outer(tau, k)           # (n,K)
    basis = jnp.stack([jnp.cos(ph), jnp.sin(ph)], axis=1)   # (n,2,K)
    raw = jnp.einsum("nck,ack->na", basis, coeffs)          # (n,3)  [axis order r,t,n]
    return a_max * jnp.tanh(raw)


def rollout_rtn(a_rtn, s0, dt):
    """Apply per-step impulse dv = R(rv)·a_rtn·dt (RTN→ECI) then coast one RK4 step."""
    def step(rv, a):
        r, v = rv[:3], rv[3:]
        rh = r / jnp.sqrt(r @ r)
        h = jnp.cross(r, v); nh = h / jnp.sqrt(h @ h)
        th = jnp.cross(nh, rh)
        dv = (a[0] * rh + a[1] * th + a[2] * nh) * dt
        rv = rv.at[3:].add(dv)
        return rk4(rv, dt), 0.0
    rvT, _ = lax.scan(step, s0, a_rtn)
    return rvT


def ang_wrap(d):
    return jnp.arctan2(jnp.sin(d), jnp.cos(d))


def orbit_normal(inc, raan):
    return jnp.stack([jnp.sin(inc) * jnp.sin(raan),
                      -jnp.sin(inc) * jnp.cos(raan), jnp.cos(inc)])


def plane_change_dv(dg, a0, r_boost):
    """Δv to rotate the orbit plane by angle dg at semi-major a0, taking the cheaper of:
      - single impulse: 2 v_c0 sin(dg/2)   (optimal for small dg)
      - 3-burn BI-ELLIPTIC: raise apoapsis to r_boost, rotate where v is small, return
        (optimal above ~49°). J2-blind either way.
    Differentiable in dg; the min steel-mans the baseline at LARGE node changes."""
    vc0 = jnp.sqrt(MU / a0)
    dv_single = 2.0 * vc0 * jnp.sin(dg / 2.0)
    at = 0.5 * (a0 + r_boost)
    v_p = jnp.sqrt(MU * (2.0 / a0 - 1.0 / at))
    v_a = jnp.sqrt(MU * (2.0 / r_boost - 1.0 / at))
    dv_be = 2.0 * jnp.abs(v_p - vc0) + 2.0 * v_a * jnp.sin(dg / 2.0)
    return jnp.minimum(dv_single, dv_be)


def cleanup_dv(a, e, inc, raan, tgt, r_boost=42164.0):
    """Impulsive Δv estimate to correct the terminal residual to the EXACT target orbit
    (a0, e0, inc0, raan_t). Prices any shortfall so the comparison is fair:
      - energy/altitude: |v_circ(a) − v_circ(a0)|  (two-burn Hohmann cost to first order
        in the radius residual; exact to O((Δa/a)²))
      - eccentricity:    v_circ(a)·|e − e0|  (impulse to restore e0)
      - plane (i+RAAN):  min(single-impulse, bi-elliptic) plane change over Δγ = angle
        between orbit normals — steel-manned so large-Δγ residuals aren't over-priced.
    Summed sequentially (no burn-sharing credited)."""
    a0, e0, inc0, raan_t = tgt
    vc0 = jnp.sqrt(MU / a0); vcf = jnp.sqrt(MU / a)
    dv_a = jnp.sqrt((vcf - vc0) ** 2 + 1e-12)
    dv_e = vcf * jnp.abs(e - e0)
    cg = jnp.clip(orbit_normal(inc, raan) @ orbit_normal(inc0, raan_t), -1.0, 1.0)
    hav = jnp.sqrt(jnp.maximum(2.0 * (1.0 - cg), 0.0) + 1e-12)   # = 2 sin(Δγ/2), smooth at 0
    dgamma = 2.0 * jnp.arcsin(jnp.clip(hav / 2.0, 0.0, 1.0))     # Δγ, gradient-safe near 0
    dv_plane = plane_change_dv(dgamma, a0, r_boost)
    return dv_a + dv_e + dv_plane


def objective(coeffs, s0, dt, n, a_max, tgt, r_boost=42164.0):
    """True total mission Δv = low-thrust Δv + impulsive cleanup to the exact target.
    No penalty weights: every residual is priced, so the optimizer minimizes the real
    quantity and trades J2-drift shaping against the plane-change it avoids."""
    a_rtn = rtn_profile(coeffs, n, a_max)
    rvT = rollout_rtn(a_rtn, s0, dt)
    a, e, inc, raan, _ = elements(rvT)
    dv_lt = jnp.sum(jnp.sqrt(jnp.sum(a_rtn ** 2, axis=1) + 1e-20)) * dt   # ∫|a|dt
    dv_cl = cleanup_dv(a, e, inc, raan, tgt, r_boost)
    return dv_lt + dv_cl, (dv_lt, dv_cl, a, e, inc, raan)


# ---------- Adam ----------
def adam(grad_fn, x0, steps, lr, log_every=100, clip=1.0, lr_final_frac=0.02):
    x = x0; m = np.zeros_like(x0); vv = np.zeros_like(x0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x, best_loss = x0, np.inf
    for t in range(1, steps + 1):
        (loss, aux), g = grad_fn(x)
        if float(loss) < best_loss:                 # track BEFORE the step: best_x must be
            best_loss, best_x = float(loss), x.copy()   # the x that PRODUCED best_loss
        g = np.asarray(g)
        gn = np.sqrt(np.sum(g * g))                 # global-norm gradient clipping
        if gn > clip:
            g = g * (clip / gn)
        m = b1 * m + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = vv / (1 - b2 ** t)
        lr_t = lr * (lr_final_frac + (1 - lr_final_frac)
                     * 0.5 * (1 + np.cos(np.pi * t / steps)))   # cosine LR decay
        x = x - lr_t * mh / (np.sqrt(vh) + eps)
        if t % log_every == 0 or t == 1:
            dv_lt, dv_cl, a, e, inc, raan = [float(z) for z in aux]
            print(f"  it{t:4d} totΔv={float(loss):6.3f} (lt={dv_lt:.3f} clean={dv_cl:.3f}) "
                  f"a={a:7.1f} e={e:.4f} i={np.degrees(inc):6.2f} raan={np.degrees(raan):7.3f}")
    return best_x, best_loss


# ---------- passive-J2 baseline (best stopping time) ----------
def passive_best(s0, dt, n, tgt, r_boost):
    """Best PASSIVE-J2 cost: coast (zero control) and let J2 drift the node; the cheapest
    strategy is to stop whenever the cleanup-to-target is minimal (handles overshoot — if
    free drift reaches the target mid-coast, stop there for ~0). Running min over t∈[0,T]
    (carried in the scan, so no n-length array is materialised for long horizons)."""
    cl0 = cleanup_dv(*elements(s0)[:4], tgt, r_boost)          # include t=0
    def step(carry, _):
        rv, best = carry
        rv2 = rk4(rv, dt)
        cl = cleanup_dv(*elements(rv2)[:4], tgt, r_boost)
        return (rv2, jnp.minimum(best, cl)), None
    (_, best), _ = lax.scan(step, (s0, cl0), None, length=n)
    return float(best)


# ---------- scenarios ----------
def scenario(args):
    """Build the start state, target tuple, and step count from args (circular or eccentric).
    --alt is the PERIAPSIS altitude (kept fixed & valid across e0); a0 grows with e0 so
    periapsis never drops sub-surface. Reduces to circular a0=R+alt at e0=0."""
    a0 = (R_BODY + args.alt) / (1.0 - args.e0)
    s0 = jnp.asarray(orbit_state(a0, args.e0, args.inc, 0.0))
    # J2 regresses the node westward for prograde (i<90°), eastward for retrograde. Orient
    # the target ΔΩ along that direction so --draan is a MAGNITUDE and passive drift moves
    # TOWARD the target (fighting J2 would collapse the passive baseline to t=0).
    draan_signed = -np.sign(np.cos(np.radians(args.inc))) * abs(args.draan)
    tgt = (float(a0), float(args.e0), float(np.radians(args.inc)),
           float(np.radians(draan_signed)))
    n = round(args.days * DAY / args.dt)
    return a0, s0, tgt, n, draan_signed


def blind_plane_dv(a0, inc_deg, draan_deg, r_boost):
    """J2-blind plane-change baseline at semi-major a0: min(single-impulse, bi-elliptic)."""
    inc, dO = np.radians(inc_deg), np.radians(draan_deg)
    cth = np.cos(inc) ** 2 + np.sin(inc) ** 2 * np.cos(dO)
    theta = float(np.arccos(np.clip(cth, -1, 1)))
    vc0 = np.sqrt(MU / a0)
    dv_single = 2 * vc0 * np.sin(theta / 2)
    at = 0.5 * (a0 + r_boost)                                   # pure 3-burn bi-elliptic
    v_p = np.sqrt(MU * (2.0 / a0 - 1.0 / at))
    v_a = np.sqrt(MU * (2.0 / r_boost - 1.0 / at))
    dv_be = 2 * abs(v_p - vc0) + 2 * v_a * np.sin(theta / 2)
    return min(dv_single, dv_be), np.degrees(theta), dv_single, dv_be


# ---------- ANALYTIC dive-drift-return baseline (the fair J2-AWARE optimum) ----------
def nodal_rate(a, inc):
    """Vallado secular nodal regression dΩ/dt (rad/s) for a circular orbit (e=0, p=a)."""
    n = np.sqrt(MU / a ** 3)
    return -1.5 * n * J2 * (R_BODY / a) ** 2 * np.cos(inc)


def hohmann_roundtrip(a0, a_low):
    """Δv and one-way transfer time for a Hohmann round trip a0↔a_low (circular↔circular)."""
    vc0 = np.sqrt(MU / a0); vcl = np.sqrt(MU / a_low)
    at = 0.5 * (a0 + a_low)
    v_apo = np.sqrt(MU * (2.0 / a0 - 1.0 / at))       # transfer speed at a0 radius
    v_peri = np.sqrt(MU * (2.0 / a_low - 1.0 / at))   # transfer speed at a_low radius
    down = abs(vc0 - v_apo) + abs(v_peri - vcl)
    return 2.0 * down, np.pi * np.sqrt(at ** 3 / MU)


def analytic_dive_best(a0, inc_deg, draan_deg, T, r_boost, h_floor=300.0, n_grid=400):
    """The fair J2-AWARE analytic optimum: min over dive altitude of
        Hohmann(a0↔h) round-trip  +  impulsive residual plane change for the RAAN the
        dive didn't drift away.
    The dive (immediately) → coast low for the remaining budget → return (at the end)
    maximises low-altitude dwell, hence drift. h=a0 recovers passive, so this is ≤ passive
    by construction. Drift during the short transfer arcs is priced at their mean a.
    Returns (best_total_Δv, best_alt_km, drift_deg, residual_deg)."""
    inc = np.radians(inc_deg)
    tgt = abs(np.radians(draan_deg))
    best = (np.inf, a0 - R_BODY, 0.0, np.degrees(tgt))
    for a_low in np.linspace(R_BODY + h_floor, a0, n_grid):
        rt, tau = hohmann_roundtrip(a0, a_low)
        at = 0.5 * (a0 + a_low)
        t_low = max(0.0, T - 2.0 * tau)
        drift = abs(nodal_rate(a_low, inc)) * t_low + abs(nodal_rate(at, inc)) * 2.0 * tau
        residual = max(0.0, tgt - drift)              # RAAN the plane change must cover
        cth = np.cos(inc) ** 2 + np.sin(inc) ** 2 * np.cos(residual)
        theta = float(np.arccos(np.clip(cth, -1, 1)))
        total = rt + float(plane_change_dv(theta, a0, r_boost))
        if total < best[0]:
            best = (total, a_low - R_BODY, np.degrees(drift), np.degrees(residual))
    return best


def check(args):
    print("=== R-L1: honesty guards + differentiability building block ===")
    a0, s0, tgt, n, draan = scenario(args)
    dt = args.dt

    # (1) eccentric-drift physics check: numeric nodal rate vs Vallado (1-e²)^-2 scaling
    def coast_raan(sv, steps):
        def st(rv, _):
            rv2 = rk4(rv, dt)
            return rv2, elements(rv2)[3]
        _, oms = lax.scan(st, sv, None, length=steps)
        return np.unwrap(np.concatenate([[float(elements(sv)[3])], np.asarray(oms)]))
    revs = 12
    n_orbit = 2 * np.pi * np.sqrt(a0 ** 3 / MU)
    steps = int(round(revs * n_orbit / dt))
    oms = coast_raan(s0, steps)
    rate_num = np.polyfit(np.arange(len(oms)) * dt, oms, 1)[0] * 86400 * 180 / np.pi
    incr = np.radians(args.inc)
    nmm = np.sqrt(MU / a0 ** 3)
    p = a0 * (1 - args.e0 ** 2)
    rate_val = -1.5 * nmm * J2 * (R_BODY / p) ** 2 * np.cos(incr) * 86400 * 180 / np.pi
    print(f"  e0={args.e0:.2f} a0={a0:.0f} i={args.inc:.1f}°: dΩ/dt numeric {rate_num:+.3f}°/day "
          f"vs Vallado(1-e²)⁻² {rate_val:+.3f}°/day (err {abs(rate_num-rate_val)/abs(rate_val)*100:.1f}%)")

    # (2) free-coast do-nothing cost + differentiability
    r_boost = args.rboost
    pas = passive_best(s0, dt, n, tgt, r_boost)
    print(f"  passive-J2 best cost over {args.days:.1f} d = {pas:.4f} km/s (do-nothing)")
    gfn = jax.jit(jax.value_and_grad(
        lambda x: objective(x, s0, dt, n, args.amax, tgt, r_boost), has_aux=True))
    (loss, aux), g = gfn(0.1 * jnp.ones((3, 2, args.harm)))
    gn = float(jnp.linalg.norm(g))
    print(f"  grad norm = {gn:.4e} (finite {np.isfinite(gn)}) n_steps={n} DOF={3*2*args.harm}")

    # (3) steel-manned J2-blind baseline: min(single, bi-elliptic) plane change
    blind, theta, dv_si, dv_be = blind_plane_dv(a0, args.inc, draan, r_boost)
    print(f"  target ΔΩ={draan:+.1f}° (oriented along J2 regression) θ={theta:.1f}°: "
          f"single {dv_si:.4f} / bi-ell(boost {r_boost:.0f}) {dv_be:.4f} → baseline {blind:.4f} km/s")


def run_starts(gfn, shape, args):
    """Multi-start Adam: zero init (passive basin) + N random; return best_x and the loss spread."""
    inits = [np.zeros(shape)] + [rng_normal(shape, i, args.init_std) for i in range(args.starts)]
    results = []
    for j, x0 in enumerate(inits):
        bx, bl = adam(gfn, x0, args.steps, args.lr, log_every=args.log_every, clip=args.clip)
        if not np.isfinite(bl):
            bl = np.inf                              # diverged start → never selected
        results.append((bl, bx))
        print(f"  start {j} ({'zero' if j == 0 else 'rand'}): best total Δv = {bl:.4f}")
    results.sort(key=lambda r: r[0])
    losses = [r[0] for r in results]
    return results[0][1], results[0][0], losses


def rng_normal(shape, seed, std):
    # deterministic per-start noise (Math.random/Date unavailable-safe: seed explicitly)
    return np.random.default_rng(1000 + seed).normal(0, std, size=shape)


def optimize(args):
    print("=== R-L: diff-sim policy vs passive-J2 (eccentric / ΔΩ sweep / multi-start) ===")
    a0, s0, tgt, n, draan = scenario(args)
    dt = args.dt; r_boost = args.rboost
    gfn = jax.jit(jax.value_and_grad(
        lambda x: objective(x, s0, dt, n, args.amax, tgt, r_boost), has_aux=True))
    shape = (3, 2, args.harm)
    print(f"  scenario: a0={a0:.0f} (alt {args.alt:.0f}) e0={args.e0:.2f} i={args.inc:.1f}° "
          f"ΔΩ={draan:+.0f}° budget {args.days:.1f} d, n={n}, DOF={3*2*args.harm}, "
          f"starts={args.starts+1}")
    best_x, best_loss, losses = run_starts(gfn, shape, args)

    total, aux = objective(jnp.asarray(best_x), s0, dt, n, args.amax, tgt, r_boost)
    dv_lt, dv_cl, a, e, inc_f, raan = [float(z) for z in aux]
    dv_total = float(total)
    blind, theta, dv_si, dv_be = blind_plane_dv(a0, args.inc, draan, r_boost)
    passive = passive_best(s0, dt, n, tgt, r_boost)
    raan_err = abs(np.degrees(float(ang_wrap(jnp.asarray(raan - tgt[3])))))
    print("\n  --- VERIFY (best of multi-start) ---")
    print(f"  reached: a={a:.1f} (tgt {a0:.1f}) e={e:.5f} (tgt {args.e0:.2f}) "
          f"i={np.degrees(inc_f):.3f}° (tgt {args.inc:.1f}) RAAN={np.degrees(raan):.3f}° "
          f"(tgt {draan:+.1f}, residual {raan_err:.3f}°)")
    print(f"  policy total Δv = {dv_total:.4f} km/s (lt {dv_lt:.4f} + cleanup {dv_cl:.4f})")
    print(f"  start-spread total Δv across {len(losses)} starts: "
          f"min {min(losses):.4f} / max {max(losses):.4f}")
    print(f"  baseline 1 — J2-BLIND = {blind:.4f} km/s (single {dv_si:.3f} / bi-ell {dv_be:.3f})")
    print(f"  baseline 2 — PASSIVE-J2 (best stop) = {passive:.4f} km/s")
    r_blind = dv_total / blind; r_pass = dv_total / passive
    print(f"  → vs J2-blind:  ratio {r_blind:.3f}  [{'BEAT' if r_blind < 1 else 'no beat'}]")
    print(f"  → vs PASSIVE-J2: ratio {r_pass:.3f}  [{'BEAT' if r_pass < 1 else 'no beat'}] "
          f"(GENUINE test)")


def analytic(args):
    """R-M1: pit the ANALYTIC dive-drift-return optimum against the verified diff-sim
    policy and passive-J2 at the three frontier points. If analytic-dive ≈ policy, the
    'beat' is REDISCOVERY of a known operational strategy, not a novel diff-sim find."""
    print("=== Build M R-M1: analytic dive-drift vs diff-sim policy vs passive-J2 ===")
    r_boost = args.rboost; inc = args.inc
    # (ΔΩ°, budget days, dt, VERIFIED Build L policy total Δv km/s)
    pts = [(30.0, 8.0, 60.0, 0.540), (60.0, 16.0, 120.0, 0.548), (90.0, 24.0, 120.0, 0.571)]
    print(f"  scenario: alt {args.alt:.0f} km circular, i={inc:.1f}°, r_boost={r_boost:.0f} km")
    for draan, days, dt, pol in pts:
        a0 = R_BODY + args.alt
        T = days * DAY
        n = round(T / dt)
        draan_s = -np.sign(np.cos(np.radians(inc))) * abs(draan)
        s0 = jnp.asarray(orbit_state(a0, 0.0, inc, 0.0))
        tgt = (float(a0), 0.0, float(np.radians(inc)), float(np.radians(draan_s)))
        passive = passive_best(s0, dt, n, tgt, r_boost)
        adv, alt_km, drift, resid = analytic_dive_best(a0, inc, draan, T, r_boost)
        print(f"\n  ΔΩ={draan:.0f}° over {days:.0f} d:")
        print(f"    diff-sim policy   {pol:.3f} km/s   (Build L, verified)")
        print(f"    ANALYTIC dive     {adv:.3f} km/s   (dive to {alt_km:.0f} km, "
              f"drift {drift:.1f}°, residual {resid:.1f}°)")
        print(f"    passive-J2        {passive:.3f} km/s")
        tag = "MATCH (rediscovery)" if abs(pol - adv) / adv < 0.10 else \
              ("policy BEATS analytic" if pol < adv else "policy LOSES to analytic")
        print(f"    → policy/analytic = {pol/adv:.3f}  [{tag}];  "
              f"analytic/passive = {adv/passive:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--analytic", action="store_true")
    ap.add_argument("--alt", type=float, default=1500.0)   # altitude defining a0=R+alt
    ap.add_argument("--e0", type=float, default=0.0)        # start (and target) eccentricity
    ap.add_argument("--inc", type=float, default=51.6)
    ap.add_argument("--draan", type=float, default=30.0)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--harm", type=int, default=8)      # Fourier harmonics per axis
    ap.add_argument("--amax", type=float, default=2e-5)  # max thrust accel km/s²
    ap.add_argument("--rboost", type=float, default=42164.0)  # bi-elliptic boost radius (km)
    ap.add_argument("--starts", type=int, default=3)    # random restarts (+1 zero-init)
    ap.add_argument("--init-std", type=float, default=0.2)   # random-start coeff std
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    if args.check:
        check(args)
    if args.optimize:
        optimize(args)
    if args.analytic:
        analytic(args)


if __name__ == "__main__":
    main()
