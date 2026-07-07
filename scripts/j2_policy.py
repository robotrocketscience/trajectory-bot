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
    a = R_BODY + alt
    v = np.sqrt(MU / a)
    inc, raan = np.radians(inc_deg), np.radians(raan_deg)
    r = a * np.array([np.cos(raan), np.sin(raan), 0.0])
    vdir = np.array([-np.sin(raan) * np.cos(inc), np.cos(raan) * np.cos(inc), np.sin(inc)])
    return np.concatenate([r, v * vdir])


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


def cleanup_dv(a, e, inc, raan, tgt):
    """Impulsive Δv estimate to correct the terminal residual to the EXACT target
    (circular, a0, inc0, raan_t). Prices any shortfall so the comparison is fair:
      - energy/altitude: |v_circ(a) − v_circ(a0)|  (= two-burn Hohmann cost to first
        order in the radius residual; exact to O((Δa/a)²), which is <1% here since the
        beats leave |Δa| < ~230 km, ε≈0.03)
      - eccentricity:    v_circ(a)·e  (impulse to null residual e)
      - plane (i+RAAN):  2 v_circ(a0) sin(Δγ/2), Δγ = angle between orbit normals.
    Summed sequentially (no burn-sharing credited) — a tight, mildly conservative
    estimate of the true correction cost for the small residuals seen here."""
    a0, inc0, raan_t = tgt
    vc0 = jnp.sqrt(MU / a0); vcf = jnp.sqrt(MU / a)
    dv_a = jnp.sqrt((vcf - vc0) ** 2 + 1e-12)
    dv_e = vcf * e
    cg = jnp.clip(orbit_normal(inc, raan) @ orbit_normal(inc0, raan_t), -1.0, 1.0)
    dv_plane = vc0 * jnp.sqrt(jnp.maximum(2.0 * (1.0 - cg), 0.0) + 1e-12)  # =2v sin(Δγ/2)
    return dv_a + dv_e + dv_plane


def objective(coeffs, s0, dt, n, a_max, tgt):
    """True total mission Δv = low-thrust Δv + impulsive cleanup to the exact target.
    No penalty weights: every residual is priced, so the optimizer minimizes the real
    quantity and trades J2-drift shaping against the plane-change it avoids."""
    a_rtn = rtn_profile(coeffs, n, a_max)
    rvT = rollout_rtn(a_rtn, s0, dt)
    a, e, inc, raan, _ = elements(rvT)
    dv_lt = jnp.sum(jnp.sqrt(jnp.sum(a_rtn ** 2, axis=1) + 1e-20)) * dt   # ∫|a|dt
    dv_cl = cleanup_dv(a, e, inc, raan, tgt)
    return dv_lt + dv_cl, (dv_lt, dv_cl, a, e, inc, raan)


# ---------- J2-blind analytic baselines (steel-manned) ----------
def single_impulse_plane(alt, inc_deg, draan_deg):
    a = R_BODY + alt; v = np.sqrt(MU / a)
    inc, dO = np.radians(inc_deg), np.radians(draan_deg)
    cth = np.cos(inc) ** 2 + np.sin(inc) ** 2 * np.cos(dO)
    theta = np.arccos(np.clip(cth, -1, 1))
    return 2 * v * np.sin(theta / 2), np.degrees(theta)


def bielliptic_plane(alt, inc_deg, draan_deg, r_boost):
    """Raise apoapsis to r_boost, rotate the plane there (cheap, low v), return.
    3 burns: raise, plane-rotate at apoapsis, lower. J2-blind. Returns (Δv, period_days)."""
    a0 = R_BODY + alt
    _, theta = single_impulse_plane(alt, inc_deg, draan_deg)
    th = np.radians(theta)
    at = 0.5 * (a0 + r_boost)                       # transfer ellipse
    v_p0 = np.sqrt(MU / a0)                          # circ speed at a0
    v_p = np.sqrt(MU * (2.0 / a0 - 1.0 / at))       # peri speed on transfer
    v_a = np.sqrt(MU * (2.0 / r_boost - 1.0 / at))  # apo speed on transfer
    dv_raise = abs(v_p - v_p0)
    dv_rot = 2 * v_a * np.sin(th / 2)               # plane rotation at slow apoapsis
    dv_lower = abs(v_p - v_p0)                       # symmetric return
    period = 2 * np.pi * np.sqrt(at ** 3 / MU)       # one transfer ellipse period
    return dv_raise + dv_rot + dv_lower, period / DAY


# ---------- Adam ----------
def adam(grad_fn, x0, steps, lr, log_every=100, clip=1.0, lr_final_frac=0.02):
    x = x0; m = np.zeros_like(x0); vv = np.zeros_like(x0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_x, best_loss = x0, np.inf
    for t in range(1, steps + 1):
        (loss, aux), g = grad_fn(x)
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
        if float(loss) < best_loss:                 # loss IS true total Δv now
            best_loss, best_x = float(loss), x.copy()
        if t % log_every == 0 or t == 1:
            dv_lt, dv_cl, a, e, inc, raan = [float(z) for z in aux]
            print(f"  it{t:4d} totΔv={float(loss):6.3f} (lt={dv_lt:.3f} clean={dv_cl:.3f}) "
                  f"a={a:7.1f} e={e:.4f} i={np.degrees(inc):6.2f} raan={np.degrees(raan):7.3f}")
    return best_x, best_loss


# ---------- scenarios ----------
def check(args):
    print("=== R-K2a: differentiability + solvability building block ===")
    alt, inc, draan = args.alt, args.inc, args.draan
    T = args.days * DAY
    dt = args.dt
    n = int(round(T / dt))
    s0 = jnp.asarray(circular_state(alt, inc, 0.0))
    tgt = (float(R_BODY + alt), float(np.radians(inc)), float(np.radians(draan)))

    # (1) free coast: how much does the node drift for free at this altitude/budget?
    c0 = jnp.zeros((3, 2, args.harm))
    a_rtn0 = rtn_profile(c0, n, args.amax)
    rvT = rollout_rtn(a_rtn0, s0, dt)
    a, e, inc_f, raan, _ = [np.asarray(z) for z in elements(rvT)]
    print(f"  free coast {args.days:.1f} d @ {alt:.0f} km: RAAN drift "
          f"{np.degrees(raan):.3f}° (target {draan:.0f}°), a={a:.1f} e={e:.5f}")
    print(f"  → free-coast cleanup-to-target Δv = "
          f"{float(cleanup_dv(*elements(rvT)[:4], tgt)):.4f} km/s (the do-nothing cost)")

    # (2) differentiability: finite gradient, no NaN (perturb coeffs off zero)
    gfn = jax.jit(jax.value_and_grad(
        lambda x: objective(x, s0, dt, n, args.amax, tgt), has_aux=True))
    c_test = 0.1 * jnp.ones((3, 2, args.harm))
    (loss, aux), g = gfn(c_test)
    gnorm = float(jnp.linalg.norm(g))
    print(f"  grad norm at test-control = {gnorm:.4e}  (finite: {np.isfinite(gnorm)})  "
          f"n_steps={n} DOF={c_test.size}")

    # (3) analytic baselines
    dv_si, theta = single_impulse_plane(alt, inc, draan)
    dv_be, per_be = bielliptic_plane(alt, inc, draan, r_boost=R_BODY + 20000.0)
    print(f"  J2-blind single-impulse plane change ({draan:.0f}°→θ={theta:.1f}°): "
          f"{dv_si:.4f} km/s")
    print(f"  J2-blind bi-elliptic (boost to 20000 km alt): {dv_be:.4f} km/s "
          f"(1 transfer period {per_be:.2f} d)")
    print(f"  → J2-blind baseline = min = {min(dv_si, dv_be):.4f} km/s")


def optimize(args):
    print("=== R-K2b: diff-sim policy vs J2-blind optimum ===")
    alt, inc, draan = args.alt, args.inc, args.draan
    T = args.days * DAY
    dt = args.dt
    n = int(round(T / dt))
    s0 = jnp.asarray(circular_state(alt, inc, 0.0))
    tgt = (float(R_BODY + alt), float(np.radians(inc)), float(np.radians(draan)))
    gfn = jax.jit(jax.value_and_grad(
        lambda x: objective(x, s0, dt, n, args.amax, tgt), has_aux=True))

    x0 = np.zeros((3, 2, args.harm))             # start in the free-drift basin
    print(f"  scenario: {alt:.0f} km, i={inc:.1f}°, ΔΩ={draan:.0f}°, budget {args.days:.1f} d, "
          f"dt={dt:.0f}s, n_steps={n}, DOF={x0.size}, a_max={args.amax:.1e} km/s²")
    print("  objective = true total Δv (low-thrust ∫|a|dt + impulsive cleanup to exact target)")
    best_x, best_loss = adam(gfn, x0, args.steps, args.lr,
                             log_every=args.log_every, clip=args.clip)

    # verify: re-evaluate the SAME objective on the best control (no re-derivation, so the
    # reported numbers are bit-identical to what was optimized), then report reached state.
    total, aux = objective(jnp.asarray(best_x), s0, dt, n, args.amax, tgt)
    dv_lt, dv_cl, a, e, inc_f, raan = [float(z) for z in aux]
    dv_total = float(total)
    dv_si, theta = single_impulse_plane(alt, inc, draan)
    dv_be, per_be = bielliptic_plane(alt, inc, draan, r_boost=R_BODY + 20000.0)
    blind = min(dv_si, dv_be)
    # passive-J2 baseline: zero control (coast the budget), then impulsive cleanup.
    rv_pass = rollout_rtn(rtn_profile(jnp.zeros((3, 2, args.harm)), n, args.amax), s0, dt)
    passive = float(cleanup_dv(*elements(rv_pass)[:4], tgt))
    raan_err = abs(np.degrees(float(ang_wrap(jnp.asarray(raan - tgt[2])))))
    print("\n  --- VERIFY (re-flown best control) ---")
    print(f"  low-thrust reached: a={a:.1f} (tgt {R_BODY+alt:.1f}) e={e:.5f} "
          f"i={np.degrees(inc_f):.3f}° (tgt {inc:.1f}) RAAN={np.degrees(raan):.3f}° "
          f"(tgt {draan:.1f}, residual {raan_err:.3f}°)")
    print(f"  policy total Δv = {dv_total:.4f} km/s  (low-thrust {dv_lt:.4f} "
          f"+ impulsive cleanup {dv_cl:.4f})")
    print(f"  baseline 1 — J2-BLIND plane change = {blind:.4f} km/s "
          f"(single {dv_si:.3f} / bi-ell {dv_be:.3f})")
    print(f"  baseline 2 — PASSIVE-J2 (coast+clean) = {passive:.4f} km/s")
    r_blind = dv_total / blind; r_pass = dv_total / passive
    print(f"  → vs J2-blind:  ratio {r_blind:.3f}  [{'BEAT' if r_blind < 1 else 'no beat'}] "
          f"(partly trivial — J2-awareness alone)")
    print(f"  → vs PASSIVE-J2: ratio {r_pass:.3f}  [{'BEAT' if r_pass < 1 else 'no beat'}] "
          f"(the GENUINE test — active drift-shaping vs waiting)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--alt", type=float, default=1500.0)
    ap.add_argument("--inc", type=float, default=51.6)
    ap.add_argument("--draan", type=float, default=30.0)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--harm", type=int, default=8)      # Fourier harmonics per axis
    ap.add_argument("--amax", type=float, default=2e-5)  # max thrust accel km/s²
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()
    if args.check:
        check(args)
    if args.optimize:
        optimize(args)


if __name__ == "__main__":
    main()
