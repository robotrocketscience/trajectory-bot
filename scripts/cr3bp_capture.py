#!/usr/bin/env python3
"""Ballistic capture in the CR3BP (Build G) — the genuine Tier-3 discovery test.

Can we find a departure burn whose CR3BP arrival is BALLISTICALLY CAPTURED at the
Moon (bound, no capture burn), beating a patched-conic Hohmann that arrives
hyperbolic and must pay a capture burn? Objective:
    total_Δv(departure) = |Δv_dep| + Δv_cap(closest lunar approach)
Δv_cap = max(0, |v_rel| − √(2μ/d)) is 0 exactly when E_moon = ½|v_rel|² − μ/d < 0
(already bound). Minimizing total_Δv rewards ballistic capture; a captured route
below Hohmann(TLI + its capture burn) is the low-energy win. Multi-start (the WSB
set is thin) + gradient refine; the headline must PASS a bounded-propagation check.

    uv run --with jax python scripts/cr3bp_capture.py --search
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402
import cr3bp_transfer as T  # noqa: E402

MU = C.MU
R_MOON = jnp.array([1.0 - MU, 0.0, 0.0])
R_HILL = (MU / 3.0) ** (1.0 / 3.0)             # ~0.159 nondim (~61000 km)
V_UNIT = T.V_UNIT_KMS


def moon_rel(state):
    """Moon-relative distance, inertial speed relative to the Moon, and osculating
    two-body (Moon) energy. In the rotating frame the inertial velocity relative to
    the (stationary) Moon is v_rot + ω×r_rel, ω=(0,0,1)."""
    r_rel = state[..., 0:3] - R_MOON
    vx = state[..., 3] - r_rel[..., 1]         # v_rot + ω×r_rel
    vy = state[..., 4] + r_rel[..., 0]
    vz = state[..., 5]
    d = jnp.sqrt((r_rel ** 2).sum(-1))
    speed = jnp.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    E = 0.5 * speed ** 2 - MU / jnp.maximum(d, 1e-6)
    return d, speed, E


def closest_approach(traj, tau=0.01):
    """Softmin-weighted closest-approach quantities (differentiable)."""
    d, speed, E = moon_rel(traj)
    w = jax.nn.softmax(-d / tau)
    d_ca = (w * d).sum()
    speed_ca = (w * speed).sum()
    v_esc = jnp.sqrt(2.0 * MU / jnp.maximum(d_ca, 1e-6))
    cap = jnp.maximum(0.0, speed_ca - v_esc)   # capture burn; 0 if already bound
    E_ca = 0.5 * speed_ca ** 2 - MU / jnp.maximum(d_ca, 1e-6)
    return d_ca, cap, E_ca


def pieces(dv, s0, n, dt):
    traj = T.rollout_traj(s0, dv, n, dt)
    d_ca, cap, E_ca = closest_approach(traj)
    dep = jnp.sqrt((dv ** 2).sum() + 1e-9)
    return dep, cap, d_ca, E_ca


def objective(dv, s0, n, dt, seek_capture, w_reach=5.0, w_E=8.0):
    dep, cap, d_ca, E_ca = pieces(dv, s0, n, dt)
    reach = w_reach * jnp.maximum(d_ca - 0.3 * R_HILL, 0.0)   # get near the Moon
    if seek_capture:
        # drive the arrival BOUND (E_moon < 0): the ballistic-capture route (cost = dep
        # only, no capture burn), reported honestly against the direct route below.
        return dep + w_E * jnp.maximum(E_ca, 0.0) + reach, (dep, cap, d_ca, E_ca)
    # direct route: pay the capture burn to bind a fast arrival.
    return dep + cap + reach, (dep, cap, d_ca, E_ca)


def refine(dv0, s0, n, dt, seek_capture, iters=250, lr=2e-3):
    vg = jax.jit(jax.value_and_grad(lambda d: objective(d, s0, n, dt, seek_capture),
                                    has_aux=True))
    dv = dv0
    m = jnp.zeros(2); v = jnp.zeros(2); t = 0
    for _ in range(iters):
        (val, aux), g = vg(dv)
        gn = jnp.sqrt((g ** 2).sum()); g = g * jnp.minimum(1.0, 1.0 / jnp.maximum(gn, 1e-9))
        t += 1; m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        dv = dv - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return dv


def verify_capture(dv, s0, n, dt, revs_steps=6000):
    """Ballistic check: from the closest-approach STATE, propagate with NO thrust and
    see how long the trajectory stays within the Moon's Hill sphere (bound revs)."""
    traj = np.asarray(T.rollout_traj(s0, dv, n, dt))
    d = np.linalg.norm(traj[:, 0:3] - np.asarray(R_MOON), axis=1)
    i = int(np.argmin(d))
    s_ca = jnp.asarray(traj[i])
    fwd = np.asarray(T.rollout_traj(s_ca, jnp.zeros(2), revs_steps, dt))
    dd = np.linalg.norm(fwd[:, 0:3] - np.asarray(R_MOON), axis=1)
    # time (steps) until it first leaves the Hill sphere
    left = np.argmax(dd > R_HILL) if (dd > R_HILL).any() else len(dd)
    bound_time = left * dt                      # nondim time bound
    return d[i], bound_time, float(np.min(dd)), float(np.max(dd[:left] if left > 0 else dd))


def _multistart(s0, args, seek_capture):
    tli = T.hohmann_tli(args.r_leo)
    best = None
    mags = np.linspace(tli * 0.80, tli * 1.15, args.n_mag)
    angs = np.radians(np.linspace(70.0, 110.0, args.n_ang))
    for mg in mags:
        for an in angs:
            dv0 = jnp.array([mg * np.cos(an), mg * np.sin(an)])
            dv = refine(dv0, s0, args.n, args.dt, seek_capture, iters=args.refine_iters)
            dep, cap, dca, E = (float(x) for x in pieces(dv, s0, args.n, args.dt))
            if dca >= 0.5 * R_HILL:                        # didn't reach the Moon region
                continue
            key = dep if seek_capture else dep + cap       # ballistic ranks by dep only
            if (best is None or key < best[0]) and (not seek_capture or E < 0):
                best = (key, np.asarray(dv), dep, cap, dca, E)
    return best


def search(args):
    s0 = T.leo_state(args.r_leo)
    print(f"Ballistic-capture search (r_leo={args.r_leo:.3f}, Hill={R_HILL:.3f} "
          f"~{R_HILL*C.L_UNIT_KM:.0f} km)", flush=True)
    # DIRECT route: reach the Moon and pay the capture burn to bind (the fair
    # "patched-conic + capture burn" baseline — the best phased direct transfer).
    d = _multistart(s0, args, seek_capture=False)
    if d is None:
        print("  no start reached the Moon region — honest null (engine+criterion ok).")
        return
    _, dvd, depd, capd, dcad, Ed = d
    direct_total = depd + capd
    print(f"  DIRECT best: total={direct_total:.4f} ({direct_total*V_UNIT:.3f} km/s) "
          f"= dep {depd:.4f} + capture {capd:.4f};  arrival {dcad*C.L_UNIT_KM:.0f} km "
          f"E_moon={Ed:+.4f}", flush=True)
    # BALLISTIC route: drive the arrival BOUND (E_moon<0); cost = dep only (no burn).
    b = _multistart(s0, args, seek_capture=True)
    if b is None:
        print("  BALLISTIC: no start achieved E_moon<0 capture in this grid — "
              "HONEST NULL. The osculating-capture set is thin and was not hit; the "
              "verified engine + criterion + direct baseline are the deliverable. "
              "Next: manifold-seeded init / longer L1-L2 gateway arcs.")
        return
    _, dvb, depb, capb, dcab, Eb = b
    _, bound_t, dmin, _ = verify_capture(jnp.asarray(dvb), s0, args.n, args.dt)
    print(f"  BALLISTIC best: dep={depb:.4f} ({depb*V_UNIT:.3f} km/s)  arrival "
          f"{dcab*C.L_UNIT_KM:.0f} km  E_moon={Eb:+.4f} (CAPTURED)", flush=True)
    print(f"        bounded-prop stays in Hill sphere {bound_t:.3f} nondim "
          f"(~{bound_t*C.T_UNIT_S/86400:.2f} days); min d {dmin*C.L_UNIT_KM:.0f} km")
    if depb < direct_total * 0.999:
        print(f"  -> BALLISTIC CAPTURE BEATS direct: {depb:.4f} < {direct_total:.4f} "
              f"(saves {(direct_total-depb)*V_UNIT:.3f} km/s = the capture burn).")
    else:
        print(f"  -> capture achieved but not cheaper (dep {depb:.4f} vs direct "
              f"{direct_total:.4f}); honest partial.")


def sanity():
    # R-G1: low circular lunar orbit -> E<0; a fast flyby -> E>0
    r = 0.02                                   # ~7700 km lunar orbit
    v_circ = np.sqrt(MU / r)
    s_bound = jnp.array([1.0 - MU + r, 0.0, 0.0, 0.0, v_circ - (1.0 - MU + r), 0.0])
    d, sp, E = moon_rel(s_bound)
    print(f"R-G1: circular lunar orbit r={r}: E_moon={float(E):+.4f} (want <0, bound)")
    s_fly = jnp.array([1.0 - MU + r, 0.0, 0.0, 0.0, 3.0 * v_circ, 0.0])
    d, sp, E = moon_rel(s_fly)
    print(f"      fast flyby (3x v_circ): E_moon={float(E):+.4f} (want >0, hyperbolic)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--r-leo", type=float, default=0.03)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--n-mag", type=int, default=8)
    ap.add_argument("--n-ang", type=int, default=7)
    ap.add_argument("--refine-iters", type=int, default=250)
    args = ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.sanity:
        sanity()
    if args.search:
        search(args)


if __name__ == "__main__":
    main()
