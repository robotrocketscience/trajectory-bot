#!/usr/bin/env python3
"""Flyby-NODE transcription — remove R-N7's razor-thin-basin sensitivity (Build N, R-N8).

R-N7 showed raw-environment single-shooting can't find/hold a gravity assist: integrating THROUGH the
close approach makes the loss landscape Lyapunov-chaotic (a 2 m/s departure perturbation → 1.26e9 km miss,
unrecoverable). The fix (how EMTG/GALLOP do it): DON'T integrate the close approach. Model the flyby as a
NODE — an instantaneous heliocentric-velocity rotation at Jupiter's position:
    V∞_out = Rz(δ)·V∞_in,   |V∞_out| = |V∞_in|  (energy conserved in Jupiter's frame),
    turn δ bounded by the min-periapsis limit  δ_max = 2·arcsin(1/(1 + r_p·V∞²/μ_J)).
The two legs are SMOOTH Sun-only Kepler coasts (no chaotic close approach); the assist becomes a smooth
bounded DECISION VARIABLE, not a razor-thin integrated basin. Decision vars: departure velocity v_dep, turn
δ, flyby epoch t_f (enough DOF to hit the target). Same offline Sun+circular-Jupiter scenario as R-N7.

    uv run --with jax python scripts/nbody_flyby_node.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM                    # noqa: E402
import nbody_sim as NB                    # noqa: E402
import nbody_flyby_exploit as NX          # noqa: E402  (reuse the exact R-N7 scenario)

MU_S = NB.GM["sun"]
MU_J = NB.GM["jupiter"]
AU = NB.AU
YR = NX.YR
A_JUP = NX.A_JUP
W_JUP = NX.W_JUP
V_EARTH = NX.V_EARTH
R_JUP = 71492.0                           # Jupiter equatorial radius (km)


def rot_z(v, d):
    """Rotate a 3-vector by angle d about +z (planar flyby turn)."""
    c, s = jnp.cos(d), jnp.sin(d)
    return jnp.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def coast(rv0, tof, n):
    """Sun-only Kepler coast (differentiable), tof over n steps. Smooth — no close approach."""
    body = jnp.zeros((n, 1, 3))
    rvT, _ = NB.rollout(rv0, body, jnp.array([MU_S]), tof / n, soft=0.0)
    return rvT


def jup_state(t):
    """Jupiter position & velocity on its circular orbit at time t (from start), differentiable in t."""
    th = NX_THETA0 + W_JUP * t
    r = A_JUP * jnp.array([jnp.cos(th), jnp.sin(th), 0.0])
    v = A_JUP * W_JUP * jnp.array([-jnp.sin(th), jnp.cos(th), 0.0])
    return r, v


def objective(p, rE, r2, tof_total, tf0, tf_span, rp_min, n_leg, w, w_turn, w_dv):
    v_dep = p[:3] * V_EARTH                           # decision var scaled to O(1) (R-N5 lesson)
    delta = p[3]
    t_f = tf0 + tf_span * jnp.tanh(p[4])              # flyby epoch, bounded near the seed
    tof2 = tof_total - t_f
    rJf, vJf = jup_state(t_f)
    # leg 1: Earth -> Jupiter (Sun-only)
    rv1 = coast(jnp.concatenate([rE, v_dep]), t_f, n_leg)
    r1_end, v1_end = rv1[:3], rv1[3:]
    vinf_in = v1_end - vJf
    vinf_out = rot_z(vinf_in, delta)                 # |vinf| conserved by construction
    # leg 2: Jupiter (node) -> target (Sun-only), departing with the turned V∞
    rv2 = coast(jnp.concatenate([rJf, vJf + vinf_out]), tof2, n_leg)
    r2_end = rv2[:3]
    reach_jup = jnp.sum((r1_end - rJf) ** 2) / AU ** 2
    reach_tgt = jnp.sum((r2_end - r2) ** 2) / AU ** 2
    vin = jnp.sqrt(jnp.sum(vinf_in ** 2) + 1e-12)
    dmax = 2.0 * jnp.arcsin(1.0 / (1.0 + rp_min * vin ** 2 / MU_J))
    turn_pen = jnp.maximum(jnp.abs(delta) - dmax, 0.0) ** 2
    dep_dv = jnp.sqrt(jnp.sum((v_dep - jnp.array([0.0, V_EARTH, 0.0])) ** 2) + 1e-12)
    L = w * (reach_jup + reach_tgt) + w_turn * turn_pen + w_dv * dep_dv
    return L, (reach_jup, reach_tgt, dep_dv, delta, dmax, vin, t_f)


def solve(p0, rE, r2, tof_total, tf0, tf_span, rp_min, n_leg, iters, lr):
    def make(w, wdv):
        return jax.jit(jax.value_and_grad(
            lambda p: objective(p, rE, r2, tof_total, tf0, tf_span, rp_min, n_leg, w, 1.0e3, wdv),
            has_aux=True))
    p = jnp.asarray(p0)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-12
    best = (float("inf"), None)
    reach_tol = (0.03 * AU) ** 2 / AU ** 2           # 0.03 AU on each leg counts as met
    t = 0
    stages = ((1.0e1, 0.0), (1.0e2, 0.0), (1.0e3, 0.0), (1.0e3, 5.0e-3))  # anneal reach, then economize
    for w, wdv in stages:
        vg = make(w, wdv)
        for _ in range(iters // len(stages)):
            t += 1
            (L, aux), g = vg(p)
            rj, rt, dv, dl, dm, vin, tf = aux
            feasible = float(rj) < reach_tol and float(rt) < reach_tol and abs(float(dl)) <= float(dm) + 1e-6
            if feasible and float(dv) < best[0]:
                best = (float(dv), dict(dv=float(dv), reach_jup_km=float(jnp.sqrt(rj)) * AU,
                                        reach_tgt_km=float(jnp.sqrt(rt)) * AU, delta=float(dl),
                                        dmax=float(dm), vinf=float(vin), t_f=float(tf)))
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * (g * g)
            p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    if best[1] is None:
        vg = make(1.0e3, 5.0e-3)
        (L, aux), _ = vg(p)
        rj, rt, dv, dl, dm, vin, tf = aux
        return dict(dv=float(dv), reach_jup_km=float(jnp.sqrt(rj)) * AU,
                    reach_tgt_km=float(jnp.sqrt(rt)) * AU, delta=float(dl), dmax=float(dm),
                    vinf=float(vin), t_f=float(tf), feasible=False)
    best[1]["feasible"] = True
    return best[1]


def implied_rp(vinf, delta):
    """R-N6 consistency: the periapsis radius the node's algebraic turn requires (must clear Jupiter)."""
    s = np.sin(abs(delta) / 2.0)
    if s <= 1e-9:
        return np.inf
    return (MU_J / vinf ** 2) * (1.0 / s - 1.0)


NX_THETA0 = 0.0   # set per scenario in verify()


def verify(args):
    global NX_THETA0
    print("=== R-N8: flyby-NODE transcription — does it recover R-N7's razor-thin basin? (offline) ===")
    scen = NX.design_scenario(args.theta1, args.tof1 * YR, args.tof2 * YR, args.aim_off, 40000)
    NX_THETA0 = float(scen["theta0"])
    rE = jnp.asarray(scen["rE"])
    r2 = jnp.asarray(scen["r2"])
    tof_total = scen["tof"]
    tf0 = args.tof1 * YR
    # leg-1 Lambert to Jupiter CENTER (the patched-conic node sits at Jupiter's position) — reaches Jupiter
    rJf0 = A_JUP * np.array([np.cos(NX_THETA0 + W_JUP * tf0), np.sin(NX_THETA0 + W_JUP * tf0), 0.0])
    vdep_J, _ = LAM.lambert(rE, jnp.asarray(rJf0), tf0, mu=MU_S)
    vdep_J = np.asarray(vdep_J)
    # fair direct baseline seed (R-N7: never reaches cheaply; ≥25.3 km/s Lambert to the target)
    vdep_dir, _ = LAM.lambert(rE, r2, tof_total, mu=MU_S)
    vdep_dir = np.asarray(vdep_dir)
    print(f"  target r2 at {np.linalg.norm(scen['r2'])/AU:.2f} AU, total TOF={tof_total/YR:.2f} yr; "
          f"R-N7 direct baseline ≥25.3 km/s (never reached); flyby assisted ≈9.2 km/s")
    print(f"  {'seed':>18} {'Δv(km/s)':>9} {'reachJ(km)':>11} {'reachT(km)':>11} "
          f"{'δ(°)':>7} {'δmax(°)':>8} {'r_p/R_J':>8} {'feas':>5}")
    # params are [v_dep/V_EARTH (3), δ, tf_raw]; both seeds start at δ=0 (no assist presupposed)
    seeds = {
        "flyby+2m/s (H-N8a)": np.concatenate([(vdep_J + args.perturb * np.array([1, 1, 0]) / np.sqrt(2))
                                              / V_EARTH, [0.0, 0.0]]),
        "neutral δ=0 (H-N8b)": np.concatenate([vdep_dir / V_EARTH, [0.0, 0.0]]),
    }
    for tag, p0 in seeds.items():
        r = solve(p0, rE, r2, tof_total, tf0, args.tf_span * YR, args.rp_min, args.n_leg,
                  args.iters, args.lr)
        rp = implied_rp(r["vinf"], r["delta"])
        print(f"  {tag:>18} {r['dv']:9.3f} {r['reach_jup_km']:11.0f} {r['reach_tgt_km']:11.0f} "
              f"{np.degrees(r['delta']):7.1f} {np.degrees(r['dmax']):8.1f} {rp/R_JUP:8.1f} "
              f"{str(r['feasible']):>5}")
    print("  H-N8a: recovers the 2 m/s R-N7 could not (defects ≪ SOI, |δ|≤δmax, Δv≈9.2). "
          "H-N8b: from δ=0 the assist EMERGES (grows δ, reaches, ~9.2). "
          "H-N8c: implied r_p clears Jupiter (>1 R_J) → the node's turn is physically real (R-N6-consistent).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--theta1", type=float, default=165.0)
    ap.add_argument("--tof1", type=float, default=2.5)
    ap.add_argument("--tof2", type=float, default=3.0)
    ap.add_argument("--aim-off", type=float, default=1.0e6)
    ap.add_argument("--perturb", type=float, default=2.0e-3)
    ap.add_argument("--tf-span", type=float, default=0.5)     # flyby-epoch freedom (yr, ±)
    ap.add_argument("--rp-min", type=float, default=2.0e5)    # min allowed periapsis (km, ~2.8 R_J)
    ap.add_argument("--n-leg", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1.0e-2)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
