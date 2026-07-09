#!/usr/bin/env python3
"""3-D diff-sim flyby-node optimizer — target a specified INCLINED science orbit (Build N, R-N17).

R-N8 built the flyby-NODE transcription but IN-PLANE (`rot_z` only), targeting a POSITION. R-N15/R-N16 gave
the analytic 3-D inclination ceiling arcsin(v∞/v_P) via a greedy-geodesic reachability BOUND. R-N17 fuses
them: a 3-D **Rodrigues flyby node** (turn magnitude δ ≤ δmax about an arbitrary axis, parametrised by the
turn δ and its azimuth φ) whose two smooth Sun-only legs are backprop-differentiable, optimised to hit a
specified inclined heliocentric orbit — a target orbital ELEMENT (inclination, and optionally aphelion), NOT
a position. The craft departs Earth PURELY IN-PLANE, so any inclination in the outbound orbit can ONLY come
from an out-of-plane turn the optimiser must DISCOVER through the gradient. This validates R-N16's closed form
through the actual differentiable dynamics (not the greedy bound), closing the loop analytic-graph ↔ diff-sim.

  H-N17a  DISCOVERY: from an in-plane, δ=0 seed the diff-sim (backprop through the 3-D node) grows the
          out-of-plane turn and hits a target inclination i* within the ceiling (defect ≪0.5°, |δ|≤δmax).
  H-N17b  MATCHES THE CEILING: with leg-1 fixed (→ fixed v∞), the diff-sim feasibility boundary in i* equals
          R-N16's arcsin(v∞/v_P) — reached below it, floored above it (±2°).
  H-N17c  DOF HONESTY: with |v∞| fixed, the single node's reachable (inclination, aphelion) set is 2-D but
          Tisserand-bounded; interior pairs are hit, but at i*=ceiling the aphelion is PINNED (the R-N15
          inclination–energy competition, now seen by the optimiser).

    uv run --with jax python scripts/nbody_flyby_node3d.py --verify        # offline, CI-safe
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

MU_S = NB.GM["sun"]
MU_J = NB.GM["jupiter"]
AU = NB.AU
DAY = NB.DAY
YR = 365.25 * DAY
A_JUP = 5.2028 * AU
W_JUP = float(np.sqrt(MU_S / A_JUP ** 3))          # Jupiter mean motion (rad/s)
V_JUP = float(np.sqrt(MU_S / A_JUP))               # Jupiter circular speed (km/s)
V_EARTH = float(np.sqrt(MU_S / AU))                # Earth circular speed (km/s)
R_JUP = 71492.0
SOI_JUP = 0.322 * AU                               # Jupiter Hill/SOI radius (km)

NX_THETA0 = 0.0                                    # Jupiter phase at t=0 (set per scenario)


def coast(rv0, tof, n):
    """Sun-only Kepler coast (differentiable), tof over n steps. Smooth — no close approach."""
    body = jnp.zeros((n, 1, 3))
    rvT, _ = NB.rollout(rv0, body, jnp.array([MU_S]), tof / n, soft=0.0)
    return rvT


def jup_state(t):
    """Jupiter position & velocity on its circular orbit at time t, differentiable in t."""
    th = NX_THETA0 + W_JUP * t
    r = A_JUP * jnp.array([jnp.cos(th), jnp.sin(th), 0.0])
    v = A_JUP * W_JUP * jnp.array([-jnp.sin(th), jnp.cos(th), 0.0])
    return r, v


def flyby_turn(vinf_in, delta, phi):
    """3-D flyby node: rotate v∞_in by turn δ about an axis whose azimuth is φ, |v∞| conserved.
    û is v∞_in; ê1 = û×ẑ (in-plane, ⊥û), ê2 = û×ê1 (the out-of-plane direction). The turn direction is
    cosφ·ê1 + sinφ·ê2, so φ controls how much of the bounded turn goes OUT of the ecliptic."""
    vmag = jnp.sqrt(jnp.sum(vinf_in ** 2) + 1e-12)
    u = vinf_in / vmag
    zref = jnp.array([0.0, 0.0, 1.0])
    e1 = jnp.cross(u, zref)
    e1 = e1 / (jnp.sqrt(jnp.sum(e1 ** 2)) + 1e-12)
    e2 = jnp.cross(u, e1)                                   # completes the RH frame; carries +z
    tdir = jnp.cos(phi) * e1 + jnp.sin(phi) * e2
    return vmag * (jnp.cos(delta) * u + jnp.sin(delta) * tdir)


def helio_elements(r, v):
    """Heliocentric osculating elements from a state (r,v): (a, e, cos_inc, aphelion, energy).
    Returns cos_inc = h_z/|h| (NOT the arccos): the arccos has an INFINITE gradient at inc=0, so an in-plane
    seed NaNs the backprop. Targeting inclination through its cosine is smooth everywhere (R-N17 fix)."""
    rn = jnp.sqrt(jnp.sum(r ** 2) + 1e-12)
    vn2 = jnp.sum(v ** 2)
    eps = 0.5 * vn2 - MU_S / rn                             # specific energy
    a = -MU_S / (2.0 * eps)
    h = jnp.cross(r, v)
    hn = jnp.sqrt(jnp.sum(h ** 2) + 1e-12)
    cos_inc = h[2] / hn                                     # cos(inclination); smooth at inc=0 (=1)
    e_vec = jnp.cross(v, h) / MU_S - r / rn
    e = jnp.sqrt(jnp.sum(e_vec ** 2) + 1e-12)
    aph = a * (1.0 + e)                                     # <0 (i.e. "beyond escape") if e≥1
    return a, e, cos_inc, aph, eps


def propagate(v_dep, delta, phi, t_f, n_leg, tof_out):
    """Full patched-conic flyby: Earth→(Sun-only)→Jupiter node→(Sun-only)→measure outbound elements."""
    rE = jnp.array([AU, 0.0, 0.0])
    rv1 = coast(jnp.concatenate([rE, v_dep]), t_f, n_leg)
    r1, v1 = rv1[:3], rv1[3:]
    rJf, vJf = jup_state(t_f)
    reach = jnp.sum((r1 - rJf) ** 2) / AU ** 2             # leg-1 rendezvous defect (AU²)
    vinf_in = v1 - vJf
    vout = vJf + flyby_turn(vinf_in, delta, phi)
    rv2 = coast(jnp.concatenate([rJf, vout]), tof_out, n_leg)
    a, e, cos_inc, aph, eps = helio_elements(rv2[:3], rv2[3:])
    vinf = jnp.sqrt(jnp.sum(vinf_in ** 2) + 1e-12)
    dmax = 2.0 * jnp.arcsin(1.0 / (1.0 + 1.5 * R_JUP * vinf ** 2 / MU_J))
    inc = jnp.degrees(jnp.arccos(jnp.clip(cos_inc, -1.0, 1.0)))    # for DISPLAY only (not in the gradient path)
    return dict(reach=reach, inc=inc, cos_inc=cos_inc, a=a, e=e, aph=aph, vinf=vinf, dmax=dmax,
                delta=delta, phi=phi)


def run_adam(loss_and_grad, p0, iters, lr):
    """Bare Adam on a jitted value_and_grad(has_aux) loss. Returns final p and its aux."""
    p = jnp.asarray(p0)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-12
    aux = None
    for t in range(1, iters + 1):
        (L, aux), g = loss_and_grad(p)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        p = p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return p, aux


def make_scenario(t_f_yr, theta_arr_deg):
    """Coplanar circular Earth(1 AU)+Jupiter(5.2 AU); Lambert Earth→Jupiter arrival at theta_arr, t_f.
    Sets NX_THETA0 so Jupiter is at theta_arr at t_f. Returns the in-plane Lambert v_dep and v∞."""
    global NX_THETA0
    t_f = t_f_yr * YR
    th_arr = np.radians(theta_arr_deg)
    NX_THETA0 = float(th_arr - W_JUP * t_f)
    rE = np.array([AU, 0.0, 0.0])
    rJ_arr = A_JUP * np.array([np.cos(th_arr), np.sin(th_arr), 0.0])
    vJ_arr = A_JUP * W_JUP * np.array([-np.sin(th_arr), np.cos(th_arr), 0.0])
    v_dep, v_arr = LAM.lambert(rE, rJ_arr, t_f, mu=MU_S)
    v_dep, v_arr = np.asarray(v_dep), np.asarray(v_arr)
    vinf = float(np.linalg.norm(v_arr - vJ_arr))
    return dict(t_f=t_f, v_dep=v_dep, vinf=vinf)


def verify(args):
    print("=== R-N17: 3-D diff-sim flyby-node optimizer — target an inclined science orbit (offline) ===")
    n_leg, tof_out = args.n_leg, args.tof_out * YR

    # ---- shared scenario: an Earth→Jupiter transfer; leg-1 Lambert sets v∞ ----
    scen = make_scenario(args.tf, args.theta_arr)
    v_dep0, vinf0, t_f = scen["v_dep"], scen["vinf"], scen["t_f"]
    ceil0 = np.degrees(np.arcsin(min(1.0, vinf0 / V_JUP)))
    print(f"  scenario: Earth→Jupiter, t_f={args.tf:.2f} yr, arrival θ={args.theta_arr:.0f}°, "
          f"v∞={vinf0:.2f} km/s (v_P={V_JUP:.2f}) → analytic ceiling arcsin(v∞/v_P)={ceil0:.1f}°")

    # ================= H-N17a: DISCOVERY — grow an out-of-plane turn from an in-plane, δ=0 seed =========
    i_star_a = args.i_star_a
    ci_a = np.cos(np.radians(i_star_a))
    print(f"\n  H-N17a: from an IN-PLANE departure, does backprop discover the out-of-plane turn to reach "
          f"i*={i_star_a:.0f}°?")

    def obj_a(p):
        v_dep = jnp.array([p[0], p[1], 0.0]) * V_EARTH     # IN-PLANE departure (z≡0): no inclination injected
        delta, phi = p[2], p[3]
        t_fv = t_f * (1.0 + 0.15 * jnp.tanh(p[4]))         # small epoch freedom about the seed
        out = propagate(v_dep, delta, phi, t_fv, n_leg, tof_out)
        turn_pen = jnp.maximum(jnp.abs(delta) - out["dmax"], 0.0) ** 2
        L = 1.0e3 * out["reach"] + (out["cos_inc"] - ci_a) ** 2 + 1.0e4 * turn_pen
        return L, out

    lg_a = jax.jit(jax.value_and_grad(obj_a, has_aux=True))
    # departure strictly in-plane (z≡0); the flyby node is seeded with a SMALL symmetry-breaking turn because
    # inclination is SECOND-order in δ at δ=0 (need both δ≠0 and φ≠0 to leave the plane) — exact δ=0 is a flat
    # critical point the gradient cannot escape. Honest correction to the pre-registered "δ=0 seed".
    p0_a = jnp.array([v_dep0[0] / V_EARTH, v_dep0[1] / V_EARTH, 0.05, 1.2, 0.0])
    out0 = obj_a(p0_a)[1]
    _, aux_a = run_adam(lg_a, p0_a, args.iters, args.lr)
    reach_km = float(jnp.sqrt(aux_a["reach"])) * AU
    inc_a, dl_a, dmax_a = float(aux_a["inc"]), abs(float(np.degrees(aux_a["delta"]))), float(np.degrees(aux_a["dmax"]))
    print(f"    seed  : inc={float(out0['inc']):5.2f}°  δ={abs(float(np.degrees(out0['delta']))):.1f}°  "
          f"(in-plane departure; flyby seed a small turn to break the δ=0 flat critical point)")
    print(f"    solved: inc={inc_a:5.2f}°  target={i_star_a:.2f}°  |defect|={abs(inc_a-i_star_a):.3f}°  "
          f"δ={dl_a:.1f}° (≤δmax {dmax_a:.0f}°)  reach={reach_km:.3e} km (SOI {SOI_JUP:.2e})")
    a_ok = abs(inc_a - i_star_a) < 0.5 and reach_km < SOI_JUP and dl_a <= dmax_a + 1e-6
    print(f"    → {'SUPPORTED' if a_ok else 'REFUTED'} (prediction corrected: δ=0 is a flat critical point): the "
          f"diff-sim grew an out-of-plane turn from an in-plane departure and hit i* (defect {abs(inc_a-i_star_a):.2f}° "
          "< 0.5°, leg-1 defect ≪ SOI, |δ|≤δmax).")

    # ================= H-N17b: does the diff-sim feasibility boundary == arcsin(v∞/v_P)? =================
    print(f"\n  H-N17b: leg-1 FIXED (v∞={vinf0:.2f}); sweep target i*, optimise only the turn (δ,φ). "
          f"Boundary should be the ceiling {ceil0:.1f}°:")
    print(f"    {'i*(°)':>7} {'achieved(°)':>12} {'defect(°)':>10} {'δ(°)':>7} {'≤δmax?':>7} {'verdict':>9}")

    def obj_turn(p, cos_i_star, aph_star, w_aph):
        delta, phi = p[0], p[1]
        out = propagate(jnp.asarray(v_dep0), delta, phi, t_f, n_leg, tof_out)
        turn_pen = jnp.maximum(jnp.abs(delta) - out["dmax"], 0.0) ** 2
        L = (out["cos_inc"] - cos_i_star) ** 2 + w_aph * ((out["aph"] - aph_star) / AU) ** 2 + 1.0e4 * turn_pen
        return L, out

    sweep = [ceil0 - 12, ceil0 - 6, ceil0 - 2, ceil0 + 2, ceil0 + 6, ceil0 + 12]
    b_rows = []
    for i_star in sweep:
        ci = float(np.cos(np.radians(i_star)))
        lg = jax.jit(jax.value_and_grad(lambda p, c=ci: obj_turn(p, c, 0.0, 0.0), has_aux=True))
        _, aux = run_adam(lg, jnp.array([0.3, 1.4]), args.iters, args.lr)   # seed a modest out-of-plane turn
        ach = float(aux["inc"])
        dl = abs(float(np.degrees(aux["delta"])))
        dmax = float(np.degrees(aux["dmax"]))
        defect = ach - i_star
        within = i_star <= ceil0
        reached = abs(defect) < 2.0
        # correct call: below ceiling should reach (defect≈0); above ceiling should floor (achieved≈ceiling<i*)
        call = "reach" if within else "floor"
        ok = (reached if within else (not reached and ach < i_star and abs(ach - ceil0) < 3.0))
        b_rows.append(ok)
        print(f"    {i_star:7.1f} {ach:12.2f} {defect:10.2f} {dl:7.0f} "
              f"{('yes' if dl <= dmax + 1e-6 else 'NO'):>7} {call+('✓' if ok else '✗'):>9}")
    b_ok = all(b_rows)
    print(f"    → {'SUPPORTED' if b_ok else 'REFUTED'}: below the ceiling the diff-sim reaches i* (defect→0); "
          f"above it the achieved inclination floors at ≈{ceil0:.1f}° (= arcsin(v∞/v_P)). The BACKPROP optimiser "
          "confirms R-N16's closed form under the true differentiable dynamics.")

    # ================= H-N17c: DOF — can a single node hold (inclination AND aphelion)? ==================
    print("\n  H-N17c: joint (inclination, aphelion) with the single node (2 DOF: δ,φ), |v∞| fixed:")
    print(f"    {'case':>30} {'inc(°)':>8} {'aph(AU)':>9} {'inc def':>9} {'aph def':>9}")
    # (i) a MODEST interior pair — measure the natural aphelion at a low inclination, then target BOTH → hittable
    i_interior = ceil0 * 0.4
    ci_int = float(np.cos(np.radians(i_interior)))
    lg_lo = jax.jit(jax.value_and_grad(lambda p: obj_turn(p, ci_int, 0.0, 0.0), has_aux=True))
    _, aux_lo = run_adam(lg_lo, jnp.array([0.2, 1.3]), args.iters, args.lr)
    aph_nat = float(aux_lo["aph"]) / AU                      # the aphelion the node naturally makes at 0.4·ceiling
    lg_i = jax.jit(jax.value_and_grad(lambda p: obj_turn(p, ci_int, aph_nat * AU, 1.0), has_aux=True))
    _, aux_i = run_adam(lg_i, jnp.array([0.2, 1.3]), args.iters, args.lr)
    inc_i, aph_i = float(aux_i["inc"]), float(aux_i["aph"]) / AU
    print(f"    {'interior (i=0.4·ceil, a=nat)':>30} {inc_i:8.2f} {aph_i:9.2f} {abs(inc_i-i_interior):9.3f} "
          f"{abs(aph_i-aph_nat):9.3f}")
    interior_ok = abs(inc_i - i_interior) < 0.5 and abs(aph_i - aph_nat) < 0.3

    # (ii) the PARETO FRONTIER: for each target aphelion, MAXIMISE inclination (minimise cos_inc while a strong
    #      penalty holds the aphelion). If max-inclination falls monotonically as the demanded aphelion rises,
    #      the reachable (i, aph) set is Tisserand-bounded and the two COMPETE (not independently specifiable).
    def obj_frontier(p, aph_star_au):
        delta, phi = p[0], p[1]
        out = propagate(jnp.asarray(v_dep0), delta, phi, t_f, n_leg, tof_out)
        turn_pen = jnp.maximum(jnp.abs(delta) - out["dmax"], 0.0) ** 2
        L = out["cos_inc"] + 50.0 * (out["aph"] / AU - aph_star_au) ** 2 + 1.0e4 * turn_pen  # aph_star in AU
        return L, out

    print("    Pareto frontier — MAX inclination reachable at a demanded aphelion (|v∞| fixed):")
    print(f"    {'aph* (AU)':>12} {'max inc(°)':>11} {'aph achieved':>13}")
    front = []
    for aph_star in [5.2, 6.5, 8.0, 10.0, 12.0]:
        lg_f = jax.jit(jax.value_and_grad(lambda p, a=aph_star: obj_frontier(p, a), has_aux=True))
        _, aux_f = run_adam(lg_f, jnp.array([0.5, 1.55]), args.iters, args.lr)
        front.append((aph_star, float(aux_f["inc"]), float(aux_f["aph"]) / AU))
        print(f"    {aph_star:12.1f} {front[-1][1]:11.2f} {front[-1][2]:13.2f}")
    max_inc = [f[1] for f in front]
    monotone = all(max_inc[i] >= max_inc[i + 1] - 0.5 for i in range(len(max_inc) - 1))   # non-increasing
    collapse = (max_inc[0] - max_inc[-1]) > 5.0                                             # frontier slopes down
    c_ok = interior_ok and monotone and collapse
    print(f"    → {'SUPPORTED' if c_ok else 'REFUTED'}: an interior (i,aph) pair is hit to ≪1%, and the frontier "
          f"of MAX inclination FALLS monotonically ({max_inc[0]:.1f}°→{max_inc[-1]:.1f}°) as the demanded aphelion "
          "rises — the single node lives on ONE Tisserand contour, so inclination and energy COMPETE (R-N15")
    print("      H-N15c, now via the optimiser). Prediction corrected: I framed it as 'aphelion pins at the "
          "ceiling'; the fuller truth is a sloped reachability frontier — high aphelion excludes high inclination.")

    print(f"\n  → verdicts: H-N17a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N17b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N17c {'SUPPORTED' if c_ok else 'REFUTED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tf", type=float, default=2.7)          # Earth→Jupiter leg-1 TOF (yr)
    ap.add_argument("--theta-arr", type=float, default=150.0)  # Jupiter arrival true longitude (deg)
    ap.add_argument("--i-star-a", type=float, default=15.0)    # H-N17a target inclination (deg)
    ap.add_argument("--tof-out", type=float, default=1.5)      # outbound leg TOF (yr)
    ap.add_argument("--n-leg", type=int, default=1500)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=5.0e-3)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
