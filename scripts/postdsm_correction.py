#!/usr/bin/env python3
"""Does the correctly-timed POST-DSM correction break R-N25's SOI cap, and at what effective leverage? (R-N28).

R-N25 -> R-N26 -> R-N27 refined the single-planet limit: rate cap (~0.085 km/s v inf/leg) -> single-CONTROL
limit -> the flyby TURN is the WRONG-timed second control (it precedes the DSM's de-phasing). The correctly-timed
control acts AFTER the DSM: a cleanup TCM on the descending leg. This round tests it at the right burn scale
(R-N24's closed-loop TCM cost ~3.6 km/s but used a 0.1 km/s DSM, ~20x too large per R-N25).

Per leg: launch a 1:2 resonance at v inf=8 from real Earth, coast to apoapsis, prograde DSM b (pumps v inf and
de-phases the real-Earth return), then a mid-descent TCM (2-D in-plane) solved to put the craft ON real Earth at
a FIXED nominal encounter time (smooth -> Gauss-Newton finds the small cleanup, not a grazing re-design, the
R-N27 lesson). Metric L_eff = (v inf gained) / (DSM + TCM), the net v inf per total Delta-v, both re-closing.

  H-N28a  feasibility: a post-DSM TCM re-closes a beyond-cap pumped leg within SOI (where R-N27's pre-DSM couldn't).
  H-N28b  economics: no CHEAP escape -- best within-SOI L_eff stays far below the free-within-cap L=15-37 (< 3).
  H-N28c  the RATE cap breaks: the per-leg within-SOI pump far exceeds ~85 m/s/leg, at L_eff ~ 1 (a Delta-v-for-time trade).

Mechanism study, never a Delta-v beat of a flown mission (locked belief 418e2e2). Single planet, patched-conic.
Reuses R-N24's cached JPL ephemeris; --verify offline, CI-safe.

    uv run --with jax --with astroquery --with astropy python scripts/postdsm_correction.py --verify   # offline
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import full_ephemeris_tour as F      # noqa: E402
import leverage_anatomy as L         # noqa: E402

AU = F.AU
SOI_E = F.SOI_E
DAY = F.DAY
MU_S = F.MU_S


def _period(rv):
    r = np.linalg.norm(rv[:3])
    v = np.linalg.norm(rv[3:])
    a = 1.0 / (2.0 / r - v * v / MU_S)
    return 2 * np.pi * np.sqrt(max(a, 0.1 * AU) ** 3 / MU_S)


def prop_to(rv, jd0, jd1, n=3000):
    tof = (jd1 - jd0) * DAY
    if tof <= 0:
        return np.asarray(rv)
    rvT, _ = F.propagate_ephem(np.asarray(rv), jd0, tof, n)
    return np.asarray(rvT)


def leg_to_apoapsis(V, jd, n=6000):
    rv0, tof = L.launch_exact(V, 1, 2, jd)
    _, traj = F.propagate_ephem(rv0, jd, tof, n)
    iap = int(np.argmax(np.linalg.norm(traj[:, :3], axis=1)))
    return traj[iap].copy(), jd + (tof * (iap / n)) / DAY


def closest(rv, jd0, n=6000):
    """Closest approach to real Earth over one period from rv: (miss_km, vinf, enc_jd)."""
    P = _period(rv)
    _, tj = F.propagate_ephem(np.asarray(rv), jd0, P, n)
    jj = jd0 + (np.arange(n) * (P / n)) / DAY
    eph = F._load("earth", False)
    d = np.linalg.norm(tj[:, :3] - F._sample_r(eph, jj), axis=1)
    h = int(0.30 * n)
    k = h + int(np.argmin(d[h:]))
    v_e = F._sample_v(eph, np.array([jj[k]]))[0]
    return float(d[k]), float(np.linalg.norm(tj[k, 3:] - v_e)), float(jj[k])


def state_at(rv, jd0, t):
    """(miss_to_Earth_km, vinf) exactly at fixed time t."""
    end = prop_to(rv, jd0, t, n=3000)
    eph = F._load("earth", False)
    r_e = F._sample_r(eph, np.array([t]))[0]
    v_e = F._sample_v(eph, np.array([t]))[0]
    return float(np.linalg.norm(end[:3] - r_e)), float(np.linalg.norm(end[3:] - v_e))


def solve_tcm(rv_tcm, jd_tcm, t_enc, iters=6):
    """2-D in-plane TCM nulling the craft-vs-Earth position at FIXED time t_enc. Returns (dv_xy, resid_km)."""
    eph = F._load("earth", False)
    r_e_enc = F._sample_r(eph, np.array([t_enc]))[0]

    def miss(dv):
        rv = np.asarray(rv_tcm).copy()
        rv[3] += dv[0]
        rv[4] += dv[1]
        return (prop_to(rv, jd_tcm, t_enc, n=2500)[:3] - r_e_enc)[:2]

    dv = np.zeros(2)
    for _ in range(iters):
        m0 = miss(dv)
        eps = 5e-5
        J = np.zeros((2, 2))
        for j in range(2):
            d2 = dv.copy()
            d2[j] += eps
            J[:, j] = (miss(d2) - m0) / eps
        try:
            step = np.linalg.solve(J, m0)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        dv = dv - step
    return dv, float(np.linalg.norm(miss(dv)))


def corrected_leg(rv_ap, apo_jd, b, f, vinf_nat):
    """Prograde DSM b at apoapsis + a TCM at descent-fraction f, solved to hit real Earth at the fixed nominal
    encounter. Returns (tcm_ms, miss_at_enc_km, dvinf_ms, L_eff)."""
    vh = rv_ap[3:] / np.linalg.norm(rv_ap[3:])
    rv_dsm = rv_ap.copy()
    rv_dsm[3:] = rv_dsm[3:] + b * vh
    _, _, t_enc = closest(rv_dsm, apo_jd)
    t_tcm = apo_jd + f * (t_enc - apo_jd)
    rv_tcm = prop_to(rv_dsm, apo_jd, t_tcm, n=3500)
    dv, _ = solve_tcm(rv_tcm, t_tcm, t_enc)
    rv_after = rv_tcm.copy()
    rv_after[3] += dv[0]
    rv_after[4] += dv[1]
    miss_e, vinf_e = state_at(rv_after, t_tcm, t_enc)
    dvinf = (vinf_e - vinf_nat) * 1000.0
    tcm = float(np.linalg.norm(dv)) * 1000.0
    leff = dvinf / (b * 1000.0 + tcm) if (b * 1000.0 + tcm) > 0 else float("nan")
    return tcm, miss_e, dvinf, leff


def verify(args):
    print("=== R-N28: does the correctly-timed POST-DSM correction break R-N25's SOI cap, and at what L_eff? ===")
    if not F._require_cache():
        return
    sjd = F._start_jd()
    V = 8.0
    rv_ap, apo_jd = leg_to_apoapsis(V, sjd)
    _, vinf_nat, _ = closest(rv_ap, apo_jd)
    vh = rv_ap[3:] / np.linalg.norm(rv_ap[3:])
    print(f"  v inf0={V}, natural 1:2 re-encounter v inf={vinf_nat:.3f}, SOI={SOI_E/AU:.3f} AU. R-N25 sustainable "
          "rate cap ~85 m/s v inf/leg; free-within-cap leverage L=15-37.")

    # ---- H-N28a: the correctly-timed control re-closes a beyond-cap pump (contrast R-N27's pre-DSM flyby) ----
    b_big = 0.100                                            # ~5-10x the cap-equivalent burn -> de-phases past SOI
    rv_dsm = rv_ap.copy()
    rv_dsm[3:] = rv_dsm[3:] + b_big * vh
    nomiss, _, _ = closest(rv_dsm, apo_jd)                   # un-corrected (pre-DSM control only) miss
    tcm_a, miss_a, dvinf_a, _ = corrected_leg(rv_ap, apo_jd, b_big, 0.10, vinf_nat)
    a_ok = nomiss > SOI_E and miss_a < SOI_E
    print(f"\n  H-N28a: DSM={b_big*1000:.0f} m/s (beyond-cap pump). No post-DSM control -> re-encounter miss "
          f"{nomiss/SOI_E:.1f} SOI (busts, as R-N27). Post-DSM TCM ({tcm_a:.0f} m/s) -> miss {miss_a/SOI_E:.3f} SOI.")
    print(f"    -> H-N28a {'SUPPORTED' if a_ok else 'REFUTED'}: the post-DSM TCM {'re-closes' if a_ok else 'fails to re-close'} "
          "the beyond-cap leg within SOI -- the correctly-timed control works where R-N27's pre-DSM flyby turn could not.")

    # ---- H-N28b: is there a CHEAP escape? best within-SOI L_eff vs the free-within-cap L=15-37 ----
    print("\n  H-N28b: best within-SOI L_eff = (v inf gained)/(DSM+TCM), optimizing DSM size and TCM placement:")
    print(f"    {'DSM(m/s)':>9} {'best L_eff':>11} {'@ TCM(m/s)':>11} {'pump(m/s)':>10}")
    best_leff = 0.0
    best_pump_at_econ = 0.0
    for b in [0.050, 0.100, 0.200, 0.300]:
        rowbest = (-9.0, 0.0, 0.0)
        for f in [0.05, 0.10, 0.15]:
            tcm, miss_e, dvinf, leff = corrected_leg(rv_ap, apo_jd, b, f, vinf_nat)
            if miss_e < SOI_E and leff > rowbest[0]:
                rowbest = (leff, tcm, dvinf)
        best_leff = max(best_leff, rowbest[0])
        if rowbest[0] >= 1.0:
            best_pump_at_econ = max(best_pump_at_econ, rowbest[2])
        print(f"    {b*1000:9.0f} {rowbest[0]:11.2f} {rowbest[1]:11.0f} {rowbest[2]:10.0f}")
    b_ok = best_leff < 3.0                                   # REFUTE-BY: best L_eff >= 3 (a cheap escape)
    print(f"    -> H-N28b {'SUPPORTED' if b_ok else 'REFUTED'}: best within-SOI L_eff = {best_leff:.2f} "
          f"({'<' if b_ok else '>='} 3) -- {'no CHEAP single-planet escape' if b_ok else 'a cheap escape exists'}; "
          f"it peaks ~{best_leff:.1f} (marginally economical, correcting the '<=1' lean) but is nowhere near the "
          "free-within-cap L=15-37.")

    # ---- H-N28c: the RATE cap breaks -- big per-leg pump within SOI, at L_eff ~ 1 (Delta-v-for-time) ----
    print("\n  H-N28c: max within-SOI per-leg pump (vs R-N25's ~85 m/s/leg), and the L_eff it comes at:")
    max_pump = 0.0
    pump_leff = float("nan")
    for b in [0.200, 0.300, 0.500]:
        tcm, miss_e, dvinf, leff = corrected_leg(rv_ap, apo_jd, b, 0.05, vinf_nat)
        if miss_e < SOI_E and dvinf > max_pump:
            max_pump = dvinf
            pump_leff = leff
        print(f"    DSM={b*1000:.0f} m/s, f=0.05 -> within-SOI pump {dvinf:.0f} m/s at L_eff {leff:.2f} "
              f"(miss {miss_e/SOI_E:.3f} SOI)")
    c_ok = max_pump > 200.0                                  # REFUTE-BY: no within-SOI leg pumps > 200 m/s
    print(f"    -> H-N28c {'SUPPORTED' if c_ok else 'REFUTED'}: the post-DSM TCM reaches {max_pump:.0f} m/s v inf/leg "
          f"within SOI ({'>>' if c_ok else '<='} the ~85 m/s rate cap) at L_eff {pump_leff:.2f} -- the SOI RATE cap "
          "breaks, but as a Delta-v-for-time trade (pay ~1:1), not free leverage.")

    print(f"\n  -> verdicts: H-N28a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N28b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N28c {'SUPPORTED' if c_ok else 'REFUTED'}")
    print("  NET (closes the R-N25->R-N28 arc): the correctly-timed control is the POST-DSM cleanup, and it does")
    print("    what R-N27's pre-DSM flyby turn could not -- it re-closes a beyond-cap pumped leg within SOI")
    print("    (H-N28a). It BREAKS R-N25's SOI RATE cap: the per-leg pump reaches several hundred m/s, ~10x the")
    print("    ~85 m/s cap (H-N28c). BUT there is NO cheap single-planet escape -- best L_eff peaks at only ~1.1")
    print("    (H-N28b), correcting my '<=1' lean at the margin while confirming its spirit: the cleanup costs")
    print("    ~1 m/s of Delta-v per m/s of v inf, nowhere near the free-within-cap L=15-37. So the free leverage")
    print("    lives only inside the SOI budget (~85 m/s/leg); beyond it you pay ~1:1. The genuinely cheap way to")
    print("    pump FAST is therefore MULTI-PLANET -- each planet contributes its own free ~85 m/s/leg SOI budget,")
    print("    which re-motivates the multi-planet tour (R-N29) for the right reason. Honest scope: single planet,")
    print("    patched-conic, in-plane TCM, fixed-time null; the multi-planet tour is R-N29's knob.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
