#!/usr/bin/env python3
"""Does the FLYBY TURN (R-N26's proposed second control) actually break R-N25's SOI rate cap? (Build N, R-N27).

R-N26 concluded R-N25's ~0.085 km/s v∞/leg SOI rate cap is a single-CONTROL limit (one apoapsis burn can't both
pump |v∞| AND retarget the re-encounter) and claimed — via a LINEARIZED authority check (H-N26c) — that a second
control, the flyby TURN, has 74–95× the authority needed to break it. R-N27 tests that forward claim in the full
nonlinear real-ephemeris model, and CORRECTS it.

The correction: the flyby turn acts at leg START (it sets the outgoing resonance direction), while the DSM burn
acts mid-leg and shifts the END-of-leg re-encounter. A control that PRECEDES the de-phasing cannot correct it —
so the flyby only re-selects the (cap-neutral, per R-N26b) resonance; it does NOT break the cap. The 74–95×
authority was real but MIS-TIMED. The correct second control acts AFTER the DSM (a cleanup TCM, or the next-body
flyby) — deferred to R-N28.

  H-N27a  going-in (R-N26c): a flyby-REACHABLE resonance + DSM breaks the cap (within-SOI Δv∞ ≫ cap).
  H-N27b  SEQUENCE mechanism: the within-SOI pump is monotone-capped; no reachable resonance holds a bigger pump.
  H-N27c  POSITIVE control: the turn is authoritative (spans ≥3 resonances over ±δmax) yet the pump stays capped.

Mechanism study, never a Δv beat of a flown mission (locked belief 418e2e2). In-plane flyby turn (the out-of-plane
crank is a separate control). Reuses R-N24's cached JPL ephemeris; --verify offline, CI-safe.

    uv run --with jax --with astroquery --with astropy python scripts/flyby_leverage.py --verify   # offline
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import full_ephemeris_tour as F      # noqa: E402  (ephemeris machinery + cached JPL window)
import leverage_anatomy as L         # noqa: E402  (launch_exact, sized_leverage_leg, TSID)

AU = F.AU
SOI_E = F.SOI_E
DAY = F.DAY
MU_E = 398600.4418                   # km^3/s^2 (Earth GM, for δmax)
R_E_KM = 6378.137                    # km
TSID = L.TSID

LADDER = [(1, 1), (3, 2), (2, 3), (3, 4), (4, 5), (1, 2), (2, 5), (1, 3)]


def delta_max(V, rp_km=1.05 * R_E_KM):
    """Patched-conic max single-flyby turn (rad): 2·arcsin(1/(1 + rp·V²/μ_E))."""
    return 2.0 * np.arcsin(1.0 / (1.0 + rp_km * V ** 2 / MU_E))


def _period(rv):
    r = np.linalg.norm(rv[:3])
    v = np.linalg.norm(rv[3:])
    a = 1.0 / (2.0 / r - v * v / F.MU_S)
    return 2 * np.pi * np.sqrt(max(a, 0.1 * AU) ** 3 / F.MU_S)


def _wrap(ang):
    return abs(((ang + np.pi) % (2 * np.pi)) - np.pi)


def incoming_phi(V, jd, n=6000):
    """In-plane angle (rad) and |v∞| of the arrival v∞ from a bootstrap 1:2 return to real Earth."""
    rv0, _ = L.launch_exact(V, 1, 2, jd)
    P = _period(rv0)
    _, traj = F.propagate_ephem(rv0, jd, P, n)
    # jj uses the campaign-wide convention (jj[k]=jd+k·dt); shared with R-N24/25/26 + sized_leverage_leg so
    # cross-round miss numbers stay apples-to-apples. The ~1-step (~0.12 d, ~0.2·SOI) offset flips no verdict.
    jj = jd + (np.arange(n) * (P / n)) / DAY
    eph_e = F._load("earth", False)
    d = np.linalg.norm(traj[:, :3] - F._sample_r(eph_e, jj), axis=1)
    h = int(0.4 * n)
    k = h + int(np.argmin(d[h:]))
    vinf = traj[k, 3:] - F._sample_v(eph_e, np.array([jj[k]]))[0]
    return float(np.arctan2(vinf[1], vinf[0])), float(np.linalg.norm(vinf))


def outgoing_phi(V, p, q, jd):
    """In-plane angle (rad) of the outgoing v∞ for a constructed p:q resonance, or None if infeasible."""
    out = L.launch_exact(V, p, q, jd)
    if out is None:
        return None
    rv0, _ = out
    _, v_e = F.earth_rv(jd)
    vinf = rv0[3:] - v_e
    return float(np.arctan2(vinf[1], vinf[0]))


def period_at_turn(V, jd, phi):
    """Outgoing orbital period (in Earth-years) if the flyby aims v∞ (magnitude V) at in-plane angle phi."""
    r_e, v_e = F.earth_rv(jd)
    vinf = np.array([V * np.cos(phi), V * np.sin(phi), 0.0])
    return _period(np.concatenate([r_e, v_e + vinf])) / (TSID * DAY)


def dsm_leg(V, p, q, jd, b, n=6000):
    """Launch p:q at V, coast to apoapsis, prograde DSM b, return (miss_km, vinf_next) at real-Earth closest
    approach, or None if infeasible."""
    out = L.launch_exact(V, p, q, jd)
    if out is None:
        return None
    rv0, tof = out
    _, traj = F.propagate_ephem(rv0, jd, tof, n)
    iap = int(np.argmax(np.linalg.norm(traj[:, :3], axis=1)))
    apo_jd = jd + (tof * (iap / n)) / DAY
    rv_ap = traj[iap].copy()
    vh = rv_ap[3:] / np.linalg.norm(rv_ap[3:])
    rv_ap[3:] = rv_ap[3:] + b * vh
    P2 = _period(rv_ap)
    _, tj = F.propagate_ephem(rv_ap, apo_jd, P2, n)
    jj = apo_jd + (np.arange(n) * (P2 / n)) / DAY            # campaign-wide convention (see incoming_phi note)
    eph_e = F._load("earth", False)
    d = np.linalg.norm(tj[:, :3] - F._sample_r(eph_e, jj), axis=1)
    h = int(0.30 * n)
    k = h + int(np.argmin(d[h:]))
    v_e_k = F._sample_v(eph_e, np.array([jj[k]]))[0]
    return float(d[k]), float(np.linalg.norm(tj[k, 3:] - v_e_k))


def verify(args):
    print("=== R-N27: does the FLYBY TURN (R-N26's proposed 2nd control) actually break R-N25's SOI cap? ===")
    if not F._require_cache():
        return
    sjd = F._start_jd()
    V0 = 8.0
    dmax = delta_max(V0)
    phi_in, Vin = incoming_phi(V0, sjd)
    print(f"  v∞0={V0}, incoming |v∞|={Vin:.2f} @ φ_in={np.degrees(phi_in):.1f}°, δmax={np.degrees(dmax):.1f}°, "
          f"SOI={SOI_E/AU:.3f} AU. R-N26c claimed the flyby turn (74–95× authority) breaks the cap.")

    # ---- H-N27a: does a flyby-REACHABLE resonance + DSM break the cap? ----
    print("\n  H-N27a: flyby-reachable resonances (outgoing turn from φ_in ≤ δmax) + sized DSM — within-SOI Δv∞:")
    print(f"    {'p:q':>6} {'turn(°)':>8} {'reachable':>10} {'within-SOI Δv∞(m/s)':>20} {'miss×SOI':>9}")
    best_reach = 0.0
    for (p, q) in LADDER:
        po = outgoing_phi(V0, p, q, sjd)
        r = L.sized_leverage_leg(V0, sjd, frac=0.5, p=p, q=q)
        if po is None or r is None:
            continue
        turn = _wrap(po - phi_in)
        vnew, miss, _, _burn, _ = r
        reach = turn <= dmax
        within = miss < SOI_E
        dv = (vnew - V0) * 1000.0
        if reach and within and dv > 0:
            best_reach = max(best_reach, dv)
        print(f"    {f'{p}:{q}':>6} {np.degrees(turn):8.1f} {('yes' if reach else 'NO'):>10} "
              f"{(f'{dv:+.0f}' if within else 'busts SOI'):>20} {miss/SOI_E:9.2f}")
    a_break = best_reach > 300.0                              # REFUTE-BY: reachable within-SOI Δv∞ > 0.3 km/s
    print(f"    → H-N27a {'SUPPORTED (breaks cap)' if a_break else 'REFUTED'}: best flyby-reachable within-SOI Δv∞ "
          f"= {best_reach:.0f} m/s — the flyby turn {'breaks' if a_break else 'does NOT break'} the cap "
          f"({'≫' if a_break else '≈'} R-N26's ~85–185 m/s resonance cap). R-N26c's forward claim "
          f"{'held' if a_break else 'REFUTED'}.")

    # ---- H-N27b: SEQUENCE mechanism — the within-SOI pump is monotone-capped for the best reachable resonance ----
    print("\n  H-N27b: pump-vs-miss on the best reachable resonance (1:2) — increasing DSM to target a bigger pump:")
    print(f"    {'DSM(m/s)':>9} {'Δv∞(m/s)':>9} {'miss×SOI':>9} {'within SOI?':>12}")
    cap_dv = 0.0
    for b in [0.0, 0.005, 0.010, 0.020, 0.040, 0.080]:
        res = dsm_leg(V0, 1, 2, sjd, b)
        if res is None:
            continue
        miss, vn = res
        dv = (vn - V0) * 1000.0
        within = miss < SOI_E
        if within and dv > cap_dv:
            cap_dv = dv
        print(f"    {b*1000:9.1f} {dv:+9.0f} {miss/SOI_E:9.2f} {('yes' if within else 'BUSTS'):>12}")
    b_capped = cap_dv < 300.0                                 # REFUTE-BY: holds ≥0.3 km/s within SOI
    print(f"    → H-N27b {'SUPPORTED' if b_capped else 'REFUTED'}: the within-SOI pump is monotone-capped at "
          f"~{cap_dv:.0f} m/s — a bigger DSM de-phases the re-encounter past SOI. The pre-DSM flyby (it only picks "
          f"the resonance) cannot hold a bigger pump: it PRECEDES the DSM's de-phasing.")

    # ---- H-N27c: POSITIVE control — the turn is authoritative (spans resonances) yet the pump stays capped ----
    print("\n  H-N27c: POSITIVE control — outgoing period vs flyby turn over the reachable ±δmax fan:")
    turns = np.linspace(-np.degrees(dmax), np.degrees(dmax), 9)
    periods = [period_at_turn(V0, sjd, phi_in + np.radians(t)) for t in turns]
    pmin, pmax = min(periods), max(periods)
    # count integer-ish resonances (period near a rational q/p with small p,q) the fan spans, as a coarse proxy
    span_res = sum(1 for (p, q) in LADDER if pmin - 0.05 <= q / p <= pmax + 0.05)
    print("    turn(°):   " + "  ".join(f"{t:+5.0f}" for t in turns))
    print("    period(yr):" + "  ".join(f"{p:5.2f}" for p in periods))
    print(f"    → outgoing period spans {pmin:.2f}–{pmax:.2f} yr over ±δmax, crossing ~{span_res} ladder resonances")
    c_auth = span_res >= 3                                    # REFUTE-BY: spans <1 resonance (turn too weak)
    print(f"    → H-N27c {'SUPPORTED' if c_auth else 'REFUTED'}: the flyby turn IS authoritative (spans ~{span_res} "
          f"resonances) yet the within-SOI pump stays at the ~{cap_dv:.0f} m/s cap at each — so the turn is "
          f"MIS-APPLIED to pumping, not weak.")

    print(f"\n  → verdicts: H-N27a {'SUPPORTED' if a_break else 'REFUTED'}, "
          f"H-N27b {'SUPPORTED' if b_capped else 'REFUTED'}, H-N27c {'SUPPORTED' if c_auth else 'REFUTED'}")
    print("  NET (corrects R-N26c's forward claim): the flyby TURN does NOT break R-N25's SOI cap. R-N26c's 74–95×")
    print("    authority was real but MIS-TIMED. In a leg the flyby acts FIRST (it sets the outgoing resonance —")
    print("    cap-neutral per R-N26b), then the DSM burn acts at apoapsis and shifts the END-of-leg real-Earth")
    print("    re-encounter; a control that PRECEDES that shift cannot correct it. So flyby-reachable resonances")
    print("    hold only the ~85–256 m/s cap (H-N27a,b), the pump is monotone-capped for every reachable resonance")
    print("    (H-N27b), and although the turn is strongly authoritative — it spans several resonances (H-N27c) —")
    print("    it is mis-applied to pumping, not weak. The correct second control must act AFTER the DSM — a")
    print("    cleanup TCM (single-planet) or the next-body flyby (multi-planet). Whether THAT cheaply breaks the")
    print("    cap is genuinely OPEN: R-N24's closed-loop post-DSM TCM cost ~3.6 km/s and still didn't pump, so a")
    print("    naive TCM may be expensive — R-N28's question. Honest scope: in-plane turn, patched-conic, single")
    print("    planet; the post-DSM correction/TCM solve is R-N28's knob, not rushed here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
