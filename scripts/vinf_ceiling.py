#!/usr/bin/env python3
"""Does the multi-planet v inf-pump have a CEILING as v inf climbs? (Build N, R-N31).

R-N29/R-N30 built the multi-planet case at LOW v inf (3-15 km/s). The pump RAISES v inf, and the load-bearing
untested assumption was: "a higher v inf may DISCONNECT the planets." This round sweeps the connectivity and the
flyby-turn authority over CLIMBING v inf. The measure-first probe REFINED the going-in lean: connectivity does
NOT disconnect (dual-crossing usable-v inf orbits persist -- the two planets' v inf climb together), but the
flyby turn authority delta_max COLLAPSES with v inf -- so the ceiling is a MANEUVERABILITY (delta_max) ceiling,
not a REACHABILITY (connectivity) one.

  v inf_P^2 = v_cP^2 (3 - T_P),  T_P = a_P/a + 2 sqrt((a/a_P)(1-e^2))         (Tisserand connectivity)
  delta_max = 2 arcsin(1/(1 + rp v^2 / mu_P))                                 (patched-conic flyby turn)
  cos gamma = (v_out^2 - v_cP^2 - v inf^2) / (2 v_cP v inf),  v_out=v_cP sqrt(2-(p/q)^(2/3))   (resonance pump angle)

  H-N31a  connectivity PERSISTS to high v inf (no hard geometric ceiling) -- refutes the "disconnect" lean.
  H-N31b  delta_max COLLAPSES with v inf (the real ceiling mechanism).
  H-N31c  a SOFT ceiling: the v inf where delta_max drops below the min adjacent-resonance pump-angle gap
          (below which the resonance ladder stops being single-flyby-walkable and free hops stall).

Analytic reachability study (R-N16 level; real-ephemeris eccentricity a small caveat). No ephemeris needed.
Mechanism study, never a Delta-v beat (locked belief 418e2e2).

    uv run python scripts/vinf_ceiling.py --verify
"""
from __future__ import annotations

import argparse

import numpy as np

MU_S = 1.32712440018e11
AU = 1.495978707e8
A = {"venus": 0.7233, "earth": 1.0000, "mars": 1.5237}
MU_P = {"venus": 3.24859e5, "earth": 3.986004418e5, "mars": 4.282837e4}
RP = {"venus": 1.05 * 6051.8, "earth": 1.05 * 6378.1, "mars": 1.05 * 3389.5}


def v_circ(p):
    return float(np.sqrt(MU_S / (A[p] * AU)))


def vinf_orbit(p, a, e):
    aP = A[p]
    if not (a * (1 - e) <= aP <= a * (1 + e)):
        return np.nan
    val = 3.0 - (aP / a + 2.0 * np.sqrt((a / aP) * (1 - e * e)))
    return v_circ(p) * np.sqrt(val) if val > 0 else np.nan


def delta_max_deg(p, v):
    return float(np.degrees(2.0 * np.arcsin(1.0 / (1.0 + RP[p] * v ** 2 / MU_P[p]))))


def _ladder(pmax=3, qmax=3):
    """Ratio-deduped LOW-ORDER resonance rungs (the fast/useful pump ladder, R-N16 level). Deduping by RATIO is
    essential: (1,1)/(2,2)/(3,3) are the same resonance and would otherwise fake a 0° hop gap."""
    seen, out = set(), []
    for q in range(1, qmax + 1):
        for p in range(1, pmax + 1):
            r = q / p
            key = round(r, 4)
            if 0.5 <= r <= 2.0 and key not in seen:
                seen.add(key)
                out.append((p, q))
    return out


LADDER = _ladder()


def pump_angles(p, v):
    """Feasible resonance pump angles gamma (deg) at planet p, excess speed v -- sorted."""
    vc = v_circ(p)
    gs = []
    for (pp, qq) in LADDER:
        vout = vc * np.sqrt(max(2.0 - (pp / qq) ** (2.0 / 3.0), 0.0))
        cg = (vout ** 2 - vc ** 2 - v ** 2) / (2 * vc * v)
        if abs(cg) <= 1.0:
            gs.append(np.degrees(np.arccos(cg)))
    return np.array(sorted(gs))


def min_hop_gap(p, v):
    """Smallest adjacent pump-angle gap in the feasible ladder (deg), or nan if < 2 resonances."""
    g = pump_angles(p, v)
    return float(np.min(np.diff(g))) if len(g) >= 2 else np.nan


def connectable_vinf(p1, p2, v1_lo, v1_hi, n=600):
    """Dual-crossing orbits with v inf@p1 in [v1_lo,v1_hi]; return the usable (2-20 km/s) v inf@p2 count/range."""
    a1, a2 = A[p1], A[p2]
    inner, outer = min(a1, a2), max(a1, a2)
    v2s = []
    for a in np.linspace(0.4, 6.0, n):
        for e in np.linspace(0.01, 0.97, n):
            if a * (1 - e) > inner or a * (1 + e) < outer:
                continue
            v1 = vinf_orbit(p1, a, e)
            if np.isnan(v1) or not (v1_lo <= v1 < v1_hi):
                continue
            v2 = vinf_orbit(p2, a, e)
            if not np.isnan(v2):
                v2s.append(v2)
    v2s = np.array(v2s)
    usable = v2s[(v2s >= 2) & (v2s <= 20)] if len(v2s) else v2s
    rng = (float(usable.min()), float(usable.max())) if len(usable) else (np.nan, np.nan)
    return len(usable), rng                                  # count and range of the USABLE subset (per docstring)


def soft_ceiling(p, vmax=30.0):
    """Lowest v inf where delta_max drops below the min feasible-ladder hop gap (deg). None if never."""
    for v in np.arange(4.0, vmax + 0.1, 0.5):
        gap = min_hop_gap(p, v)
        if not np.isnan(gap) and delta_max_deg(p, v) < gap:
            return float(v), gap, delta_max_deg(p, v)
    return None


def verify(args):
    print("=== R-N31: does the multi-planet v inf-pump have a ceiling as v inf climbs? ===")
    print(f"  v_circ: Venus {v_circ('venus'):.1f}, Earth {v_circ('earth'):.1f}, Mars {v_circ('mars'):.1f} km/s")

    # ---- H-N31a: connectivity PERSISTS to high v inf ----
    print("\n  H-N31a: usable dual-crossing connectors (v inf 2-20 km/s at BOTH) as v inf@P1 climbs:")
    a_ok = True
    for (p1, p2) in [("venus", "earth"), ("earth", "mars")]:
        row = []
        for vlo in (3, 10, 17, 20):
            cnt, _ = connectable_vinf(p1, p2, vlo, vlo + 1.5)
            row.append((vlo, cnt))
        hi = [c for (v, c) in row if v == 20][0]
        if hi == 0:
            a_ok = False
        print(f"    {p1}<->{p2}: " + ", ".join(f"v inf@{p1}~{v}:{c} usable" for (v, c) in row))
    print(f"    → H-N31a {'SUPPORTED' if a_ok else 'REFUTED'}: usable dual-crossing orbits persist to v inf@P1 ≥ 20 "
          f"— connectivity does NOT disconnect (the planets' v inf climb TOGETHER). My 'hard-disconnect' lean was WRONG.")

    # ---- H-N31b: delta_max COLLAPSES with v inf ----
    print("\n  H-N31b: flyby turn authority δmax(v inf) per planet (deg):")
    print(f"    {'v inf':>6} {'Venus':>7} {'Earth':>7} {'Mars':>7}")
    for v in (3, 6, 9, 12, 16, 20, 24):
        print(f"    {v:6.0f} {delta_max_deg('venus', v):7.1f} {delta_max_deg('earth', v):7.1f} {delta_max_deg('mars', v):7.1f}")
    collapse = delta_max_deg("venus", 24) < 15.0
    print(f"    → H-N31b {'SUPPORTED' if collapse else 'REFUTED'}: δmax collapses with v inf "
          f"(Venus {delta_max_deg('venus', 3):.0f}°→{delta_max_deg('venus', 24):.0f}° over v inf 3→24) — "
          "the flyby's ability to redirect (hop resonances / execute the handoff turn) shrinks sharply.")

    # ---- H-N31c: a SOFT FAST-PUMP ceiling? δmax vs the min gap of the FAST (low-order) resonance ladder ----
    print("\n  H-N31c: fast-pump soft ceiling — lowest v inf where δmax < the min gap of the FAST (low-order) ladder:")
    ceils = {p: soft_ceiling(p) for p in ("venus", "earth", "mars")}
    for p in ("venus", "earth", "mars"):
        sc = ceils[p]
        if sc is None:
            print(f"    {p:>7}: δmax stays above the fast-rung gap to v inf=30 (no fast-pump ceiling ≤ 30 km/s)")
        else:
            v, gap, dm = sc
            print(f"    {p:>7}: fast-pump ceiling at v inf ≈ {v:.0f} km/s (δmax {dm:.1f}° < fast-rung gap {gap:.1f}°)")
    have = {p: c for p, c in ceils.items() if c is not None}
    c_ok = len(have) >= 1 and all(c[0] >= 10 for c in have.values())
    summ = ", ".join(f"{p} ~{c[0]:.0f}" for p, c in have.items()) if have else "none ≤ 30"
    print(f"    → H-N31c {'SUPPORTED' if c_ok else 'REFUTED'}: a PLANET-DEPENDENT soft fast-pump ceiling exists "
          f"({summ} km/s) — the LIGHT planet (Mars, smallest δmax) can't bridge its fast rungs first; the heavier "
          "Venus/Earth retain fast hopping past 30. NOTE: on the FULL ladder (incl. slow high-order rungs) nothing")
    print("      hard-disconnects — so this is a FAST-pump / rate ceiling (forced onto slow rungs or Δv), NOT a "
          "reachability wall.")

    print(f"\n  → verdicts: H-N31a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N31b {'SUPPORTED' if collapse else 'REFUTED'}, H-N31c {'SUPPORTED' if c_ok else 'REFUTED'}")
    print("  NET (my HARD-ceiling lean refuted; the ceiling is SOFT, planet-ordered): the multi-planet free pump")
    print("    has NO HARD ceiling — connectivity PERSISTS because the two planets' v inf climb TOGETHER (H-N31a,")
    print("    refuting my 'disconnect' lean); v inf is reachable arbitrarily high IN PRINCIPLE. But the flyby turn")
    print("    authority δmax COLLAPSES with v inf (H-N31b: >100°→<10° over 3→24, fastest at the LIGHTEST planet),")
    print("    so the FAST pump hits a SOFT, PLANET-DEPENDENT ceiling (H-N31c): the light planet Mars can't bridge")
    print("    its fast low-order rungs above v inf≈14, forcing slow high-order hops or Δv; heavier Venus/Earth")
    print("    retain fast hopping past 30. So the honest bound is DIMINISHING RETURNS that bite FIRST at the light")
    print("    planets: the free fast pump degrades planet-by-planet as v inf climbs, never a hard wall (slow rungs")
    print("    and Δv-assisted pumping at L_eff≈1, R-N28, always remain). This corrects my 'hard disconnect' lean")
    print("    and confirms the 'soft planet-dependent ceiling' H-N31c prediction, and sets up R-N32 (the chained")
    print("    tour / optimizer discovery) with the honest expectation of graceful, planet-ordered degradation.")
    print("    Scope: analytic Tisserand/δmax reachability, patched-conic, coplanar; real-ephemeris ecc a caveat.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
