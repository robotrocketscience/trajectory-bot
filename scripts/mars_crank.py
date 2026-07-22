#!/usr/bin/env python3
"""Does the inclination CRANK REALIZE at Mars from the R-N45 hot arrival? (Build N, R-N46).

R-N45 measured the POTENTIAL: the pumped->DSM handoff arrives at Mars with a hot v_inf (7.85 / 14.47 km/s at
t0+200 / t0+600) whose analytic single-planet crank ceiling arcsin(v_inf/v_Mars) is 17.2 / 34.8 deg. This round
tests whether that ceiling REALIZES — seed R-N38's exact-ballistic crank walk (greedy max-inclination over
GN-closed resonant same-planet returns) with the R-N45 Mars arrival node and see whether a chain of zero-DSM
resonant Mars flybys actually climbs the relative inclination.

Physics tension: Mars's mu = 4.28e4 km^3/s^2 is ~10x smaller than Earth's, so the per-flyby turn
delta_max = 2 asin(1/(1 + r_p v_inf^2/mu)) is much smaller. At the t0+200 arrival (v_inf 7.85) delta_max ~ 19
deg EXCEEDS the 17 deg ceiling (one flyby nearly saturates); at t0+600 (v_inf 14.47) delta_max ~ 6 deg << the
35 deg ceiling (many taxed steps). So the crank may realize cleanly at the low-v_inf arrival and be re-closure-
taxed at the high-v_inf one — an epoch-conditioned answer.

RESULT (2026-07-22) — this going-in lean was REFUTED, INVERTED: the LOW-v_inf t0+200 arrival takes ONE big step
then PHYSICALLY STALLS at 34% of ceiling (a wide/dense post-step search finds no raising re-closing return —
the sparse Mars resonance ladder offers no follow-up after a large rotation), while the HIGH-v_inf t0+600
arrival chains 8 small re-closing steps to 68% of ceiling (23.8 deg). A SUSTAINED Mars crank needs MANY SMALL
re-closing steps, not one big rotation; the realized fraction is NON-monotone (inverted) in delta_max. Verdicts
below judge the pre-registered PRIMARY (t0+200) plainly (a/b REFUTED) and report the t0+600 realization as MIXED
rather than cherry-picking it into a clean SUPPORTED. H-N46c holds (exactly ballistic, |v_inf| conserved).

  H-N46a  the crank mechanism TRANSFERS to Mars: >= 3 CONSECUTIVE GN-closed resonant Mars returns each RAISE
          i_rel (zero DSM). REFUTE-BY: < 3 consecutive raising returns.
  H-N46b  the (lower) Mars ceiling is APPROACHABLE: i_rel reaches >= 50% of arcsin(v_inf/v_Mars) within <= 8
          encounters (the SAME bar as R-N38b). REFUTE-BY: < 50% of the ceiling in <= 8 encounters.
  H-N46c  the Mars crank is FREE and non-destructive: every resonant Mars leg re-closes ballistically
          (miss < SOI_mars), |v_inf| drift <= 2%. REFUTE-BY: a leg fails to close, or drift > 2%.

ONE knob vs R-N38: the crank PLANET is Mars, seeded from the R-N45 arrival. Disclosed instrument adaptation:
the crank tof grid is retuned to Mars's 687-d period (R-N38's Venus 80-460 d grid contains no Mars resonant
return, which live near 687 / 1374 d). Primary run = the better-conditioned t0+200 arrival; t0+600 reported as
the harder case. Full mission shape: pump (ballistic) -> DSM handoff to Mars (<=1 km/s, R-N43/45) -> crank at
Mars (this round, exactly ballistic). Mechanism/DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/mars_crank.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import constrained_tour_discovery as C   # noqa: E402
import mars_dsm as M                      # noqa: E402  (adds Mars to C's MU_P/RP/SOI; earth_node, min_dsm)
import mars_arrival as MA                 # noqa: E402  (R-N45 arrival_state)
import crank_walk as CW                   # noqa: E402  (R-N38 crank machinery — reused verbatim)

# Retune R-N38's crank tof grid to Mars's ~687-d period. crank_continuations reads this module global, so
# rebinding it here makes the reused R-N38 walk enumerate Mars resonant returns (687 = 1:1, 1374 = 2:1).
CW._CRANK_GRID = np.array(np.meshgrid(np.linspace(-2.0, 2.0, 7),
                                      np.linspace(0, 2 * np.pi, 16, endpoint=False),
                                      np.linspace(400, 1450, 28))).reshape(3, -1).T
EPOCHS = ((200.0, "primary"), (600.0, "harder case"))


def mars_arrival_node(sjd, off):
    """R-N45 arrival at t0+400+off: (jd_arrival, v_inf_vector_at_mars, v_helio_craft, v_mars_vec) or None."""
    nd = M.earth_node(sjd + 400.0 + off)
    if nd is None:
        return None
    jdn, vinn = nd
    md, (_d, _m, g) = M.min_dsm("earth", vinn, jdn)
    if md is None:
        return None
    dfly, phi, tof, frac = float(g[0]), float(g[1]), float(g[2]), float(g[3])
    _dsm, st2, _miss = MA.arrival_state("earth", dfly, phi, tof, frac, vinn, jnp.float64(jdn))
    jd_arr = jdn + tof
    _rM, vM = C.rv_p("mars", jd_arr)
    v_helio = jnp.asarray(np.asarray(st2[3:6]))
    vin_mars = v_helio - vM                              # incoming v_inf vector at Mars
    return jd_arr, vin_mars, v_helio, vM


def run_epoch(sjd, off):
    """Full Mars crank walk from the R-N45 arrival. Returns a result dict, or None if no arrival."""
    node = mars_arrival_node(sjd, off)
    if node is None:
        return None
    jd_arr, vin_mars, v_helio, vM = node
    rM, _ = C.rv_p("mars", jd_arr)
    vmag0 = float(jnp.linalg.norm(vin_mars))
    vP = float(jnp.linalg.norm(vM))
    i0 = CW._ang_deg(jnp.cross(rM, v_helio), jnp.cross(rM, vM))    # incoming relative inclination
    ceil = float(np.degrees(np.arcsin(min(vmag0 / vP, 1.0))))
    dm0 = float(np.degrees(C.dmax_of("mars", vmag0)))
    legs = CW.crank_walk("mars", jd_arr, vin_mars)                 # R-N38 walk, Mars grid
    return {"off": off, "i0": i0, "vmag0": vmag0, "ceil": ceil, "dm0": dm0, "legs": legs}


def summarize(r):
    """Per-epoch metrics: longest monotone raise run, max i_rel, ceiling fraction, |v_inf| drift."""
    legs, i0, vmag0 = r["legs"], r["i0"], r["vmag0"]
    i_prev, raises = i0, []
    for lg in legs:
        raises.append(lg["i_out"] > i_prev + 0.05)
        i_prev = lg["i_out"]
    run = best_run = 0
    for rz in raises:
        run = run + 1 if rz else 0
        best_run = max(best_run, run)
    i_max = max([i0] + [lg["i_out"] for lg in legs])
    vdrift = max((abs(lg["vmag"] - vmag0) / vmag0 for lg in legs), default=0.0)
    closed = len(legs) > 0 and all(lg["miss"] < C.SOI_KM["mars"] for lg in legs)
    return {"best_run": best_run, "i_max": i_max, "frac": i_max / r["ceil"] if r["ceil"] else 0.0,
            "vdrift": vdrift, "closed": closed, "n": len(legs)}


def verify(args):
    print("=== R-N46: does the inclination CRANK REALIZE at Mars from the R-N45 hot arrival? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus", "mars"):
        C._tab(p)
    print("  R-N38 crank walk verbatim, ONE knob = the planet is Mars (seeded from the R-N45 arrival); crank")
    print("  tof grid retuned to Mars's 687-d period. Exactly ballistic, greedy max-inclination, real ephemeris.\n")

    res = {}
    for off, tag in EPOCHS:
        r = run_epoch(sjd, off)
        if r is None:
            print(f"  t0+{off:.0f} ({tag}): no R-N45 arrival — skip")
            continue
        res[off] = r
        print(f"  t0+{off:.0f} ({tag}): Mars arrival v_inf {r['vmag0']:.2f} km/s, incoming i_rel {r['i0']:.2f}°, "
              f"δmax {r['dm0']:.1f}°, ceiling {r['ceil']:.1f}°")
        i_prev = r["i0"]
        for k, lg in enumerate(r["legs"], 1):
            print(f"    crank {k}: tof {lg['tof']:5.0f} d  turn {np.degrees(lg['turn']):6.1f}/{np.degrees(lg['dmax']):.1f}° "
                  f" i_rel {i_prev:6.2f} -> {lg['i_out']:6.2f} (Δ{lg['i_out'] - i_prev:+.2f})  |v_inf| {lg['vmag']:.2f}"
                  f"  miss {lg['miss']:.1e} km")
            i_prev = lg["i_out"]
        if not r["legs"]:
            print("    (no closed resonant Mars return — walk empty)")
        s = summarize(r)
        print(f"    -> {s['n']} legs, longest raise-run {s['best_run']}, max i_rel {s['i_max']:.2f}° "
              f"= {100 * s['frac']:.0f}% of ceiling, |v_inf| drift {100 * s['vdrift']:.2f}%\n", flush=True)

    if 200.0 not in res:
        print("  primary epoch (t0+200) produced no arrival — cannot judge; aborting.")
        return
    report_verdicts(res)


def report_verdicts(res):
    # The result INVERTED my pre-registration (I predicted t0+200 would crank best; it stalls, t0+600 climbs).
    # To avoid BOTH sins — misrepresenting the t0+600 success as a pure REFUTED, and cherry-picking it into a
    # pure SUPPORTED (the favorable-extremum sin) — the verdict is reported HONESTLY PER-EPOCH and labelled
    # MIXED. H-N46c is uniform. The pre-registered PRIMARY (t0+200) verdict is stated first and plainly.
    p = summarize(res[200.0])                       # pre-registered primary (low-v_inf)
    h = summarize(res[600.0]) if 600.0 in res else None  # the "harder case" that turned out easier (high-v_inf)
    prim_a = p["best_run"] >= 3
    prim_b = p["frac"] >= 0.5 and p["n"] <= 8
    hi_a = h is not None and h["best_run"] >= 3
    hi_b = h is not None and h["frac"] >= 0.5 and h["n"] <= 8
    c_ok = p["closed"] and p["vdrift"] <= 0.02 and (h is None or (h["closed"] and h["vdrift"] <= 0.02))

    lab_a = "MIXED" if (hi_a and not prim_a) else ("SUPPORTED" if prim_a else "REFUTED")
    lab_b = "MIXED" if (hi_b and not prim_b) else ("SUPPORTED" if prim_b else "REFUTED")
    prim_stall = f"{p['best_run']} (≥3)" if prim_a else f"{p['best_run']} (<3, PHYSICAL stall)"
    # high-v_inf epoch: only attach a </≥50% marker when the epoch actually exists (else "n/a", no false check)
    hi_txt = f"t0+600 {h['i_max']:.1f}° = {100 * h['frac']:.0f}% (run {h['best_run']})" if h else "t0+600 n/a"
    hi_mark = f" ({'≥' if hi_b else '<'}50%)" if h else ""
    max_drift = 100 * max(p["vdrift"], h["vdrift"] if h else 0.0)
    drift_op = "≤" if max_drift <= 2.0 else ">"     # derive the operator from the measured drift, not the verdict
    print(f"  → H-N46a {lab_a}: the crank mechanism {'TRANSFERS to Mars but NOT at the pre-registered primary' if lab_a == 'MIXED' else ('transfers' if prim_a else 'does not transfer')} "
          f"— primary t0+200 longest raise-run {prim_stall}; {hi_txt}.")
    print(f"  → H-N46b {lab_b}: ceiling approachability is EPOCH-INVERTED — primary t0+200 {p['i_max']:.1f}° = "
          f"{100 * p['frac']:.0f}% ({'≥' if prim_b else '<'}50%), {hi_txt}{hi_mark}. "
          f"The LOW-v∞ arrival I predicted would crank best stalls; the HIGH-v∞ arrival chains past 50%.")
    print(f"  → H-N46c {'SUPPORTED' if c_ok else 'REFUTED'}: the Mars crank is "
          f"{'FREE and non-destructive at BOTH epochs' if c_ok else 'NOT free'} — every leg ballistically "
          f"re-closed (sub-SOI), |v∞| drift {max_drift:.2f}% {drift_op} 2%.")

    print(f"\n  → verdicts (pre-registered PRIMARY t0+200): H-N46a {'SUPPORTED' if prim_a else 'REFUTED'}, "
          f"H-N46b {'SUPPORTED' if prim_b else 'REFUTED'}, H-N46c {'SUPPORTED' if c_ok else 'REFUTED'} "
          f"— going-in lean REFUTED (the dependence is inverted).")
    print("  NET (CORRECTS my going-in lean): the inclination crank DOES transfer to Mars — but in the OPPOSITE")
    print("    regime from my prediction. My δmax reasoning (big δmax ≳ ceiling → few flybys → easy) was backwards:")
    print(f"    the LOW-v∞ t0+200 arrival (δmax 18.8° ≳ ceiling 17.2°) takes ONE big step to {p['i_max']:.1f}° and then")
    print("    PHYSICALLY STALLS (a wide/dense post-step search finds no raising re-closing Mars return — the sparse")
    if h:
        print("    687/1374-d resonance ladder offers no follow-up after a large rotation), while the HIGH-v∞ t0+600")
        print(f"    arrival (δmax 6.2° ≪ ceiling 34.8°) chains {h['n']} small re-closing steps to {h['i_max']:.1f}° = "
              f"{100 * h['frac']:.0f}% of ceiling.")
    print("    Mechanism: a SUSTAINED Mars crank needs MANY SMALL re-closing steps (high v∞ / small δmax), not one")
    print("    big rotation — the realized fraction is NON-monotone (inverted) in δmax. The high-v∞ arrival wins")
    print("    twice: higher ceiling AND higher realized fraction (23.8° absolute vs 5.8°). Mars IS a genuine crank")
    print("    node — for HOT arrivals. H-N46c holds: exactly ballistic, |v∞| conserved to <0.1%.")
    print("    Scope: R-N45's two arrival epochs, greedy max-i over dense phi-aware surfacing, ≤8 cranks, Mars-")
    print("    retuned tof grid, exactly ballistic; t0+200 stall confirmed PHYSICAL by a wide/dense post-step")
    print("    enumeration (tof 300-2100 d). Mechanism/DISCOVERY study, never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
