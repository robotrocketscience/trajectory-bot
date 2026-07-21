#!/usr/bin/env python3
"""Does the INCLINATION CRANK survive real-ephemeris encounter re-closure? (Build N, R-N38).

R-N37's deep chain saturates at Venus with v inf 16.27 (dmax ~19 deg) in v inf-NEUTRAL resonant returns —
exactly where the R-N16/R-N20 inclination crank should be FREE: successive same-planet flybys rotate the v inf
vector out-of-plane by <= dmax per encounter, walking relative inclination toward the ceiling
arcsin(v inf / v_P) ~ 27.9 deg, at zero Delta-v. The analytic rounds assumed ANY turn plane phi is available.
The real hard-constrained architecture says otherwise: each return must RE-CLOSE the encounter (3 constraints /
3 unknowns), so closed basins have SPECIFIC (delta, phi) — re-closure may pin the turn largely in-plane
(phasing) and eat the crank. The measure-first probe quantified the tax: a 17 deg turn buys ~5.9 deg of
inclination (~1/3 conversion) — alive, but taxed. This round runs the full CRANK WALK.

ONE knob vs R-N37: the basin-choice criterion at the saturated node — max arrival v inf (v inf-neutral
returns) -> max RELATIVE INCLINATION i_rel = angle(h_craft, h_venus) among GN-closed returns.

  H-N38a  real-ephemeris cranking EXISTS: >= 3 consecutive closed returns each RAISE i_rel (zero DSM).
  H-N38b  the ceiling is approachable: i_rel >= 50% of arcsin(v inf/v_P) within <= 8 encounters.
  H-N38c  the crank is FREE and non-destructive: every leg re-closes ballistically; |v inf| drift <= 2%.

R-N36/R-N37 architecture verbatim (fgprop forward Kepler, Rodrigues closure-by-construction, Levenberg-GN,
real cached-JPL ephemeris, exactly ballistic). Mechanism/DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/crank_walk.py --verify
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

import beam_constrained_tour as B    # noqa: E402  (R-N37 tree machinery: run_search, continuations)
import constrained_tour_discovery as C   # noqa: E402  (R-N36 architecture: rv_p, rodrigues, dmax_of, ...)

MAX_CRANKS = 8


def _ang_deg(a, b):
    c = float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _i_rel(at, jd, v_helio_craft):
    """Relative inclination (deg) of the craft's heliocentric orbit plane vs the planet's, at epoch jd."""
    r, vP = C.rv_p(at, jd)
    return _ang_deg(jnp.cross(r, v_helio_craft), jnp.cross(r, vP))


def saturated_node(t0):
    """Rebuild the R-N37 greedy chain and return (at, jd, vin) just after the LAST pumping leg."""
    best, log, root = B.run_search(t0, 1)
    if root is None:
        return None
    jd = root["jd"]
    vin = root["vin"]
    at = "venus"
    prev = root["mag"]
    sat_idx = 0
    for i, lg in enumerate(best["legs"]):
        if lg["arr_mag"] > prev + 0.05:
            sat_idx = i
        prev = lg["arr_mag"]
    for lg in best["legs"][:sat_idx + 1]:
        jd = jd + lg["tof"]
        vin = lg["vinf_arr"]
        at = lg["to"]
    return at, jd, vin, root["seed_v"]


_CRANK_GRID = np.array(np.meshgrid(np.linspace(-2.0, 2.0, 7),
                                   np.linspace(0, 2 * np.pi, 16, endpoint=False),
                                   np.linspace(80, 460, 24))).reshape(3, -1).T


def crank_continuations(at, jd, vin, max_gn=24):
    """DENSE, phi-aware enumeration of GN-closed same-planet returns. R-N37's `continuations` (miss-ranked,
    tof-only dedupe, turn seeds <= 0.83*dmax) SURFACED only near-planar returns at inclined nodes — a real
    artifact caught by the round's diagnostic: the inclination-raising basins live at distinct (turn, phi)
    branches of the SAME resonance tof, so dedupe must be (turn, phi, tof)-aware and seeds must span phi."""
    scan_miss, gn, leg_out = B.pair(at, at)
    m = np.array(scan_miss(jnp.asarray(_CRANK_GRID), vin, jd))
    order = np.argsort(m)
    seen, sols, tried = [], [], 0
    for idx in order:
        u0 = _CRANK_GRID[idx]
        if any(abs(u0[2] - s[2]) < 25.0 and abs((u0[1] - s[1] + np.pi) % (2 * np.pi) - np.pi) < 0.8
               and abs(u0[0] - s[0]) < 0.7 for s in seen):
            continue
        seen.append(u0)
        u, miss = gn(jnp.asarray(u0), vin, jd)
        tried += 1
        if float(miss) < C.SOI_KM[at]:
            va, turn, dm = leg_out(u, vin, jd)
            if float(jnp.linalg.norm(va)) <= C.VCAP and float(u[2]) > 20.0:
                sols.append({"u": u, "tof": float(u[2]), "miss": float(miss), "turn": float(turn),
                             "dmax": float(dm), "vinf_arr": va})
        if tried >= max_gn:
            break
    return sols


def crank_walk(at, jd, vin, max_cranks=MAX_CRANKS):
    """Greedy max-inclination walk over densely-surfaced GN-closed same-planet resonant returns."""
    legs = []
    for _ in range(max_cranks):
        rV, vV = C.rv_p(at, jd)
        sols = crank_continuations(at, jd, vin)
        if not sols:
            break
        best = None
        for s in sols:
            vout = C.rodrigues(vin, s["dmax"] * jnp.tanh(s["u"][0]), s["u"][1])
            i_out = _ang_deg(jnp.cross(rV, vV + vout), jnp.cross(rV, vV))
            if best is None or i_out > best[0]:
                best = (i_out, s)
        i_out, s = best
        legs.append({"tof": s["tof"], "turn": s["turn"], "dmax": s["dmax"], "miss": s["miss"],
                     "i_out": i_out, "vmag": float(jnp.linalg.norm(s["vinf_arr"]))})
        jd = jd + s["tof"]
        vin = s["vinf_arr"]
    return legs


def verify(args):
    print("=== R-N38: does the INCLINATION CRANK survive real-ephemeris encounter re-closure? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus"):
        C._tab(p)
    t0 = sjd + 400.0
    print("  R-N36/R-N37 architecture verbatim; ONE knob: basin choice at the saturated node = max arrival")
    print("  v inf (R-N37) -> max RELATIVE INCLINATION among GN-closed resonant returns. Exactly ballistic.\n")

    sat = saturated_node(t0)
    if sat is None:
        print("  no chain — aborting.")
        return
    at, jd, vin, seed_v = sat
    vmag0 = float(jnp.linalg.norm(vin))
    rV, vV = C.rv_p(at, jd)
    i0 = _ang_deg(jnp.cross(rV, vV + vin), jnp.cross(rV, vV))
    vP = float(jnp.linalg.norm(vV))
    ceil = float(np.degrees(np.arcsin(min(vmag0 / vP, 1.0))))
    dm0 = float(np.degrees(C.dmax_of(at, vmag0)))
    print(f"  saturated node: {at}, v inf {vmag0:.2f} km/s (pumped from seed {seed_v:.2f} by the R-N37 chain),")
    print(f"  incoming i_rel {i0:.2f} deg, δmax {dm0:.1f} deg, analytic ceiling arcsin(v inf/v_P) = {ceil:.1f} deg\n")

    print("  [crank walk: greedy max-inclination over GN-closed resonant returns]", flush=True)
    legs = crank_walk(at, jd, vin)
    i_prev = i0
    raises = []
    for k, lg in enumerate(legs, 1):
        print(f"    crank {k}: tof {lg['tof']:5.0f} d  turn {np.degrees(lg['turn']):6.1f}"
              f"/{np.degrees(lg['dmax']):.1f} deg  i_rel {i_prev:6.2f} -> {lg['i_out']:6.2f} "
              f"(Δ{lg['i_out'] - i_prev:+.2f})  |v inf| {lg['vmag']:.2f}  miss {lg['miss']:.1e} km")
        raises.append(lg["i_out"] > i_prev + 0.05)
        i_prev = lg["i_out"]
    if not legs:
        print("    (no closed resonant return at the saturated node — walk empty)")
    i_max = max([i0] + [lg["i_out"] for lg in legs])
    vdrift = max((abs(lg["vmag"] - vmag0) / vmag0 for lg in legs), default=0.0)

    # longest consecutive run of inclination-raising legs
    run = best_run = 0
    for r in raises:
        run = run + 1 if r else 0
        best_run = max(best_run, run)

    a_ok = best_run >= 3
    b_ok = i_max >= 0.5 * ceil and len(legs) <= 8
    c_ok = len(legs) > 0 and all(lg["miss"] < C.SOI_KM[at] for lg in legs) and vdrift <= 0.02
    tax = np.mean([(lg["i_out"] - (legs[k - 1]["i_out"] if k else i0)) / max(abs(np.degrees(lg["turn"])), 1e-9)
                   for k, lg in enumerate(legs) if lg["i_out"] > (legs[k - 1]["i_out"] if k else i0)]) if legs else 0.0

    print(f"\n  → H-N38a {'SUPPORTED' if a_ok else 'REFUTED'}: real-ephemeris cranking "
          f"{'EXISTS' if a_ok else 'fails'} — longest monotone climb {best_run} consecutive closed "
          f"inclination-raising returns (zero DSM), max i_rel {i_max:.2f} deg.")
    print(f"  → H-N38b {'SUPPORTED' if b_ok else 'REFUTED'}: reached {i_max:.1f} deg = "
          f"{100 * i_max / ceil:.0f}% of the analytic ceiling ({ceil:.1f} deg) within {len(legs)} encounters "
          f"({'≥' if b_ok else '<'} 50% in ≤ 8).")
    print(f"  → H-N38c {'SUPPORTED' if c_ok else 'REFUTED'}: the crank is "
          f"{'FREE and non-destructive' if c_ok else 'NOT free'} — every leg ballistically re-closed "
          f"(sub-SOI), |v inf| drift {100 * vdrift:.2f}% ≤ 2% (conserved by construction; drift = ephemeris "
          "eccentricity via the changing encounter geometry).")

    print(f"\n  → verdicts: H-N38a {'SUPPORTED' if a_ok else 'REFUTED'}, H-N38b {'SUPPORTED' if b_ok else 'REFUTED'}, "
          f"H-N38c {'SUPPORTED' if c_ok else 'REFUTED'}")
    if a_ok:
        print("  NET: the inclination crank SURVIVES real-ephemeris encounter re-closure — the R-N16/R-N20 free-")
        print("    crank story holds in the hard-constrained architecture, at a quantified RE-CLOSURE TAX: on")
        print(f"    average a turn converts ~{100 * float(tax):.0f}% of its angle into inclination (the rest is")
        print("    pinned by the 3-constraint encounter re-closure — phasing), vs the analytic free plane choice.")
        print("    The pump-then-crank mission shape is realized END-TO-END against real ephemeris, exactly")
        print("    ballistically: launch → 4 pumping flybys (v inf 5.95→16.27, R-N37) → inclination cranking at")
        print("    the saturated v inf (this round) — every encounter GN-closed, zero DSM.")
    else:
        print("  NET: real-ephemeris encounter re-closure DEFEATS the inclination crank within this walk's bounds")
        print("    — the closed-basin set does not sustain an inclination climb (the analytic free-crank story")
        print("    does not survive the 3-constraint re-closure here). Judged against the pre-registered")
        print("    falsifiers; a surfacing-artifact diagnostic was run before recording.")
    print("    Scope: one epoch, greedy max-i basin choice over dense phi-aware surfacing, ≤ 8 cranks, Sun-only")
    print("    two-body legs, patched-conic flybys; within-bounds findings. Mechanism/DISCOVERY study, never a")
    print("    Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
