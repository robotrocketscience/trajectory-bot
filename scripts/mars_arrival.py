#!/usr/bin/env python3
"""What does the pumped->DSM Mars handoff ARRIVE with — a crank node or a hot flyby? (Build N, R-N45).

R-N43 (mars_dsm.py) built the machinery to REACH Mars from a pumped Earth node with one bounded mid-course DSM
(<= 1 km/s at well-phased epochs) and reported only the reach COST (min-|DSM|). But its inner-GN solver already
propagates that pumped->DSM leg all the way to Mars's real position — so it already computes the full ARRIVAL
STATE at Mars and never surfaced it. This round reads out that one quantity, the arrival v_inf at Mars, and
characterizes it. It ties the last three rounds together (pump -> R-N43 DSM handoff -> Mars) into a single
characterized encounter.

ONE knob vs R-N43: no new capability. The identical R-N43 closed solution (same pumped node, same bounded DSM
hitting Mars exactly), now reporting the arrival v_inf plus two one-line derived diagnostics OF that same
arrival state — a crank ceiling and a capture cost.

  H-N45a  HOT arrival: hyperbolic excess at Mars v_inf_M >= 4.0 km/s — well above a Hohmann Earth->Mars arrival
          (~2.65 km/s, textbook min-energy transfer) — because the pumped orbit is highly energetic at Mars's
          ~1.5 AU radius (R-N43 aphelion 3.26 AU). REFUTE-BY: v_inf_M <= 2.65 km/s (a Hohmann-like gentle
          arrival; the DSM + geometry bleed the excess away).
  H-N45b  Mars becomes a CRANK node: the resonant single-planet inclination-crank ceiling
          arcsin(v_inf_M / v_Mars) >= 8 deg (v_Mars = Mars's heliocentric speed AT the encounter) — a real
          per-node crank, like the Venus/Earth nodes that built the R-N38 crank. REFUTE-BY: ceiling < 3 deg.
  H-N45c  it is a fast FLYBY, not a capture: the Mars-relative braking dv to capture into a bound (parabolic-
          limit) Mars orbit at periapsis r_p = 1.05*R_Mars is >> the <= 1 km/s DSM budget. REFUTE-BY: capture
          dv <= 1.0 km/s (already near-capture; cheap Mars orbit insertion falls out of the pumped handoff).

The capture dv is the parabolic-limit value v_hyp(r_p) - v_esc(r_p) — the MINIMUM braking to become barely
bound (a lower bound on any real capture); if even that exceeds the budget, capture is expensive. Measured at
R-N43's well-phased closing epochs (t0+200, t0+600). Mechanism/DISCOVERY study, characterizing what the pumped
tour DELIVERS at Mars — never a Delta-v beat of a flown mission (locked 418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/mars_arrival.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, lax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import constrained_tour_discovery as C   # noqa: E402
import mars_dsm as M                     # noqa: E402  (R-N43: reach cost; adds Mars to the architecture)
from fgprop import fg_propagate          # noqa: E402

DAY, MU_S = C.DAY, C.MU_S
MU_M = C.MU_P["mars"]                     # km^3/s^2 (added by mars_dsm)
RP_M = C.RP["mars"]                       # km, capture periapsis 1.05*R_Mars
HOHMANN_VINF = 2.65                       # km/s, textbook Earth->Mars min-energy arrival (H-N45a reference)
HOT_BAR = 4.0                             # km/s, H-N45a
CRANK_BAR = 8.0                           # deg, H-N45b
CAPTURE_BUDGET = 1.0                      # km/s, H-N45c (the R-N43 DSM budget)
EPOCH_OFFSETS = (200.0, 600.0)            # R-N43's well-phased closing epochs (relative to sjd+400)


def arrival_state(dep, dfly, phi, tof, frac, vinn, jdn):
    """Re-solve the mid-course DSM 3-vec to hit Mars exactly for a fixed grid row, return the full arrival
    heliocentric state st2 (pos+vel) at Mars — the quantity R-N43's solver computed but never reported."""
    def leg(dsm):
        dm = C.dmax_of(dep, jnp.linalg.norm(vinn))
        vout = C.rodrigues(vinn, dm * jnp.tanh(dfly), phi)
        rP, vP = C.rv_p(dep, jdn)
        st1 = fg_propagate(jnp.concatenate([rP, vP + vout]), frac * tof * DAY, mu=MU_S, iters=12)
        st1 = st1.at[3:6].add(dsm)
        return fg_propagate(st1, (1 - frac) * tof * DAY, mu=MU_S, iters=12)

    def resid(dsm):
        rM, _ = C.rv_p("mars", jdn + tof)
        return (leg(dsm)[0:3] - rM) / 1e6

    @jit
    def solve():
        def body(dsm, _):
            r = resid(dsm)
            J = jacfwd(resid)(dsm)
            return dsm - jnp.linalg.solve(J.T @ J + 1e-9 * jnp.eye(3), J.T @ r), None
        dsm, _ = lax.scan(body, jnp.zeros(3), None, length=30)
        st2 = leg(dsm)
        rM, _ = C.rv_p("mars", jdn + tof)
        miss = jnp.linalg.norm(st2[0:3] - rM)      # terminal Mars miss (km) — must be sub-SOI to trust st2
        return dsm, st2, miss
    return solve()


def characterize(jdn, vinn):
    """At a pumped Earth node, close to Mars with min-|DSM| (R-N43) and characterize the arrival. Returns a
    dict, or None if the DSM did not close at this epoch."""
    md, (_dbest, mbest, g) = M.min_dsm("earth", vinn, jdn)
    if md is None:
        return None
    dfly, phi, tof, frac = float(g[0]), float(g[1]), float(g[2]), float(g[3])
    _dsm, st2, arr_miss = arrival_state("earth", dfly, phi, tof, frac, vinn, jnp.float64(jdn))
    # the arrival diagnostics are only meaningful if st2 is actually AT Mars — assert the re-solved leg closed
    # (min_dsm already found this grid row sub-SOI; arrival_state re-solves it deterministically, so this is a
    # guard against a silent divergence, not an expected failure). Refuse to publish v_inf off a non-arrival.
    if float(arr_miss) >= C.SOI_KM["mars"]:
        raise AssertionError(f"arrival_state did not close at Mars: miss {float(arr_miss):.3e} km "
                             f">= SOI {C.SOI_KM['mars']:.3e} km (grid-min reported {mbest:.3e})")
    _rM, vM = C.rv_p("mars", jdn + tof)
    vM = np.asarray(vM)
    v_sc = np.asarray(st2[3:6])
    vinf_M = float(np.linalg.norm(v_sc - vM))
    v_mars = float(np.linalg.norm(vM))
    r_arr = float(np.linalg.norm(np.asarray(st2[0:3]))) / C.AU
    ceil_deg = float(np.degrees(np.arcsin(min(1.0, vinf_M / v_mars))))
    v_esc = float(np.sqrt(2 * MU_M / RP_M))                 # parabolic speed at r_p
    v_hyp = float(np.sqrt(vinf_M ** 2 + 2 * MU_M / RP_M))   # hyperbolic periapsis speed
    dv_capture = v_hyp - v_esc                              # parabolic-limit (minimum) capture dv
    return {"vinf_node": float(jnp.linalg.norm(vinn)), "dsm": md, "miss": float(arr_miss), "tof": tof,
            "vinf_M": vinf_M, "v_mars": v_mars, "r_arr": r_arr, "ceil": ceil_deg, "dv_cap": dv_capture}


def verify(args):
    print("=== R-N45: what does the pumped->DSM Mars handoff ARRIVE with? (crank node or hot flyby?) ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus", "mars"):
        C._tab(p)
    print("  R-N43's identical closed handoff (pumped Earth node + one bounded DSM hitting Mars exactly), now")
    print(f"  reporting the ARRIVAL v_inf at Mars and its two derived diagnostics. Hohmann ref {HOHMANN_VINF} km/s.\n")

    rows = []
    for off in EPOCH_OFFSETS:
        nd = M.earth_node(sjd + 400.0 + off)
        if nd is None:
            print(f"  t0+{off:.0f}: no Earth node")
            continue
        r = characterize(*nd)
        if r is None:
            print(f"  t0+{off:.0f}: DSM did not close — skip")
            continue
        rows.append(r)
        print(f"  t0+{off:.0f}: pumped node v_inf {r['vinf_node']:.2f}, min|DSM| {r['dsm']:.3f} km/s "
              f"(miss {r['miss']:.1e} km), tof {r['tof']:.0f} d")
        print(f"    arrival {r['r_arr']:.2f} AU, Mars helio speed {r['v_mars']:.2f} km/s | "
              f"ARRIVAL v_inf {r['vinf_M']:.2f} km/s | crank ceiling {r['ceil']:.1f} deg | "
              f"min capture dv {r['dv_cap']:.2f} km/s", flush=True)

    # Require EVERY configured epoch to close before emitting a "both epochs" verdict — a favorable extremum
    # from one epoch must not carry the claim while the other failed or was skipped (CodeRabbit). Judge each
    # hypothesis on its WORST-supporting epoch (min v_inf, min ceiling, min capture dv) so the verdict is the
    # conservative both-epochs claim, not a best-case one.
    both_present = len(rows) == len(EPOCH_OFFSETS)
    if not both_present:
        print(f"\n  ⚠ only {len(rows)}/{len(EPOCH_OFFSETS)} configured epochs closed — cannot assert a both-epochs")
        print("    verdict; treating the missing epoch as a failure of the all-epochs contract.")
    if not rows:
        print("  no closed handoff at any epoch — aborting.")
        return

    vinf_worst = min(r["vinf_M"] for r in rows)   # coldest arrival across epochs (hardest case for H-N45a)
    ceil_worst = min(r["ceil"] for r in rows)      # weakest crank across epochs (hardest case for H-N45b)
    dv_worst = min(r["dv_cap"] for r in rows)      # cheapest capture across epochs (hardest case for H-N45c)
    a_ok = both_present and all(r["vinf_M"] >= HOT_BAR for r in rows)
    b_ok = both_present and all(r["ceil"] >= CRANK_BAR for r in rows)
    c_ok = both_present and all(r["dv_cap"] > CAPTURE_BUDGET for r in rows)

    # SUPPORTED/REFUTED is governed by the all-epochs verdict (a_ok/…); the displayed inequality is derived
    # from the metric itself, so a missing-epoch REFUTED never prints a false comparison (e.g. "7.85 < 4.0").
    print(f"\n  → H-N45a {'SUPPORTED' if a_ok else 'REFUTED'}: the handoff arrives HOT at EVERY epoch — coldest "
          f"arrival v_inf {vinf_worst:.2f} km/s {'≥' if vinf_worst >= HOT_BAR else '<'} {HOT_BAR:.1f} "
          f"(vs Hohmann {HOHMANN_VINF}); the pumped orbit is highly energetic at Mars's radius.")
    print(f"  → H-N45b {'SUPPORTED' if b_ok else 'REFUTED'}: Mars is a CRANK node at EVERY epoch — weakest crank "
          f"ceiling arcsin(v_inf/v_Mars) {ceil_worst:.1f}° {'≥' if ceil_worst >= CRANK_BAR else '<'} "
          f"{CRANK_BAR:.0f}° (a real per-node crank, like Venus/Earth in R-N38).")
    print(f"  → H-N45c {'SUPPORTED' if c_ok else 'REFUTED'}: it is a fast FLYBY not a capture at EVERY epoch — "
          f"cheapest capture dv {dv_worst:.2f} km/s {'>' if dv_worst > CAPTURE_BUDGET else '≤'} the "
          f"{CAPTURE_BUDGET:.1f} km/s DSM budget (hot-arrival, not near-capture).")

    print(f"\n  → verdicts: H-N45a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N45b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N45c {'SUPPORTED' if c_ok else 'REFUTED'}")
    if a_ok and b_ok and c_ok:
        vinf_hi = max(r["vinf_M"] for r in rows)
        ceil_hi = max(r["ceil"] for r in rows)
        print("  NET: the pumped tour reaches Mars as a HOT, crank-capable FLYBY — the OPPOSITE trade from a")
        print("    Hohmann arrival. One bounded DSM (≤1 km/s, R-N43) buys the phasing to REACH Mars, but the")
        print(f"    pumped orbit's energy delivers a fast {vinf_worst:.1f}–{vinf_hi:.1f} km/s hyperbolic excess "
              f"(≫ Hohmann's {HOHMANN_VINF}) —")
        print(f"    enough to make Mars a genuine inclination-crank node (ceiling {ceil_worst:.0f}–{ceil_hi:.0f}°), "
              f"while capture")
        print(f"    into a bound Mars orbit would cost ≥ {dv_worst:.1f} km/s of braking (≫ the reach budget). So the")
        print("    pumped/crank tour EXTENDS to Mars as a crank node, but Mars orbit insertion is a separate,")
        print("    expensive maneuver the free flyby budget does not afford — cheap-to-REACH, hot-to-ARRIVE.")
    print("    Scope: R-N43's two well-phased epochs, grid-min DSM (upper bound), parabolic-limit capture dv")
    print("    (lower bound), resonant-crank ceiling as a per-node availability proxy. Never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
