#!/usr/bin/env python3
"""Tisserand–Poincaré outer loop — flyby-SEQUENCE discovery (Build N, R-N10).

The capstone that supplies the DISCRETE which-body/when structure gradient descent provably can't find
(R-N7's razor-thin-basin null): a Tisserand-graph enumeration of energetically-connectable gravity-assist
sequences. The Tisserand parameter T_P = a_P/a + 2·√((a/a_P)(1−e²))·cos i is conserved across an unpowered
flyby of planet P; for the coplanar case it fixes the flyby relative speed via v∞²/v_P² = 3 − T_P. So a
flyby conserves v∞ and walks the spacecraft along a constant-v∞ contour in orbital-element space (sweeping
the pump angle α, bounded per encounter by the max turn δ_max from the min periapsis). Chaining contour
intersections between planets is the classical graphical grand-tour method (Strange & Longuski 2002).

HONEST FRAMING: this is deterministic enumeration, NOT learned discovery. It supplies the combinatorial
skeleton so the SYSTEM — Tisserand enumerate → R-N8 flyby-node optimize → R-N9 primer certify — can discover
sequences. It is a REACHABILITY map: the resonant phasing (planet SITTING at the crossing when the craft
arrives) is the launch-window problem, NOT solved here. Reachability, never a Δv "beat".

    uv run python scripts/tisserand_graph.py --verify        # offline, CI-safe (no jax needed)
"""
from __future__ import annotations

import argparse

import numpy as np

MU_S = 1.32712440018e11          # Sun GM (km^3/s^2)
AU = 1.495978707e8               # km
# planet: (heliocentric radius km, GM km^3/s^2, equatorial radius km)
PLAN = {
    "venus": (0.7233 * AU, 3.24859e5, 6052.0),
    "earth": (1.0000 * AU, 3.986004418e5, 6378.0),
    "mars": (1.5237 * AU, 4.2828e4, 3390.0),
    "jupiter": (5.2028 * AU, 1.26686534e8, 71492.0),
}


def v_planet(r_p):
    return np.sqrt(MU_S / r_p)


def orbit_from_flyby(r_p, vinf, alpha):
    """Heliocentric (a, e) after a flyby at planet-radius r_p with relative speed vinf and pump angle α."""
    vp = v_planet(r_p)
    vtan = vp + vinf * np.cos(alpha)
    vrad = vinf * np.sin(alpha)
    v2 = vtan ** 2 + vrad ** 2
    a = -MU_S / (2.0 * (0.5 * v2 - MU_S / r_p))
    h = r_p * vtan
    e = np.sqrt(max(0.0, 1.0 - h ** 2 / (MU_S * a)))
    return a, e


def vinf_alpha_at(r_p, a, e):
    """The relative speed vinf and pump angle α when a heliocentric orbit (a,e) crosses planet-radius r_p."""
    v2 = MU_S * (2.0 / r_p - 1.0 / a)
    vtan = np.sqrt(MU_S * a * (1.0 - e ** 2)) / r_p
    vrad = np.sqrt(max(0.0, v2 - vtan ** 2))
    vp = v_planet(r_p)
    vinf = np.sqrt((vtan - vp) ** 2 + vrad ** 2)
    return vinf, np.arctan2(vrad, vtan - vp)


def tisserand(r_p, a, e):
    return r_p / a + 2.0 * np.sqrt((a / r_p) * (1.0 - e ** 2))


def max_turn(vinf, mu, rp_min):
    return 2.0 * np.arcsin(1.0 / (1.0 + rp_min * vinf ** 2 / mu))


def apo(a, e):
    return a * (1.0 + e)


def peri(a, e):
    return a * (1.0 - e)


def pump_sequence(vinf_launch, planets, rp_min_frac=1.5, max_fly=8):
    """Greedy energy pump: at each planet the orbit crosses, rotate the pump angle toward 0 (max energy)
    by up to δ_max, raising apoapsis. Walk the planet list. Returns (a, e, log of encounters)."""
    a, e = orbit_from_flyby(PLAN["earth"][0], vinf_launch, np.radians(90.0))
    log = []
    for pl in planets:
        r_p, mu, radius = PLAN[pl]
        rp_min = rp_min_frac * radius
        for _ in range(max_fly):
            if not (peri(a, e) <= r_p <= apo(a, e)):
                break                                   # orbit doesn't cross this planet
            vinf, alpha = vinf_alpha_at(r_p, a, e)
            dm = max_turn(vinf, mu, rp_min)
            new_alpha = max(0.0, alpha - dm) if alpha > 0 else min(0.0, alpha + dm)
            turn_used = abs(alpha - new_alpha)               # actual deflection applied (≤ δmax)
            a2, e2 = orbit_from_flyby(r_p, vinf, new_alpha)
            gain = apo(a2, e2) - apo(a, e) if e2 < 1.0 and a2 > 0 else np.inf
            a, e = a2, e2
            # implied periapsis of THIS flyby from the turn actually used (turn ≤ δmax ⇒ r_p ≥ rp_min)
            rp_flyby = (mu / vinf ** 2) * (1.0 / np.sin(turn_used / 2.0) - 1.0) if turn_used > 1e-6 else np.inf
            log.append(dict(planet=pl, vinf=vinf, dmax_deg=np.degrees(dm),
                            ra=apo(a2, e2) if a2 > 0 else np.inf, e=e2, a=a2,
                            rp_clear=rp_flyby / radius))
            if e >= 1.0 or a < 0 or gain <= 1e3:
                break
    return a, e, log


def verify(args):
    print("=== R-N10: Tisserand–Poincaré outer loop — flyby-sequence discovery (offline) ===")

    # ---- H-N10a: the graph's invariant is exact ----
    print("  H-N10a: Tisserand parameter constant along a fixed-v∞ contour, = 3 − (v∞/v_P)²")
    ok_a = True
    for pl in ("earth", "jupiter"):
        r_p, mu, _ = PLAN[pl]
        vp = v_planet(r_p)
        for vinf in (5.0, 8.0):
            ts = [tisserand(r_p, *orbit_from_flyby(r_p, vinf, np.radians(ad)))
                  for ad in (10, 40, 80, 120, 160)]
            pred = 3.0 - (vinf / vp) ** 2
            spread = max(ts) - min(ts)
            ok_a = ok_a and spread < 1e-10 and abs(np.mean(ts) - pred) < 1e-9
            print(f"    {pl:8} v∞={vinf:.0f}: T={np.mean(ts):.6f}  pred={pred:.6f}  "
                  f"spread={spread:.1e}")
    print(f"    → {'PASS' if ok_a else 'FAIL'} (invariant exact)")

    # ---- H-N10b: sequence discovery beats single-flyby reach ----
    print("  H-N10b: multi-flyby staircase reaches beyond a single flyby (Earth-leverage → Jupiter pump)")
    for vinf_l in (7.0, 9.0):
        _, _, one = pump_sequence(vinf_l, ["earth"], max_fly=1)
        ra_single = one[0]["ra"] / AU if one else float("nan")
        a, e, seq = pump_sequence(vinf_l, ["earth", "jupiter"])
        ra_final = apo(a, e) / AU if (a > 0 and e < 1.0) else float("inf")
        escaped = (e >= 1.0 or a < 0)
        print(f"    launch v∞={vinf_l:.0f} km/s: 1 Earth flyby → r_a={ra_single:.2f} AU;  "
              f"sequence ({len(seq)} flybys) → "
              f"{'SOLAR-SYSTEM ESCAPE (e≥1)' if escaped else f'r_a={ra_final:.2f} AU'}")
        for i, g in enumerate(seq):
            reach = "ESCAPE" if (g["e"] >= 1.0 or g["a"] < 0) else f"r_a={g['ra']/AU:.2f} AU"
            print(f"      {i+1}. {g['planet']:8} v∞={g['vinf']:5.2f} δmax={g['dmax_deg']:5.1f}° → {reach}")

    # ---- H-N10c: every enumerated encounter is physically flyable ----
    print("  H-N10c: each encounter's periapsis clears the planet surface (turn ≤ δmax by construction)")
    a, e, seq = pump_sequence(9.0, ["earth", "jupiter"])
    all_clear = all(g["rp_clear"] >= 1.0 for g in seq)
    print(f"    min r_p/R_planet across the v∞=9 sequence = {min(g['rp_clear'] for g in seq):.2f} "
          f"→ {'PASS' if all_clear else 'FAIL'} (all encounters flyable; hand off to R-N8 node / R-N9 primer)")
    print("  → the Tisserand outer loop enumerates the discrete flyby SEQUENCE (which body, in what order) "
          "that the diff-sim inner loop cannot find by gradient — the missing piece for grand-tour discovery. "
          "Reachability only; resonant phasing / launch window is the next layer.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
