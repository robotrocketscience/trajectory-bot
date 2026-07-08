#!/usr/bin/env python3
"""Launch-window / phasing search — turn a Tisserand sequence into a date-specific transfer (Build N, R-N12).

R-N11 ran the pipeline at an UN-SEARCHED epoch: Earth-Jupiter sweep ~59° (badly phased) → expensive transfer
(dep Δv 33 km/s). R-N12 is the porkchop launch-window search that finds WELL-PHASED windows. Analytic
circular coplanar planets (offline/CI-safe, clean synodic structure): grid over (launch epoch, TOF), Lambert
per cell, minimize departure Δv; verify the windows recur at the Earth–Jupiter synodic period; and separate
two things R-N11 conflated — PHASING quality (dep Δv) vs DSM-optimality (the primer).

    uv run --with jax python scripts/launch_window.py --verify        # offline, CI-safe
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp               # noqa: E402
import lambert as LAM                 # noqa: E402
import primer_vector as PV            # noqa: E402

MU_S = 1.32712440018e11
AU = 1.495978707e8
DAY = 86400.0
R_E, R_J = 1.0 * AU, 5.2028 * AU
W_E, W_J = np.sqrt(MU_S / R_E ** 3), np.sqrt(MU_S / R_J ** 3)
VC_E, VC_J = np.sqrt(MU_S / R_E), np.sqrt(MU_S / R_J)
PHI_J0 = np.radians(100.0)
T_SYN = 2 * np.pi / (W_E - W_J) / DAY          # Earth–Jupiter synodic period (days)


def states(t0, tof):
    pe = W_E * t0
    pj = PHI_J0 + W_J * (t0 + tof)
    r1 = R_E * np.array([np.cos(pe), np.sin(pe), 0.0])
    vc1 = VC_E * np.array([-np.sin(pe), np.cos(pe), 0.0])
    r2 = R_J * np.array([np.cos(pj), np.sin(pj), 0.0])
    vc2 = VC_J * np.array([-np.sin(pj), np.cos(pj), 0.0])
    sweep = np.degrees(np.arctan2(np.sin(pj - pe), np.cos(pj - pe))) % 360
    return r1, vc1, r2, vc2, sweep


def dep_dv(t0, tof):
    r1, vc1, r2, _, sweep = states(t0, tof)
    try:
        vd, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=MU_S)
        return float(np.linalg.norm(np.asarray(vd) - vc1)), sweep
    except Exception:
        return np.nan, np.nan


def porkchop(launch_days, tof_days):
    """Return min dep Δv (over TOF) per launch day, and the global-best (day, tof, Δv, sweep)."""
    best_per_day = []
    gbest = (np.inf, None)
    for d in launch_days:
        t0 = d * DAY
        row = [(dep_dv(t0, tf * DAY)[0], tf) for tf in tof_days]
        row = [(v, tf) for v, tf in row if np.isfinite(v)]
        mv, mtf = min(row)
        best_per_day.append((d, mv, mtf))
        if mv < gbest[0]:
            _, sw = dep_dv(t0, mtf * DAY)
            gbest = (mv, (d, mtf, mv, sw))
    return np.array(best_per_day), gbest[1]


def verify(args):
    print("=== R-N12: launch-window / phasing search — Earth→Jupiter porkchop (offline) ===")
    print(f"  Earth–Jupiter synodic period T_syn = {T_SYN:.1f} d; Hohmann ideal dep Δv ≈ 8.79 km/s (sweep 180°)")

    # ---- H-N12a: the search finds well-phased windows recurring at the synodic period ----
    days = np.arange(0, 3 * T_SYN, 12.0)
    tofs = np.arange(750, 1300, 25)
    bpd, gbest = porkchop(days, tofs)
    d0, tof0, dv0, sw0 = gbest
    # local minima = launch windows
    mins = [bpd[i, 0] for i in range(1, len(bpd) - 1)
            if bpd[i, 1] < bpd[i - 1, 1] and bpd[i, 1] < bpd[i + 1, 1] and bpd[i, 1] < 12]
    spac = np.diff(mins) if len(mins) > 1 else [np.nan]
    print(f"  H-N12a: global-best window — launch day {d0:.0f}, dep Δv={dv0:.3f} km/s (ideal 8.79), "
          f"TOF={tof0:.0f} d, sweep={sw0:.0f}°")
    print(f"          launch windows at days {[f'{m:.0f}' for m in mins]}; spacing "
          f"{[f'{s:.0f}' for s in spac]} d vs T_syn={T_SYN:.0f} d — "
          f"{'MATCH' if len(spac) and abs(spac[0]-T_SYN) < 40 else 'CHECK'}")
    print(f"          anti-window (worst) dep Δv = {bpd[:, 1].max():.1f} km/s (the R-N11 regime)")

    # ---- H-N12b: primer measures DSM-optimality, NOT phasing quality (CORRECTION) ----
    print("  H-N12b [CORRECTED]: the primer certifies DSM-optimality, which is ORTHOGONAL to phasing:")
    print(f"    {'window':>16} {'sweep':>6} {'depΔv':>7} {'|p|max':>7} {'interpretation':>34}")
    iworst = int(np.argmax(bpd[:, 1]))
    cases = [("GOOD (min-Δv)", d0, tof0),
             ("GOOD neighbour", d0 + 15, 900),
             ("BAD anti-window", bpd[iworst, 0], bpd[iworst, 2])]
    for tag, d, tf in cases:
        r1, vc1, r2, vc2, sweep = states(d * DAY, tf * DAY)
        dv, _ = dep_dv(d * DAY, tf * DAY)
        _, _, pmax, _, _, _ = PV.transfer_primer(r1, r2, vc1, vc2, MU_S, tf * DAY, args.steps)
        interp = "cheap AND DSM-optimal" if dv < 12 and pmax <= 1.05 else \
                 ("EXPENSIVE yet DSM-optimal" if pmax <= 1.05 else "a DSM would help")
        print(f"    {tag:>16} {sweep:6.0f} {dv:7.2f} {pmax:7.3f} {interp:>34}")
    print("    → phasing quality = dep Δv (this search); DSM-optimality = |p| (R-N9). DIFFERENT metrics: a")
    print("      badly-phased EXPENSIVE window can still be primer-optimal. R-N11's |p|=3.14 was a specific")
    print("      contorted geometry where a DSM genuinely helped — NOT a generic bad-phasing signature.")

    # ---- H-N12c: resonant-return phasing for the v∞-leveraging staircase ----
    print("  H-N12c: v∞-leveraging needs the craft to RETURN to Earth — only on near-resonant orbits:")
    T_earth = 2 * np.pi / W_E / DAY
    print(f"    {'pumped orbit a(AU)':>18} {'period(yr)':>11} {'P/T_earth':>10} {'nearest k:n':>12}")
    for ra_au in (2.0, 2.6, 3.5, 5.0):          # apoapsis after successive Earth flybys (peri≈1 AU)
        a = 0.5 * (R_E + ra_au * AU)
        P = 2 * np.pi * np.sqrt(a ** 3 / MU_S) / DAY
        ratio = P / T_earth
        # nearest low-order resonance k:n (craft does n orbits while Earth does k)
        best_kn, best_err = None, 1e9
        for n in range(1, 5):
            for k in range(1, 9):
                if abs(ratio - k / n) < best_err:
                    best_err, best_kn = abs(ratio - k / n), (k, n)
        print(f"    {ra_au:18.1f} {P/365.25:11.2f} {ratio:10.3f} "
              f"{f'{best_kn[0]}:{best_kn[1]} (err {best_err:.2f})':>12}")
    print("    → the leveraging orbits sit near low-order resonances; a resonant return re-phases the craft")
    print("      to Earth for the next flyby (the classic v∞-leveraging mechanism). Reachability→phasing closed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
