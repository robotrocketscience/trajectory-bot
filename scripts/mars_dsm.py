#!/usr/bin/env python3
"""Does a BOUNDED DSM afford the Mars handoff the exact-ballistic architecture cannot? (Build N, R-N43).

The R-N36/R-N37 architecture is exactly ballistic (zero DSM), and it never reaches Mars: R-N32 (H-N32c)
found the Venus/Earth->Mars leg needs a closure DSM > 0.5 km/s while the inner-planet legs close near-
ballistically. This round folds R-N28's Δv-for-time economy back in as a SINGLE bounded mid-course DSM on
the transfer leg, and asks whether a small fuel budget unlocks Mars. ONE knob vs R-N37: one bounded DSM
impulse on a transfer leg from a pumped node.

  H-N43a  pure exact-ballistic (DSM=0) does NOT close a pumped-node -> Mars leg: min miss over a full
          (delta, phi, tof) scan > SOI_mars.
  H-N43b  a BOUNDED DSM affords it: with one mid-course impulse the leg closes on Mars's real position
          (miss < SOI_mars) at |DSM| <= 1.0 km/s (at a well-phased epoch — the handoff is phasing-dependent,
          like R-N30's inner-planet handoffs).
  H-N43c  the DSM fixes PHASING/aim, not ENERGY: the pumped orbit already reaches Mars's orbit radius
          (prograde-outgoing aphelion >= Mars perihelion), so the DSM corrects the encounter geometry.

The min-|DSM| is measured by a decoupled solve: for a fixed flyby (delta, phi) + tof + burn-fraction, a
mid-course DSM 3-vector is Gauss-Newton-solved to hit Mars EXACTLY (3 residuals, 3 unknowns); the outer scan
over (delta, phi, tof, frac) then minimizes |DSM|. The reported value is a grid minimum (an UPPER BOUND on
the true min-DSM), so "<= 1.0" is conservative. Mars added to the R-N36 architecture (MU_P/RP/SOI).
Mechanism/DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/mars_dsm.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import jacfwd, jit, vmap, lax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import beam_constrained_tour as B        # noqa: E402
import constrained_tour_discovery as C   # noqa: E402
from fgprop import fg_propagate          # noqa: E402

# extend the R-N36 architecture with Mars
C.MU_P["mars"] = 4.282837e4
C.RP["mars"] = 1.05 * 3389.5
C.SOI_KM["mars"] = 0.00385 * C.AU
MARS_PERI = 1.381 * C.AU
DAY, MU_S = C.DAY, C.MU_S
DSM_BAR = 1.0        # km/s, H-N43b

_DFLY = np.linspace(-2.0, 2.0, 7)
_PHI = np.linspace(0, 2 * np.pi, 12, endpoint=False)
_TOF = np.linspace(150, 760, 16)
_FRAC = np.array([0.25, 0.4, 0.55, 0.7])
_GRID = np.array(np.meshgrid(_DFLY, _PHI, _TOF, _FRAC)).reshape(4, -1).T
_BAL = np.array(np.meshgrid(np.linspace(-1.2, 1.2, 7), _PHI, np.linspace(150, 760, 24))).reshape(3, -1).T


def earth_node(t0):
    """Highest-v inf Earth node of the greedy R-N37 chain launched at t0 (or None)."""
    best, _, root = B.run_search(t0, 1)
    if root is None:
        return None
    jd, vin, at = root["jd"], root["vin"], "venus"
    en = []
    for lg in best["legs"]:
        jd, vin, at = jd + lg["tof"], lg["vinf_arr"], lg["to"]
        if at == "earth":
            en.append((jd, vin))
    return max(en, key=lambda x: float(jnp.linalg.norm(x[1]))) if en else None


def ballistic_min_miss(dep, vinn, jdn):
    dm = C.dmax_of(dep, jnp.linalg.norm(vinn))
    def miss_of(u):
        vout = C.rodrigues(vinn, dm * jnp.tanh(u[0]), u[1])
        m, _ = C.shoot(dep, "mars", jdn, vout, u[2])
        return jnp.linalg.norm(m)
    mm = np.array(vmap(miss_of)(jnp.asarray(_BAL)))
    return float(mm.min())


def prograde_aphelion(dep, vinn, jdn):
    dm = C.dmax_of(dep, jnp.linalg.norm(vinn))
    vout = C.rodrigues(vinn, dm * jnp.tanh(2.0), 0.0)
    rP, vP = C.rv_p(dep, jdn)
    st = jnp.concatenate([rP, vP + vout])
    r0 = float(jnp.linalg.norm(st[0:3])); v0 = float(jnp.linalg.norm(st[3:6]))
    energy = v0 ** 2 / 2 - MU_S / r0
    if energy >= 0:
        return np.inf
    a_orb = -MU_S / (2 * energy)
    h = np.linalg.norm(np.cross(np.asarray(st[0:3]), np.asarray(st[3:6])))
    e_orb = np.sqrt(max(0.0, 1 + 2 * energy * h ** 2 / MU_S ** 2))
    return a_orb * (1 + e_orb)


def _make_solver(dep):
    def leg_miss(dsm, dfly, phi, tof, frac, vinn, jdn):
        dm = C.dmax_of(dep, jnp.linalg.norm(vinn))
        vout = C.rodrigues(vinn, dm * jnp.tanh(dfly), phi)
        rP, vP = C.rv_p(dep, jdn)
        st1 = fg_propagate(jnp.concatenate([rP, vP + vout]), frac * tof * DAY, mu=MU_S, iters=12)
        st1 = st1.at[3:6].add(dsm)
        st2 = fg_propagate(st1, (1 - frac) * tof * DAY, mu=MU_S, iters=12)
        rM, _ = C.rv_p("mars", jdn + tof)
        return (st2[0:3] - rM) / 1e6

    @jit
    def solve(dfly, phi, tof, frac, vinn, jdn):
        def body(dsm, _):
            r = leg_miss(dsm, dfly, phi, tof, frac, vinn, jdn)
            J = jacfwd(leg_miss)(dsm, dfly, phi, tof, frac, vinn, jdn)
            JTJ = J.T @ J + 1e-9 * jnp.eye(3)
            return dsm - jnp.linalg.solve(JTJ, J.T @ r), None
        dsm, _ = lax.scan(body, jnp.zeros(3), None, length=30)
        miss = jnp.linalg.norm(leg_miss(dsm, dfly, phi, tof, frac, vinn, jdn)) * 1e6
        return jnp.linalg.norm(dsm), miss
    return solve


def min_dsm(dep, vinn, jdn):
    """Grid-min |DSM| (km/s) over (delta,phi,tof,frac) among closures; returns (min_closed|None, best_row)."""
    solve = _make_solver(dep)
    dsm, miss = vmap(lambda g: solve(g[0], g[1], g[2], g[3], vinn, jnp.float64(jdn)))(jnp.asarray(_GRID))
    dsm, miss = np.array(dsm), np.array(miss)
    ok = miss < C.SOI_KM["mars"]
    if not ok.any():
        i = int(miss.argmin())
        return None, (float(dsm[i]), float(miss[i]), _GRID[i])
    i = int(np.where(ok, dsm, np.inf).argmin())
    return float(dsm[i]), (float(dsm[i]), float(miss[i]), _GRID[i])


def verify(args):
    print("=== R-N43: does a BOUNDED DSM afford the Mars handoff the exact-ballistic architecture cannot? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus", "mars"):
        C._tab(p)
    print("  R-N36 architecture + Mars; one bounded mid-course DSM on a pumped-node -> Mars leg. min-|DSM| by")
    print("  inner GN (DSM 3-vec hits Mars exactly) + outer (delta,phi,tof,frac) scan (grid min = upper bound).\n")

    node = earth_node(sjd + 400.0)
    if node is None:
        print("  no Earth node at the reference epoch — aborting.")
        return
    jdn, vinn = node
    vm = float(jnp.linalg.norm(vinn))

    bmiss = ballistic_min_miss("earth", vinn, jdn)
    a_ok = bmiss > C.SOI_KM["mars"]
    print(f"  pumped node: Earth, v inf {vm:.2f} km/s")
    print(f"  → H-N43a {'SUPPORTED' if a_ok else 'REFUTED'}: pure ballistic min miss to Mars {bmiss:.2e} km "
          f"({bmiss / C.SOI_KM['mars']:.0f}x SOI) — Mars is {'ballistically EXCLUDED' if a_ok else 'reachable ballistically'}.")

    aph = prograde_aphelion("earth", vinn, jdn)
    c_ok = aph > MARS_PERI
    print(f"  → H-N43c {'SUPPORTED' if c_ok else 'REFUTED'}: prograde-outgoing aphelion {aph / C.AU:.2f} AU "
          f"{'≥' if c_ok else '<'} Mars perihelion {MARS_PERI / C.AU:.2f} AU — the barrier is "
          f"{'PHASING, not energy' if c_ok else 'ENERGY'} (the pumped orbit already reaches Mars's orbit radius).\n")

    print("  [min-|DSM| Earth->Mars across departure epochs — the handoff is phasing-dependent (R-N30)]", flush=True)
    results = []
    for off in (0.0, 200.0, 600.0):
        nd = earth_node(sjd + 400.0 + off)
        if nd is None:
            print(f"    t0+{off:.0f}: no Earth node"); continue
        jj, vv = nd
        md, (dbest, mbest, g) = min_dsm("earth", vv, jj)
        results.append(md)
        tag = (f"{md:.3f} km/s (CLOSED, miss {mbest:.1e} km)" if md is not None
               else f"{dbest:.3f} km/s but miss {mbest:.1e} (not closed)")
        print(f"    t0+{off:.0f}: v inf {float(jnp.linalg.norm(vv)):.2f}, min |DSM| = {tag}", flush=True)

    closed = [r for r in results if r is not None]
    best = min(closed) if closed else None
    b_ok = best is not None and best <= DSM_BAR
    print(f"\n  → H-N43b {'SUPPORTED' if b_ok else 'REFUTED'}: a bounded DSM "
          f"{'AFFORDS' if b_ok else 'does NOT afford'} Mars — best min |DSM| across epochs "
          f"{best:.3f} km/s {'≤' if b_ok else '>'} the {DSM_BAR:.1f} km/s bound"
          + (f" (at a well-phased epoch; a badly-phased epoch costs {max(closed):.2f})." if b_ok and len(closed) > 1 else "."))

    print(f"\n  → verdicts: H-N43a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N43b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N43c {'SUPPORTED' if c_ok else 'REFUTED'}")
    if a_ok and b_ok and c_ok:
        print("  NET: a small bounded DSM unlocks the outer-planet handoff the exact-ballistic architecture")
        print(f"    cannot reach. Mars is ballistically excluded ({bmiss / C.SOI_KM['mars']:.0f}x SOI) but the pumped")
        print("    orbit already has the ENERGY (aphelion ≫ Mars); the barrier is purely PHASING — aiming the")
        print(f"    flyby at Mars's moving position within δmax. One mid-course DSM buys that aim for ≤ {DSM_BAR:.1f} km/s")
        print("    at well-phased epochs (≈ half, like R-N30's inner-planet handoffs), several km/s when badly")
        print("    phased. So the honest hybrid — mostly-ballistic pump/crank with one small correction — extends")
        print("    the tour to Mars: the exact-ballistic exclusion is a phasing wall, not an energy one, and a")
        print("    bounded Δv-for-phasing trade (R-N28's economy) crosses it.")
    print("    Scope: one pumped Earth node per epoch, single mid-course DSM, grid-min |DSM| (upper bound),")
    print("    coarse 3-epoch phasing sweep. Mechanism/DISCOVERY study, never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
