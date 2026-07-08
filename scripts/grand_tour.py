#!/usr/bin/env python3
"""End-to-end grand-tour pipeline: Tisserand PROPOSE → flyby-node OPTIMIZE → primer CERTIFY (Build N, R-N11).

The three discovery layers wired into one pipeline on real ephemerides:
  1. PROPOSE  (R-N10 Tisserand): a Jupiter flyby reaches escape at the launch v∞.
  2. OPTIMIZE (R-N8 flyby-node): reach Jupiter's real position via a Lambert leg (the node "reach" — well-
     posed / razor-free, per R-N8), then apply the max bounded turn using Jupiter's real velocity.
  3. CERTIFY  (R-N9 primer): certify the Earth→Jupiter leg (no beneficial deep-space maneuver, or flag one).

A real single-Jupiter-flyby Earth-launch mission (New Horizons epochs). --verify uses an analytic circular
Jupiter (offline, CI-safe) to prove the WIRING; --fetch uses real Horizons states for the real demonstration.

    uv run --with jax python scripts/grand_tour.py --verify        # offline, CI-safe
    uv run --with jax python scripts/grand_tour.py --fetch         # real Horizons ephemerides (network)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM                 # noqa: E402
import nbody_sim as NB                 # noqa: E402
import tisserand_graph as TG           # noqa: E402  (PROPOSE)
import primer_vector as PV             # noqa: E402  (CERTIFY)

MU_S = TG.MU_S
AU = TG.AU
DAY = NB.DAY
MU_J = NB.GM["jupiter"]
R_JUP = 71492.0


def rodrigues(v, axis, ang):
    """Rotate vector v about a unit axis by angle ang (Rodrigues' formula)."""
    k = axis / (np.linalg.norm(axis) + 1e-30)
    return (v * np.cos(ang) + np.cross(k, v) * np.sin(ang) + k * (k @ v) * (1 - np.cos(ang)))


def propose(vinf_launch):
    """PROPOSE (Tisserand): does a Jupiter flyby reach escape at this launch v∞?"""
    a, e, seq = TG.pump_sequence(vinf_launch, ["earth", "jupiter"])
    escaped = (e >= 1.0 or a < 0)
    used_jup = any(g["planet"] == "jupiter" for g in seq)
    return escaped and used_jup, seq


def optimize_leg(r_e, r_j, v_j, tof1, rp_min):
    """OPTIMIZE (flyby-node): Lambert Earth→Jupiter (reach), then the max bounded turn at Jupiter's real
    state. Returns (v_dep, v_arr, reach_err_km, vinf, dmax, post-flyby a, e, v_helio_out)."""
    v_dep, v_arr = LAM.lambert(r_e, r_j, tof1, mu=MU_S)
    v_dep, v_arr = np.asarray(v_dep), np.asarray(v_arr)
    # verify the Lambert leg reaches Jupiter's position (Sun-only propagation) — the node "reach"
    n = 4000
    rv0 = jnp.asarray(np.concatenate([np.asarray(r_e), v_dep]))
    rvT, _ = NB.rollout(rv0, jnp.zeros((n, 1, 3)), jnp.array([MU_S]), tof1 / n, soft=0.0)
    reach_err = float(np.linalg.norm(np.asarray(rvT)[:3] - np.asarray(r_j)))
    vinf_in = v_arr - np.asarray(v_j)
    vinf = np.linalg.norm(vinf_in)
    dmax = 2.0 * np.arcsin(1.0 / (1.0 + rp_min * vinf ** 2 / MU_J))
    # turn v∞ toward Jupiter's velocity (prograde) to maximize post-flyby heliocentric energy
    vj = np.asarray(v_j)
    phi = np.arccos(np.clip(vinf_in @ vj / (vinf * np.linalg.norm(vj)), -1, 1))
    axis = np.cross(vinf_in, vj)
    vinf_out = rodrigues(vinf_in, axis, min(dmax, phi))
    v_out = vj + vinf_out
    r = float(np.linalg.norm(r_j))
    energy = 0.5 * (v_out @ v_out) - MU_S / r
    a = -MU_S / (2 * energy)
    h = np.linalg.norm(np.cross(np.asarray(r_j), v_out))
    # bound orbit: e=√(1−h²/(μa)); hyperbolic (energy>0, a<0): e=√(1+2·energy·h²/μ²) > 1
    e = np.sqrt(max(0.0, 1 - h ** 2 / (MU_S * a))) if a > 0 else np.sqrt(1 + 2 * energy * h ** 2 / MU_S ** 2)
    return v_dep, v_arr, reach_err, vinf, dmax, a, e, v_out


def certify_leg(r_e, r_j, v_e, v_j, tof1, n_steps):
    """CERTIFY (primer): |p|_max for the Earth→Jupiter leg as an orbit-to-orbit transfer."""
    dv2, pmag, pmax, f, cond, _ = PV.transfer_primer(np.asarray(r_e), np.asarray(r_j),
                                                     np.asarray(v_e), np.asarray(v_j), MU_S, tof1, n_steps)
    return pmax, f


def run(r_e, v_e, r_j, v_j, tof1, tag, vinf_launch, n_cert):
    rp_min = 1.5 * R_JUP
    print(f"--- {tag} ---")
    ok_prop, seq = propose(vinf_launch)
    print(f"  1. PROPOSE (Tisserand): launch v∞={vinf_launch:.1f} km/s → Jupiter flyby reaches "
          f"{'ESCAPE' if ok_prop else 'bounded'} ({len(seq)} flybys enumerated)")
    v_dep, v_arr, reach_err, vinf, dmax, a, e, v_out = optimize_leg(r_e, r_j, v_j, tof1, rp_min)
    dv_dep = np.linalg.norm(v_dep - np.asarray(v_e))
    esc = (e >= 1.0 or a < 0)
    print(f"  2. OPTIMIZE (flyby-node): Earth→Jupiter Lambert reaches Jupiter to {reach_err:.0f} km; "
          f"dep Δv={dv_dep:.3f} km/s; v∞={vinf:.2f} km/s, δmax={np.degrees(dmax):.1f}°")
    print(f"     post-flyby heliocentric orbit: a={a/AU:.2f} AU, e={e:.3f} → "
          f"{'SOLAR-SYSTEM ESCAPE' if esc else f'aphelion {a*(1+e)/AU:.2f} AU'}")
    # Tisserand consistency with the REAL Jupiter velocity
    vp = np.linalg.norm(np.asarray(v_j))
    T_from_vinf = 3.0 - (vinf / vp) ** 2
    print(f"     Tisserand check: v∞/v_J={vinf/vp:.3f} → T=3−(v∞/v_J)²={T_from_vinf:.4f} (v_J real={vp:.3f} km/s)")
    pmax, f = certify_leg(r_e, r_j, v_e, v_j, tof1, n_cert)
    verdict = "primer-OPTIMAL (no beneficial DSM)" if pmax <= 1.02 else f"primer flags a DSM at t/TOF={f:.2f}"
    print(f"  3. CERTIFY (primer): Earth→Jupiter leg |p|_max={pmax:.4f} → {verdict}")
    return ok_prop, reach_err, esc, pmax


def verify(args):
    print("=== R-N11: end-to-end grand-tour pipeline (Tisserand → flyby-node → primer) ===")
    # H-N11a: offline analytic circular Jupiter (CI-safe). Earth at 0°, Jupiter at 165°, TOF1=2.5 yr.
    r_j_au, _, _ = TG.PLAN["jupiter"]
    th = np.radians(165.0)
    r_e = np.array([AU, 0.0, 0.0])
    v_e = np.array([0.0, np.sqrt(MU_S / AU), 0.0])
    r_j = r_j_au * np.array([np.cos(th), np.sin(th), 0.0])
    v_j = np.sqrt(MU_S / r_j_au) * np.array([-np.sin(th), np.cos(th), 0.0])
    tof1 = 2.5 * 365.25 * DAY
    ok, reach, esc, pmax = run(r_e, v_e, r_j, v_j, tof1, "H-N11a offline (analytic circular Jupiter)",
                               9.0, args.steps)
    print(f"  → pipeline composes end-to-end offline: propose={ok}, reach={reach:.0f} km, "
          f"escape={esc}, leg |p|max={pmax:.3f}  ({'PASS' if ok and reach < 5e6 else 'CHECK'})")


def kepler_prop(r, v, tof, n=4000):
    """Propagate a state on its Sun-only Keplerian orbit by tof (Jupiter's orbit is Keplerian to ≪ SOI
    over a few years — the R-N3-quantified perturbation is tiny). Returns (r,v) at tof."""
    rv0 = jnp.asarray(np.concatenate([np.asarray(r), np.asarray(v)]))
    rvT, _ = NB.rollout(rv0, jnp.zeros((n, 1, 3)), jnp.array([MU_S]), tof / n, soft=0.0)
    rvT = np.asarray(rvT)
    return rvT[:3], rvT[3:]


def fetch(args):
    import nbody_transfer as NT
    print("=== R-N11 REAL ephemerides: Earth→Jupiter flyby (real Horizons states) ===")
    launch = "2005-08-12T00:00:00"                     # real Horizons state (cached offline)
    r_e, v_e = NT.body_state("earth", launch)          # real Earth departure state
    r_j0, v_j0 = NT.body_state("jupiter", launch)      # real Jupiter state at launch epoch
    r_e, v_e = np.asarray(r_e), np.asarray(v_e)
    # pick a well-phased TOF1 (min departure Δv) — a sensible transfer time, NOT a full launch-window/
    # resonance search (that is the flagged next layer); Jupiter is propagated on its real Keplerian orbit.
    best = None
    for tof1 in np.arange(2.0, 6.01, 0.25) * 365.25 * DAY:
        r_j, v_j = kepler_prop(r_j0, v_j0, tof1)
        vdep, _ = LAM.lambert(r_e, np.asarray(r_j), tof1, mu=MU_S)
        d = float(np.linalg.norm(np.asarray(vdep) - v_e))
        if best is None or d < best[0]:
            best = (d, tof1, np.asarray(r_j), np.asarray(v_j))
    _, tof1, r_j, v_j = best
    vpj = np.linalg.norm(v_j)
    print(f"  real Horizons: Earth+Jupiter states at {launch[:10]} (|v_Jupiter|={np.linalg.norm(np.asarray(v_j0)):.3f} "
          f"km/s — real, vs 13.06 circular); Jupiter propagated {tof1/DAY:.0f} d on its real orbit to the flyby "
          f"(min-Δv TOF among 2–6 yr; live Horizons at the flyby epoch unavailable — network down this turn).")
    ok, reach, esc, pmax = run(np.asarray(r_e), np.asarray(v_e), np.asarray(r_j), np.asarray(v_j),
                               tof1, "H-N11b REAL Horizons ephemerides", 9.0, args.steps)
    print(f"  → three layers compose on REAL data (real v_Jupiter={vpj:.3f} km/s at flyby): "
          f"propose={ok}, reach={reach:.0f} km, real Jupiter flyby escape={esc}, leg |p|max={pmax:.3f}")
    print("  (fixed real epoch — no launch-window/phasing search; that is the flagged next layer.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    if args.fetch:
        fetch(args)


if __name__ == "__main__":
    main()
