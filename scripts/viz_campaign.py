#!/usr/bin/env python3
"""Figures for the differentiable N-body / gravity-assist research arc (Build N), for the README and website.

Three figures, all regenerated from the live campaign code against the cached JPL ephemeris (offline):

  inclination_ceiling.png  the kinematic ceiling arcsin(v∞/v_P), the fixed-|v∞| crank climbing to it, and the
                           per-body node count that governs how fast you get there.
  ephemeris_tour.png       a real resonant leg flown by the differentiable N-body engine against the true
                           time-tagged JPL ephemeris (eccentric Earth, real perturbers), re-encountering Earth.
  leverage_cap.png         the v∞-leverage (a few-m/s apoapsis burn moves v∞ by ~100 m/s) and the SOI rate cap
                           that bounds it against the real Earth (R-N25 → R-N27).

    uv run --with jax --with astroquery --with astropy --with matplotlib python scripts/viz_campaign.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import jax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import matplotlib          # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import full_ephemeris_tour as F          # noqa: E402
import leverage_anatomy as L             # noqa: E402
import flyby_leverage as FL              # noqa: E402

MEDIA = Path("docs/media")
SUN = "#e8a33d"
EARTH = "#2b6cb0"
ORBIT = "#e0a458"
CRANK = "#2a9d8f"
LEV = "#d1495b"
INK = "#333333"
GRID = "#d8d8d8"
AU = F.AU
SOI_E = F.SOI_E
DAY = F.DAY
V_P = F.V_E                              # planet (Earth) orbital speed, 29.785 km/s


def _ceil_deg(v):
    """Kinematic inclination ceiling arcsin(v∞/v_P) in degrees, clipped for numerical safety."""
    return np.degrees(np.arcsin(np.clip(v / V_P, 0.0, 1.0)))


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- figure 1
def fig_inclination(sjd):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # panel A: the ceiling law + the crank ladder at v∞ = 8
    vv = np.linspace(0, V_P, 400)
    axA.plot(vv, _ceil_deg(vv), color=INK, lw=2.2,
             label=r"free ceiling  $\arcsin(v_\infty/v_P)$")
    alphas = np.linspace(0.0, np.pi / 2, 7)
    incs = []
    for a in alphas:
        inc, _ = L.crank_leg_real(8.0, a, sjd)
        incs.append(inc if inc is not None else np.nan)
    axA.plot(np.full(len(incs), 8.0), incs, "o-", color=CRANK, lw=1.6, ms=5, zorder=6,
             label="crank at $v_\\infty$=8 (same-body flybys)")
    ceil8 = _ceil_deg(8.0)
    axA.annotate("crank walks up to\nthe free ceiling", xy=(8.0, ceil8), xytext=(2.0, 33),
                 fontsize=8.5, color=CRANK,
                 arrowprops=dict(arrowstyle="->", color=CRANK, lw=1.1))
    # leverage pumps v∞ to the right (rate-capped 8->9.7, budget-limited beyond)
    axA.add_patch(FancyArrowPatch((8.0, ceil8), (9.7, _ceil_deg(9.7)),
                                  arrowstyle="-|>", mutation_scale=12, color=LEV, lw=1.8, zorder=7))
    axA.plot([9.7, 15.0], [_ceil_deg(9.7), _ceil_deg(15.0)], "--", color=LEV, lw=1.4, alpha=0.8)
    axA.annotate("leverage pumps $v_\\infty$\n(rate-capped ~9.7; budget beyond)", xy=(11, _ceil_deg(11)),
                 xytext=(13.5, 24), fontsize=8.5, color=LEV,
                 arrowprops=dict(arrowstyle="->", color=LEV, lw=1))
    axA.set_xlabel("hyperbolic excess speed $v_\\infty$ (km/s)")
    axA.set_ylabel("reachable inclination (deg)")
    axA.set_xlim(0, 30)
    axA.set_ylim(0, 92)
    axA.axhline(90, color=INK, lw=0.8, ls=":", alpha=0.5)
    axA.text(0.6, 86, "polar needs $v_\\infty \\geq v_P$", fontsize=8, color=INK, alpha=0.7)
    axA.legend(loc="lower right", fontsize=8.5, frameon=False)
    axA.set_title("Inclination is free up to a ceiling set by $v_\\infty$", fontsize=10.5)
    _grid(axA)

    # panel B: how many same-body flybys to reach the ceiling (governed by turn per flyby)
    bodies = ["Jupiter", "Mars", "Venus"]
    nodes = [1, 2, 5]
    dmax = [59, 17, 10]                  # representative single-flyby turn (deg), inner->outer
    cols = [ORBIT, LEV, CRANK]
    bars = axB.bar(bodies, nodes, color=cols, alpha=0.85, width=0.6)
    for b, n, d in zip(bars, nodes, dmax):
        axB.text(b.get_x() + b.get_width() / 2, n + 0.12, f"{n} flyby" + ("s" if n > 1 else ""),
                 ha="center", fontsize=9, color=INK)
        axB.text(b.get_x() + b.get_width() / 2, 0.18, f"turn ≈{d}°", ha="center", fontsize=8.2,
                 color="white", weight="bold")
    axB.set_ylabel("same-body flybys to reach the ceiling")
    axB.set_ylim(0, 6)
    axB.set_title("A bigger turn per flyby means fewer flybys", fontsize=10.5)
    axB.text(0.5, -0.22, "Solar Orbiter cranked to ~33° with ~7–8 Venus assists; the diff-sim tour reached 32.8°"
             " (ceiling 32.9°).", transform=axB.transAxes, ha="center", fontsize=8, color=INK, alpha=0.75)
    _grid(axB)
    axB.xaxis.grid(False)

    fig.tight_layout()
    fig.savefig(MEDIA / "inclination_ceiling.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote inclination_ceiling.png")


# ---------------------------------------------------------------- figure 2
def fig_ephemeris_tour(sjd):
    # a 1:2 resonant leg launched from where real Earth is, flown under Sun + real perturbers
    rv0, tof = L.launch_exact(8.0, 1, 2, sjd)
    n = 8000
    _, traj = F.propagate_ephem(rv0, sjd, tof, n)
    jj = sjd + (np.arange(n) * (tof / n)) / DAY
    eph_e = F._load("earth", False)
    r_e = F._sample_r(eph_e, jj)
    d = np.linalg.norm(traj[:, :3] - r_e, axis=1)
    kap = int(np.argmax(np.linalg.norm(traj[:, :3], axis=1)))            # apoapsis
    h = int(0.4 * n)
    kre = h + int(np.argmin(d[h:]))                                      # re-encounter

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    # Earth's real orbit over the span
    ax.plot(r_e[:, 0] / AU, r_e[:, 1] / AU, color=EARTH, lw=1.2, alpha=0.55, label="Earth (real ephemeris)")
    # the flown resonant leg
    ax.plot(traj[:, 0] / AU, traj[:, 1] / AU, color=INK, lw=1.5, label="spacecraft (1:2 resonant leg)")
    ax.plot(0, 0, "o", color=SUN, ms=13, zorder=5)
    ax.text(0.06, -0.16, "Sun", color="#b9791f", fontsize=9)
    # launch, apoapsis, re-encounter
    ax.plot(traj[0, 0] / AU, traj[0, 1] / AU, "o", color=EARTH, ms=7, zorder=6)
    ax.annotate("launch from real Earth", (traj[0, 0] / AU, traj[0, 1] / AU),
                xytext=(-1.9, 0.3), fontsize=8.5, color=EARTH,
                arrowprops=dict(arrowstyle="->", color=EARTH, lw=1))
    ax.plot(traj[kap, 0] / AU, traj[kap, 1] / AU, "s", color=LEV, ms=6, zorder=6)
    ax.annotate("apoapsis\n(leverage burn)", (traj[kap, 0] / AU, traj[kap, 1] / AU),
                xytext=(traj[kap, 0] / AU - 0.15, traj[kap, 1] / AU + 0.02), fontsize=8.5, color=LEV, ha="right",
                arrowprops=dict(arrowstyle="->", color=LEV, lw=1))
    ax.plot(traj[kre, 0] / AU, traj[kre, 1] / AU, "*", color=CRANK, ms=13, zorder=7)
    ax.annotate(f"re-encounter\n(miss {d[kre]/SOI_E:.2f}·SOI)", (traj[kre, 0] / AU, traj[kre, 1] / AU),
                xytext=(0.35, 0.45), fontsize=8.5, color=CRANK,
                arrowprops=dict(arrowstyle="->", color=CRANK, lw=1))
    ax.set_aspect("equal")
    ax.set_xlabel("x (AU, J2000 ecliptic)")
    ax.set_ylabel("y (AU)")
    ax.set_ylim(-2.35, 1.45)
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    ax.set_title("A resonant leg flown against the real solar system", fontsize=10.5, pad=10)
    _grid(ax)
    fig.tight_layout()
    fig.savefig(MEDIA / "ephemeris_tour.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote ephemeris_tour.png")


# ---------------------------------------------------------------- figure 3
def fig_leverage_cap(sjd):
    burns = np.array([0.0, 1, 2, 5, 10, 20, 40, 80]) / 1000.0            # km/s
    dvinf, misses = [], []
    base = None
    for b in burns:
        miss, vn = FL.dsm_leg(8.0, 1, 2, sjd, b)
        if base is None:
            base = vn
        dvinf.append((vn - base) * 1000.0)                              # m/s pumped vs the no-burn leg
        misses.append(miss / SOI_E)
    dvinf = np.array(dvinf)
    misses = np.array(misses)
    burns_ms = burns * 1000.0

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # panel A: leverage — a few m/s of burn moves v∞ by ~100 m/s
    axA.plot(burns_ms, dvinf, "o-", color=LEV, lw=1.8, ms=5)
    # slope (leverage) of the first small-burn segment
    lev = dvinf[3] / burns_ms[3] if burns_ms[3] > 0 else float("nan")
    axA.plot([0, 12], [0, 12 * lev], "--", color=INK, lw=1, alpha=0.6)
    axA.annotate(f"slope ≈ leverage L ≈ {lev:.0f}\n(a 5 m/s burn → {dvinf[3]:.0f} m/s of $v_\\infty$)",
                 xy=(5, dvinf[3]), xytext=(9, 60), fontsize=8.8, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    axA.set_xlabel("apoapsis burn (m/s)")
    axA.set_ylabel("$v_\\infty$ gained at re-encounter (m/s)")
    axA.set_title("V∞-leverage: a small burn moves $v_\\infty$ a lot", fontsize=10.5)
    _grid(axA)

    # panel B: the SOI rate cap — pumping harder throws the re-encounter off Earth
    within = misses < 1.0
    axB.plot(dvinf, misses, "-", color=INK, lw=1.4, zorder=2)
    axB.scatter(dvinf[within], misses[within], color=CRANK, s=38, zorder=3, label="within Earth's SOI")
    axB.scatter(dvinf[~within], misses[~within], color=LEV, s=38, zorder=3, label="misses Earth (> SOI)")
    axB.axhline(1.0, color=INK, ls=":", lw=1)
    axB.text(20, 1.06, "Earth SOI (re-encounter must land inside)", fontsize=8, color=INK, alpha=0.75)
    if within.any():
        cap = float(dvinf[within].max())
        axB.axvline(cap, color=CRANK, ls="--", lw=1.2, alpha=0.8)
        axB.annotate(f"per-leg cap ≈ {cap:.0f} m/s", xy=(cap, 0.4), xytext=(cap - 250, 0.5),
                     fontsize=8.8, color=CRANK, arrowprops=dict(arrowstyle="->", color=CRANK, lw=1))
    axB.set_xlabel("$v_\\infty$ pumped this leg (m/s)")
    axB.set_ylabel("re-encounter miss (× Earth SOI)")
    axB.set_title("...but the SOI budget rate-caps the pump", fontsize=10.5)
    axB.legend(loc="upper left", fontsize=8.5, frameon=False)
    _grid(axB)

    fig.tight_layout()
    fig.savefig(MEDIA / "leverage_cap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote leverage_cap.png")


def main():
    if not F._require_cache():
        print("no ephemeris cache — run: uv run --with jax --with astroquery --with astropy "
              "python scripts/full_ephemeris_tour.py --fetch")
        return
    sjd = F._start_jd()
    fig_inclination(sjd)
    fig_ephemeris_tour(sjd)
    fig_leverage_cap(sjd)


if __name__ == "__main__":
    main()
