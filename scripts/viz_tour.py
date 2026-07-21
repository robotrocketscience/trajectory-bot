#!/usr/bin/env python3
"""Figure for the realized pump+crank tour and its epoch robustness (R-N36 → R-N39), for the README and website.

  pump_crank_tour.png   three panels: (a) the discovered pump chain — planet-relative v∞ per flyby (R-N37,
                        `beam_constrained_tour.py --verify`); (b) the inclination crank walk at the saturated
                        node vs the analytic ceiling arcsin(v∞/v_P) (R-N38, `crank_walk.py --verify`); (c) the
                        launch-epoch sweep — final v∞ per epoch with the failure modes marked (R-N39,
                        `epoch_robustness.py --verify`).

The plotted series are the RECORDED outputs of those three verify scripts (each ~8 min against the cached JPL
ephemeris); re-run them to regenerate the numbers, then this script to regenerate the figure.

    uv run --with matplotlib python scripts/viz_tour.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

MEDIA = Path("docs/media")
PUMP = "#d1495b"
CRANK = "#2a9d8f"
EPOCH = "#2b6cb0"
FAIL = "#b9b9b9"
INK = "#333333"
GRID = "#d8d8d8"

# ---- recorded series (sources cited; regenerate via the named verify scripts) -------------------------------
# R-N37 beam_constrained_tour.py --verify: launch seed 5.95 (Earth-relative), then per-flyby arrival v∞
# (planet-relative, the flyby invariant): V->E 8.11, E->V 8.35, V->E 11.24, E->V 16.27, then two v∞-neutral
# resonant returns at 16.27. Every encounter GN-closed sub-SOI, zero DSM.
TOUR_VINF = [5.95, 8.11, 8.35, 11.24, 16.27, 16.27, 16.27]
# R-N38 crank_walk.py --verify: i_rel per crank at the saturated node (v∞ 16.27, δmax 18.6°), ceiling 27.9°.
CRANK_I = [1.16, 9.53, 17.01, 21.88, 25.72, 27.05, 27.05, 27.09, 27.09]
CRANK_CEIL = 27.9
# R-N39 epoch_robustness.py --verify: final v∞ per launch epoch offset (d); None = no viable ≤8 km/s launch.
EPOCH_OFF = [100, 250, 400, 550, 700, 850, 1000, 1150]
EPOCH_VF = [15.12, None, 17.47, 17.79, None, 15.51, 4.21, 16.67]
EPOCH_MEDIAN = 15.31


def _grid(ax):
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) the pump chain
    x = np.arange(len(TOUR_VINF))
    a.plot(x, TOUR_VINF, "o-", color=PUMP, lw=2, ms=6, zorder=3)
    a.annotate("+5.0 km/s in one\nEarth→Venus handoff", xy=(4, 16.27), xytext=(1.4, 14.6), color=INK,
               fontsize=9, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    a.annotate("resonant returns\n(v∞ conserved)", xy=(5.5, 16.27), xytext=(4.0, 11.5), color=INK,
               fontsize=9, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    a.set_xticks(x)
    a.set_xticklabels(["launch", "V→E", "E→V", "V→E", "E→V", "V→V", "V→V"], fontsize=8)
    a.set_ylabel("planet-relative v∞ (km/s)")
    a.set_title("the discovered pump: v∞ 5.95 → 16.27,\nsix flybys, zero Δv", fontsize=10)
    _grid(a)

    # (b) the crank walk vs the analytic ceiling
    xb = np.arange(len(CRANK_I))
    b.plot(xb, CRANK_I, "o-", color=CRANK, lw=2, ms=6, zorder=3)
    b.axhline(CRANK_CEIL, color=INK, lw=1.0, ls="--")
    b.text(0.1, CRANK_CEIL + 0.5, "analytic ceiling arcsin(v∞/v_P) = 27.9°", fontsize=9, color=INK)
    b.annotate("re-closure tax: each turn\nspends ~half its angle on\nre-hitting the real planet",
               xy=(2, 17.0), xytext=(3.2, 6.0), color=INK, fontsize=9,
               arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    b.set_xlabel("crank flyby (Venus resonant returns)")
    b.set_ylabel("inclination vs Venus's orbit (deg)")
    b.set_title("the free inclination crank at v∞ 16.27:\n1.2° → 27.1° = 97% of the ceiling", fontsize=10)
    b.set_ylim(0, 31)
    _grid(b)

    # (c) the epoch sweep — placeholder bars for no-launch epochs span the data range (no magic height)
    xc = np.arange(len(EPOCH_OFF))
    v_top = 1.05 * max(v for v in EPOCH_VF if v is not None)
    for i, (off, vf) in enumerate(zip(EPOCH_OFF, EPOCH_VF)):
        if vf is None:
            c.bar(i, v_top, color="none", edgecolor=FAIL, hatch="//", lw=1.0)
            c.text(i, 0.7, "no\nlaunch", ha="center", fontsize=7.5, color=INK)
        else:
            c.bar(i, vf, color=EPOCH if vf >= 10 else FAIL)
    c.axhline(EPOCH_MEDIAN, color=INK, lw=1.0, ls="--")
    c.text(0.05, EPOCH_MEDIAN + 0.4, f"median {EPOCH_MEDIAN}", fontsize=9, color=INK)
    c.text(6, 5.2, "pump\nfailure", ha="center", fontsize=7.5, color=INK)
    c.set_xticks(xc)
    c.set_xticklabels([f"+{o}" for o in EPOCH_OFF], fontsize=8)
    c.set_xlabel("launch epoch (days from window start)")
    c.set_ylabel("final v∞ (km/s)")
    c.set_title("the tour is typical, not lucky:\n5 of 8 epochs pump to 15–18 km/s", fontsize=10)
    _grid(c)

    fig.tight_layout()
    fig.savefig(MEDIA / "pump_crank_tour.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote pump_crank_tour.png")


if __name__ == "__main__":
    main()
