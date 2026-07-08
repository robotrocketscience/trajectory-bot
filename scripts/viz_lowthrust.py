#!/usr/bin/env python3
"""Render the low-thrust circularization spiral (docs/media/low-thrust-spiral.gif).

Flies an E1 low-thrust specialist through the same substep tracer
scripts/viz_readme.py uses (same rk4, same pointing controller, same fuel
gate) and animates the episode: burn arcs smeared across multiple
revolutions, spiraling out from the start ellipse to the target circle.
For pictures, not claims — E1 numbers come from eval_probe/verify_probe.

Usage (repo root; checkpoint lives on the training box):
    uv run --with jax --with matplotlib --with pillow \
        python scripts/viz_lowthrust.py [models/lt_2e-4_ema_final.npz]
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

import jaxsim as J
import viz_readme as V  # __main__-guarded; import only sets the sim knobs

A_THRUST_LOW = 2e-4  # the lt_2e-4 specialist's engine (E1, 25x below chemical)
DECISIONS = 900      # low thrust needs a long horizon
FRAMES = 180
OUT = Path("docs/media/low-thrust-spiral.gif")


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/lt_2e-4_ema_final.npz"
    J.A_THRUST = A_THRUST_LOW
    mlp = V.load_mlp(ckpt)

    # Deterministic photogenic pick: most-eccentric start the policy solves.
    states, rts = J.sample_orbits(random.PRNGKey(777_777), 48)
    _, e0s = J.elements(states)
    pick = None
    for i in np.argsort(-np.asarray(e0s)):
        pos, burn, latch = V.trace_episode(mlp, states[i], rts[i], decisions=DECISIONS)
        if latch is not None:
            pick = (states[i], float(rts[i]), pos, burn, latch, float(e0s[i]))
            break
    if pick is None:
        raise SystemExit("no solved low-thrust episode in the sample")
    s0, rt, pos, burn, latch, e0 = pick

    ang = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    revs = abs(ang[-1] - ang[0]) / (2 * np.pi)
    print(f"episode: e0={e0:.2f}, {latch} substeps = {latch * J.DT / 3600:.1f} h, "
          f"~{revs:.1f} revolutions, burn fraction {burn.mean():.0%}", flush=True)

    ell = V.orbit_ellipse(s0)
    lim = max(np.abs(pos[:, :2]).max(), rt) * 1.12
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=72)
    idx = np.linspace(1, len(pos) - 1, FRAMES).astype(int)

    def draw(f):
        ax.clear()
        i = idx[f]
        ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=V.EARTH))
        ax.plot(ell[:, 0], ell[:, 1], "--", color="#bbb", lw=0.8)
        th = np.linspace(0, 2 * np.pi, 150)
        ax.plot(rt * np.cos(th), rt * np.sin(th), ":", color=V.TARGET, lw=1.0)
        segs = np.stack([pos[: i - 1, :2], pos[1:i, :2]], axis=1)
        cols = [V.BURN if b else V.COAST for b in burn[1:i]]
        ax.add_collection(LineCollection(segs, colors=cols, lw=1.1))
        ax.scatter(*pos[i, :2], color="#111", s=22, zorder=6)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    anim = FuncAnimation(fig, draw, frames=FRAMES)
    anim.save(OUT, writer=PillowWriter(fps=24))
    plt.close(fig)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
