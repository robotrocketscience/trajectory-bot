#!/usr/bin/env python3
"""Extra README figures — the "how it works / that it works" RL showcase.

Usage (needs the checkpoint progression in models/):
    uv run --with jax --with matplotlib --with pillow \
        python scripts/viz_readme_extra.py

Figures written to docs/media/:
    learning.gif        the SAME start orbit flown by four training stages
                        (imitation -> refined), so you watch the path tighten
                        onto the target as the policy learns
    progression.png     fresh-4096 success climbing across the R&D campaign
    weights.png         the trained policy's three weight matrices (heatmaps)
    generalization.png  40 random starts flown by the final policy, colored
                        solved / failed — it works across the distribution

Reuses scripts/viz_readme.py's episode tracer (substep-for-substep mirror of
jaxsim._decision_step); these are pictures, not claims. Success numbers are the
verified fresh-4096 figures (eval_probe.py), inlined with provenance.
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np
import jax.numpy as jnp
from jax import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

import jaxsim as J
from viz_readme import load_mlp, trace_episode, orbit_ellipse, pick_episode, \
    EARTH, COAST, BURN, FINAL, TARGET

J.DV_BUDGET = 2.0
J.ABSORB = True
J.PHI_DV = True
J.D_EPS = 1e-4

MEDIA = Path("docs/media")
MEDIA.mkdir(parents=True, exist_ok=True)

# (checkpoint, label, verified fresh-4096 success %) — the campaign climb.
# Numbers from eval_probe.py (2026-07); dagger is the imitation oracle floor.
STAGES = [
    ("models/dagger_jax.npz",         "imitation (DAgger)",  79.9),
    ("models/warm_r19.npz",           "diff-sim refine · early", 90.6),
    ("models/warm_r25.npz",           "diff-sim refine · mid",   92.1),
    ("models/warm_r28_ema_final.npz", "diff-sim refine · final", 92.3),
]
PROGRESSION = [
    ("imitation\n(DAgger)", 79.9), ("warm_r19", 90.6), ("warm_r25", 92.1),
    ("warm_r26", 91.9), ("warm_r28\n(final)", 92.3),
]
FINAL_CKPT = "models/warm_r28_ema_final.npz"


def _draw_flight(ax, pos, burn, s0, rt, lim):
    ell = orbit_ellipse(s0)
    ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=EARTH, zorder=3))
    ax.plot(ell[:, 0], ell[:, 1], "--", color="#bbb", lw=0.9, zorder=2)
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rt * np.cos(th), rt * np.sin(th), ":", color=TARGET, lw=1.1, zorder=2)
    segs = np.stack([pos[:-1, :2], pos[1:, :2]], axis=1)
    cols = [BURN if b else COAST for b in burn[1:]]
    ax.add_collection(LineCollection(segs, colors=cols, lw=1.6, zorder=4))
    ax.scatter(*pos[-1, :2], color=FINAL, s=26, zorder=6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig_learning_gif():
    """One fixed start; cycle the four training stages so the path visibly
    tightens onto the target ring as the policy learns."""
    final = load_mlp(FINAL_CKPT)
    s0, rt, _, _, _, e0 = pick_episode(final)     # a start the final policy solves
    rt_a = jnp.asarray(rt)                         # trace_episode indexes rt0[None]
    flights = []
    lim = J.R_BODY
    for path, label, succ in STAGES:
        pos, burn, latch = trace_episode(load_mlp(path), s0, rt_a)
        flights.append((pos, burn, label, succ, latch))
        lim = max(lim, np.abs(pos[:, :2]).max(), rt)
    lim *= 1.12

    hold = 26                                     # frames per stage (~1.1s @ 24fps)
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=72)

    def draw(f):
        pos, burn, label, succ, latch = flights[(f // hold) % len(flights)]
        ax.clear()
        _draw_flight(ax, pos, burn, s0, rt, lim)
        tag = "solved" if latch is not None else "misses target"
        ax.set_title(f"{label}\nfresh success {succ:.1f}%   ({tag})", fontsize=10)

    anim = FuncAnimation(fig, draw, frames=hold * len(flights))
    anim.save(MEDIA / "learning.gif", writer=PillowWriter(fps=24))
    plt.close(fig)
    print("learning.gif done", flush=True)


def fig_progression():
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    labels = [x[0] for x in PROGRESSION]; succ = [x[1] for x in PROGRESSION]
    x = np.arange(len(labels))
    ax.plot(x, succ, "-o", color="#2b6cb0", lw=2.0, ms=7, zorder=3)
    ax.fill_between(x, 0, succ, color="#2b6cb0", alpha=0.07)
    ax.axhline(79.9, color="#999", ls="--", lw=1.0)
    ax.text(len(labels) - 1, 81.4, "imitation oracle floor", fontsize=8,
            color="#777", ha="right")
    for xi, v in zip(x, succ):
        ax.text(xi, v + 1.1, f"{v:.1f}", ha="center", fontsize=8.5, color="#2b6cb0")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(70, 100); ax.set_ylabel("success, fresh 4096-episode set (%)")
    ax.set_title("differentiable-sim refinement climbs past the imitation oracle",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MEDIA / "progression.png", dpi=110)
    plt.close(fig)
    print("progression.png done", flush=True)


def fig_weights():
    d = np.load(FINAL_CKPT)
    mats = [(d["w0"], "w0  obs->h1  (13x128)"),
            (d["w1"], "w1  h1->h2  (128x128)"),
            (d["w2"], "w2  h2->action  (128x4)")]
    vmax = max(np.abs(m).max() for m, _ in mats)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [13, 128, 4]})
    for ax, (m, title) in zip(axes, mats):
        im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="weight")
    fig.suptitle("trained policy weights (warm_r28) — 13→128→128→4 tanh MLP",
                 fontsize=10)
    fig.savefig(MEDIA / "weights.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("weights.png done", flush=True)


def fig_generalization(n=40):
    mlp = load_mlp(FINAL_CKPT)
    states, rts = J.sample_orbits(random.PRNGKey(2_024), n)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    lim = J.R_BODY; solved = 0
    paths = []
    for i in range(n):
        pos, burn, latch = trace_episode(mlp, states[i], rts[i])
        ok = latch is not None; solved += ok
        paths.append((pos, ok)); lim = max(lim, np.abs(pos[:, :2]).max())
    lim *= 1.1
    ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=EARTH, zorder=3))
    for pos, ok in paths:
        ax.plot(pos[:, 0], pos[:, 1], "-", lw=0.8, alpha=0.7,
                color=(FINAL if ok else "#d1495b"), zorder=4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color=FINAL, lw=2),
               Line2D([0], [0], color="#d1495b", lw=2)],
              [f"solved ({solved}/{n})", f"missed ({n - solved}/{n})"],
              loc="upper right", fontsize=8, frameon=False)
    ax.set_title(f"final policy on {n} random start orbits", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "generalization.png", dpi=110)
    plt.close(fig)
    print(f"generalization.png done ({solved}/{n} solved)", flush=True)


if __name__ == "__main__":
    fig_progression()
    fig_weights()
    fig_generalization()
    fig_learning_gif()
