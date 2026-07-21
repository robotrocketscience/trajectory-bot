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
    control_law.png     what the policy LEARNED, not its raw weights: the
                        commanded thrust read off the policy around one orbit
                        (coast, then a prograde burn at apoapsis) beside an
                        input-attribution bar chart (radius error dominates the
                        throttle) — the discovered textbook circularization burn
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
from jax import random, vmap, jacrev

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


# the 13 observation inputs, grouped into the 9 physical quantities they encode
_OBS_GROUPS = [(0, 3, "position"), (3, 6, "velocity"), (6, 7, "sma error"),
               (7, 8, "eccentricity"), (8, 9, "radius error"), (9, 10, "point·prograde"),
               (10, 11, "point·normal"), (11, 12, "point·radial"), (12, 13, "fuel")]


def _q_align_x_to(vhat):
    """Quaternion rotating body-x [1,0,0] onto unit vector vhat (points the craft prograde)."""
    x = np.array([1.0, 0, 0]); v = vhat / np.linalg.norm(vhat); c = float(np.dot(x, v))
    if c > 0.9999: return np.array([1.0, 0, 0, 0])
    if c < -0.9999: return np.array([0.0, 0, 0, 1.0])
    ax = np.cross(x, v); ax /= np.linalg.norm(ax); half = np.arccos(c) / 2.0
    return np.array([np.cos(half), *(np.sin(half) * ax)])


def _state_at(nu, rp, ra):
    """In-plane state at true anomaly nu on the (rp, ra) orbit, attitude pointed prograde, full fuel."""
    a = 0.5 * (rp + ra); e = (ra - rp) / (ra + rp)
    p = a * (1 - e ** 2); h = np.sqrt(J.MU * p); r = p / (1 + e * np.cos(nu))
    pos = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    vel = np.array([(J.MU / h) * (-np.sin(nu)), (J.MU / h) * (e + np.cos(nu)), 0.0])
    return np.concatenate([pos, vel, _q_align_x_to(vel), np.zeros(3)])


def fig_control_law():
    """Replaces the old raw-weight heatmaps. Left: the learned control law read off the policy around
    one representative orbit (coast, then a prograde burn at apoapsis). Right: input attribution — which
    of the 13 physical inputs actually drives the throttle (mean |d throttle / d input| over flown
    states). Together: the policy rediscovered the textbook apoapsis-circularization burn, keyed on the
    radius error, from a raw fuel objective — which three weight matrices could never show."""
    mlp = load_mlp(FINAL_CKPT)

    # (left) sample the policy's commanded thrust around one clearly-eccentric orbit
    rp = J.R_BODY + 600.0; ra = rp * 3.0                 # e = 0.5
    nus = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    S = np.array([_state_at(nu, rp, ra) for nu in nus])
    obs = J.observe(jnp.asarray(S), jnp.full(len(nus), ra), jnp.full(len(nus), J.DV_BUDGET))
    act = np.asarray(J.policy(mlp, obs))
    thr = np.clip(act[:, 3], 0.0, 1.0)
    burn_m = thr > 0.15
    pos = S[:, 0:2]

    # (right) input attribution over on-distribution flown states
    st, rr = J.sample_orbits(random.PRNGKey(3), 400)
    ob = J.observe(st, rr, jnp.full(400, J.DV_BUDGET))
    jac = np.abs(np.asarray(vmap(jacrev(lambda o: J.policy(mlp, o[None])[0]))(ob)))   # (400, 4, 13)
    thr_sens = jac[:, 3, :].mean(0)
    grp = np.array([thr_sens[a:b].sum() for a, b, _ in _OBS_GROUPS])
    grp = grp / grp.max()
    labels = [g[2] for g in _OBS_GROUPS]

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                  gridspec_kw={"width_ratios": [1.1, 1.0]})

    # left: orbit with coast (thin grey) / burn (thick red) + prograde thrust arrows
    segs = np.stack([pos[:-1], pos[1:]], axis=1)
    cols = [BURN if burn_m[i + 1] else COAST for i in range(len(pos) - 1)]
    lws = [4.0 if burn_m[i + 1] else 1.6 for i in range(len(pos) - 1)]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, zorder=3))
    ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=EARTH, zorder=4))
    im = len(nus) // 4
    vdir = S[im, 3:5] / np.linalg.norm(S[im, 3:5])
    ax.annotate("", xy=pos[im] + vdir * ra * 0.16, xytext=pos[im],
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.4))
    ax.text(pos[im, 0], pos[im, 1] + ra * 0.19, "motion", color="#888", fontsize=8, ha="center")
    for i in range(0, len(nus), 5):
        if burn_m[i]:
            v = S[i, 3:5] / np.linalg.norm(S[i, 3:5])
            ax.annotate("", xy=pos[i] + v * ra * 0.16, xytext=pos[i],
                        arrowprops=dict(arrowstyle="-|>", color=BURN, lw=1.7))
    ap = pos[np.argmax(np.linalg.norm(pos, axis=1))]
    pe = pos[np.argmin(np.linalg.norm(pos, axis=1))]
    ax.scatter(*ap, s=18, color=BURN, zorder=6); ax.scatter(*pe, s=18, color="#555", zorder=6)
    ax.annotate("apoapsis: burn\nprograde", xy=ap, xytext=(ap[0] - ra * 0.15, ap[1] + ra * 0.42),
                fontsize=9.5, color=BURN, ha="center", fontweight="bold")
    ax.annotate("periapsis:\ncoast", xy=pe, xytext=(pe[0] + ra * 0.30, pe[1] - ra * 0.34),
                fontsize=9.5, color="#555", ha="center")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.margins(0.16)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("what it does: coast, then burn prograde at apoapsis", fontsize=10)

    # right: input attribution bars, radius error highlighted
    order = np.argsort(grp)
    bar_c = [FINAL if labels[i] == "radius error" else "#9db4c8" for i in order]
    axb.barh(np.arange(len(grp)), grp[order], color=bar_c)
    axb.set_yticks(np.arange(len(grp))); axb.set_yticklabels([labels[i] for i in order], fontsize=9)
    axb.set_xlabel("relative influence on the throttle decision", fontsize=9)
    axb.set_title("what it keys on: the radius error", fontsize=10)
    for sp in ("top", "right"):
        axb.spines[sp].set_visible(False)

    fig.suptitle("the trained circularizer's control law — discovered from a raw Δv objective, not hand-coded",
                 fontsize=11)
    fig.text(0.5, -0.02, "13→128→128→4 tanh MLP (warm_r28). Left: policy output sampled around an e=0.5 orbit. "
             "Right: mean |∂throttle/∂input| over flown states, inputs grouped into physical quantities.",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(MEDIA / "control_law.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("control_law.png done", flush=True)


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
    fig_control_law()
    fig_generalization()
    fig_learning_gif()
