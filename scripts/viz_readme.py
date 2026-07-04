#!/usr/bin/env python3
"""Regenerate the README figures in docs/media/.

Usage:
    uv run --with jax python scripts/viz_readme.py <policy.npz> [logdir]

Figures:
    trajectory.png     one episode flown by the policy — top-down path colored
                       by thrust, plus a/rt and e vs time with tolerance bands
    maneuver.gif       the same episode animated (satellite, trail, burn arrows)
    training-curve.png the optimizer-forensics story in one plot (four runs)
    results.png        verified scoreboard — fresh-set success + float64 fuel

The trajectory tracer mirrors scripts/jaxsim.py `_decision_step` substep-for-
substep (same rk4, same pointing controller, same fuel gate) but records
position/thrust each substep; it is for pictures, not claims. Training-curve
input is the raw run logs (iter/success lines); results.png inlines the
verified numbers with provenance comments.
"""
import re
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

J.DV_BUDGET = 2.0
J.ABSORB = True
J.PHI_DV = True
J.D_EPS = 1e-4

MEDIA = Path("docs/media")
MEDIA.mkdir(parents=True, exist_ok=True)

EARTH = "#4a7ab5"
COAST = "#7f9cbf"
BURN = "#e4572e"
FINAL = "#2e9e6b"
TARGET = "#2e9e6b"


def load_mlp(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def trace_episode(mlp, s0, rt0, decisions=120):
    """Mirror of jaxsim._decision_step recording per-substep position/thrust."""
    state = s0[None]
    rt = rt0[None]
    fuel = jnp.full((1,), J.DV_BUDGET)
    pos, burning = [], []
    latch_at = None
    for k in range(decisions):
        obs = J.observe(state, rt, jnp.clip(fuel, 0.0, None))
        act = J.policy(mlp, obs)
        coeffs = act[:, 0:3]
        throttle = jnp.clip(act[:, 3], 0.0, 1.0)
        for _ in range(J.REPEAT):
            t, w, s_hat = J.orbit_frame(state[:, 0:3], state[:, 3:6])
            d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s_hat
            d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
            omega_cmd = J.point_rate(state[:, 6:10], d)
            gate = (fuel > 0).astype(jnp.float32)
            thr = throttle * gate
            fuel = fuel - thr * J.A_THRUST * J.DT
            state = J.rk4(state, omega_cmd, thr)
            pos.append(np.asarray(state[0, 0:3]))
            burning.append(float(thr[0]) > 0.02)
        ae, e = J.a_err_e(state, rt)
        if latch_at is None and bool((ae[0] < J.A_TOL) & (e[0] < J.E_TOL)):
            latch_at = (k + 1) * J.REPEAT
            break
    return np.array(pos), np.array(burning), latch_at


def pick_episode(mlp, n=64):
    """Deterministic photogenic pick: most-eccentric start the policy solves."""
    states, rts = J.sample_orbits(random.PRNGKey(777_777), n)
    a0, e0 = J.elements(states)
    order = np.argsort(-np.asarray(e0))
    for i in order:
        pos, burn, latch = trace_episode(mlp, states[i], rts[i])
        if latch is not None:
            return states[i], float(rts[i]), pos, burn, latch, float(e0[i])
    raise SystemExit("no solved episode found in the sample")


def orbit_ellipse(s0, npts=400):
    """Analytic initial orbit from the starting state, for the dashed outline."""
    r0 = np.asarray(s0[0:3]); v0 = np.asarray(s0[3:6])
    h = np.cross(r0, v0)
    evec = np.cross(v0, h) / J.MU - r0 / np.linalg.norm(r0)
    e = np.linalg.norm(evec)
    a = 1.0 / (2.0 / np.linalg.norm(r0) - np.dot(v0, v0) / J.MU)
    p_hat = evec / max(e, 1e-9)
    q_hat = np.cross(h / np.linalg.norm(h), p_hat)
    nu = np.linspace(0, 2 * np.pi, npts)
    r = a * (1 - e ** 2) / (1 + e * np.cos(nu))
    return r[:, None] * (np.cos(nu)[:, None] * p_hat + np.sin(nu)[:, None] * q_hat)


def fig_trajectory(pos, burn, s0, rt, e0):
    ell = orbit_ellipse(s0)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=EARTH, zorder=3))
    ax.plot(ell[:, 0], ell[:, 1], "--", color="#999", lw=1.0,
            label=f"initial orbit (e={e0:.2f})")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(rt * np.cos(th), rt * np.sin(th), ":", color=TARGET, lw=1.2,
            label="target radius")
    segs = np.stack([pos[:-1, :2], pos[1:, :2]], axis=1)
    colors = [BURN if b else COAST for b in burn[1:]]
    ax.add_collection(LineCollection(segs, colors=colors, lw=1.6, zorder=4))
    ax.scatter(*pos[-1, :2], color=FINAL, s=28, zorder=6, label="arrival")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("policy flight, top-down (red = thrusting)", fontsize=10)

    # elements vs time
    n = len(pos)
    tmin = np.arange(n) * J.DT / 60.0
    aa, ee = [], []
    for p_i in range(0, n, 10):
        # elements from position pairs is noisy; recompute from traced state is
        # overkill for a picture — use vis-viva on radius + local speed proxy
        pass
    # simpler: recompute elements along the path from finite-difference velocity
    vel = np.gradient(pos, J.DT, axis=0)
    r = np.linalg.norm(pos, axis=1)
    v2 = (vel ** 2).sum(1)
    energy = 0.5 * v2 - J.MU / r
    a_t = -J.MU / (2 * energy)
    hvec = np.cross(pos, vel)
    e_t = np.sqrt(np.clip(1 - (hvec ** 2).sum(1) / (J.MU * a_t), 0, None))
    ax2.plot(tmin, a_t / rt, color="#334", lw=1.4, label="a / target")
    ax2.plot(tmin, e_t, color=BURN, lw=1.4, label="eccentricity")
    ax2.axhspan(0.95, 1.05, color=TARGET, alpha=0.12, lw=0)
    ax2.axhspan(0.0, 0.05, color=BURN, alpha=0.10, lw=0)
    ax2.set_xlabel("minutes"); ax2.set_ylim(0, 1.6)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title("orbital elements vs 5% tolerance bands", fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "trajectory.png", dpi=110)
    plt.close(fig)


def fig_gif(pos, burn, s0, rt, frames=120):
    ell = orbit_ellipse(s0)
    lim = max(np.abs(pos[:, :2]).max(), rt) * 1.12
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=72)
    idx = np.linspace(1, len(pos) - 1, frames).astype(int)

    def draw(f):
        ax.clear()
        i = idx[f]
        ax.add_patch(plt.Circle((0, 0), J.R_BODY, color=EARTH))
        ax.plot(ell[:, 0], ell[:, 1], "--", color="#bbb", lw=0.8)
        th = np.linspace(0, 2 * np.pi, 150)
        ax.plot(rt * np.cos(th), rt * np.sin(th), ":", color=TARGET, lw=1.0)
        segs = np.stack([pos[: i - 1, :2], pos[1:i, :2]], axis=1)
        cols = [BURN if b else COAST for b in burn[1:i]]
        ax.add_collection(LineCollection(segs, colors=cols, lw=1.5))
        ax.scatter(*pos[i, :2], color="#111", s=22, zorder=6)
        if burn[i]:
            d = pos[i, :2] - pos[i - 1, :2]
            d = d / (np.linalg.norm(d) + 1e-9)
            ax.annotate("", xy=pos[i, :2] - d * lim * 0.09, xytext=pos[i, :2],
                        arrowprops=dict(arrowstyle="-", color=BURN, lw=2.5))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    anim = FuncAnimation(fig, draw, frames=frames)
    anim.save(MEDIA / "maneuver.gif", writer=PillowWriter(fps=24))
    plt.close(fig)


LOG_RE = re.compile(r"iter\s+(\d+)\s+loss=\S+\s+success=([\d.]+)%")


def parse_log(path):
    it, sc = [], []
    for line in Path(path).read_text().splitlines():
        m = LOG_RE.search(line)
        if m:
            it.append(int(m.group(1))); sc.append(float(m.group(2)))
    return np.array(it), np.array(sc)


def fig_training(logdir):
    runs = [  # (file, label, color)
        ("sapo24.log", "miscalibrated per-episode clip — collapse", "#c0392b"),
        ("sapo23.log", "trim-only — oscillates, recovers", "#e67e22"),
        ("sapo27.log", "measured trim+clip — stable", "#2b6cb0"),
        ("sapo30.log", "low-lr polish — holds", "#2e9e6b"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for f, label, c in runs:
        p = Path(logdir) / f
        if not p.exists():
            continue
        it, sc = parse_log(p)
        ax.plot(it, sc, color=c, lw=1.7, label=label)
    ax.set_xlabel("training iteration"); ax.set_ylabel("success on held-out eval (%)")
    ax.set_ylim(0, 100); ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("same start checkpoint, same seed — gradient aggregation decides the outcome",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MEDIA / "training-curve.png", dpi=110)
    plt.close(fig)


def fig_results():
    # Verified numbers. Success: fresh 4096-episode set (eval_probe.py,
    # 2026-07-03). Fuel: float64 dt=1s medians vs the exact-circularization
    # impulsive optimum, clean latches only (verify_probe.py, same date).
    names = ["oracle imitation\n(DAgger BC)", "warm_r19", "warm_r25",
             "warm_r26", "warm_r28 (final)"]
    succ = [79.86, 90.62, 92.14, 91.89, 92.31]
    fuel_names = ["oracle imitation", "warm_r19", "warm_r25", "warm_r26"]
    fuel = [0.989, 1.187, 1.172, 1.032]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    y = np.arange(len(names))
    ax1.barh(y, succ, color=["#999", "#7f9cbf", "#5b86c0", "#2b6cb0", "#2e9e6b"])
    ax1.set_yticks(y, names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 100); ax1.set_xlabel("success, fresh 4096-episode set (%)")
    for yi, v in zip(y, succ):
        ax1.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=8)
    ax1.spines[["top", "right"]].set_visible(False)

    y2 = np.arange(len(fuel_names))
    ax2.barh(y2, fuel, color=["#999", "#7f9cbf", "#5b86c0", "#2b6cb0"])
    ax2.set_yticks(y2, fuel_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.axvline(1.0, color="#333", lw=1.0, ls="--")
    ax2.text(1.005, -0.55, "impulsive optimum", fontsize=7)
    ax2.axvline(0.849, color=FINAL, lw=1.0, ls=":")
    ax2.text(0.852, -0.55, "tolerance-box bound", fontsize=7, color=FINAL)
    ax2.set_xlim(0.8, 1.3)
    ax2.set_xlabel("Δv / impulsive optimum (float64 re-flight, median)")
    for yi, v in zip(y2, fuel):
        ax2.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(MEDIA / "results.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ckpt = sys.argv[1]
    logdir = sys.argv[2] if len(sys.argv) > 2 else "."
    mlp = load_mlp(ckpt)
    s0, rt, pos, burn, latch, e0 = pick_episode(mlp)
    print(f"episode: e0={e0:.2f}, latched after {latch} substeps "
          f"({latch * J.DT / 60:.0f} min)", flush=True)
    fig_trajectory(pos, burn, s0, rt, e0)
    print("trajectory.png done", flush=True)
    fig_gif(pos, burn, s0, rt)
    print("maneuver.gif done", flush=True)
    fig_training(logdir)
    print("training-curve.png done", flush=True)
    fig_results()
    print("results.png done", flush=True)
