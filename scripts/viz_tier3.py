#!/usr/bin/env python3
"""Tier-3 README figures — the multi-body capture and the J2 node-change beat.

These showcase the recent research threads the original README doesn't cover:
  cr3bp_capture.png  Earth-Moon rotating frame: the stable manifold of an L2
                     Lyapunov orbit funnelling onto the Moon, and one verified
                     BALLISTIC capture arc spiralling into a bound lunar orbit.
                     Reproduces scripts/cr3bp_manifold.py --capture (Build H R-H3),
                     pure float64 numpy dynamics — no learned checkpoint.
  cr3bp_capture.gif  the same capture arc, animated into its bound Moon revs.
  j2_beat.png        the J2 node-change result (Builds K/L): a diff-sim policy
                     DIVES to speed J2 nodal drift and reaches a target RAAN for
                     less Δv than paying the impulsive plane change — and the beat
                     over the (fair) passive-J2 baseline GROWS with the node angle.

Pictures, not claims: every number is from the verified campaign logs / re-run
here from the same scripts. Small figures only (regenerable; nothing committed
that a script can't rebuild).

    uv run --with jax --with matplotlib --with pillow python scripts/viz_tier3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import cr3bp_sim as C
import cr3bp_manifold as M

MEDIA = Path("docs/media")
MEDIA.mkdir(parents=True, exist_ok=True)

EARTH = "#2b6cb0"
MOON = "#9aa0a8"
ORBIT = "#e0a458"
TUBE = "#5aa9e6"
CAPTURE = "#d1495b"
BOUND = "#2a9d8f"
INK = "#333333"


# ---------------------------------------------------------------- CR3BP capture
def compute_capture(Ax=0.02, dt=1e-4, n_seed=40, pos_disp=1e-4, t_prop=6.0,
                    rec=20, verify_steps=40000):
    """Reproduce Build H R-H3: L2 Lyapunov orbit → stable manifold → best ballistic
    capture arc → its bounded lunar orbit. Returns everything needed to draw it."""
    lp = C.lagrange_points()
    xL2 = lp["L2"]
    s0, T, N, orbit, Mm, v_s, lam, w, res = M.orbit_and_monodromy(xL2, Ax, dt)
    ics, labels = M.manifold_ics(s0, N, v_s, n_seed, pos_disp, dt)
    n_prop = int(round(t_prop / dt))
    tube = M.propagate_batch(ics, -dt, n_prop, record_every=rec)      # (F,K,6) backward
    best = None
    for k in range(ics.shape[0]):
        tk = tube[:, k, :]
        d, sp, E = M.moon_rel_np(tk)
        if d.min() < M.R_HILL:
            j = int(np.argmin(d))
            if E[j] < 0.0 and (best is None or E[j] < best["E"]):
                best = {"k": k, "E": float(E[j]), "d": float(d[j]),
                        "s_ca": tk[j].copy(), "arc": tk[:j + 1].copy()}
    bound = M.propagate_batch(best["s_ca"][None, :], dt, verify_steps,
                              record_every=rec)[:, 0, :]
    d_b = np.linalg.norm(bound[:, 0:3] - M.R_MOON, axis=1)
    left = np.where(d_b > M.R_HILL)[0]
    bound = bound[:left[0]] if len(left) else bound
    revs = M.count_moon_revs(bound)
    bt = (len(bound) * rec) * dt * C.T_UNIT_S / 86400.0
    return dict(lp=lp, orbit=orbit, tube=tube, best=best, bound=bound,
                revs=revs, days=bt, closest_km=best["d"] * C.L_UNIT_KM,
                E=best["E"])


def _draw_bodies(ax, lp, zoom):
    ax.add_patch(plt.Circle((1 - C.MU, 0), M.R_HILL, color=MOON, alpha=0.12,
                            zorder=1))
    ax.add_patch(plt.Circle((1 - C.MU, 0), 1737.4 / C.L_UNIT_KM, color=MOON,
                            zorder=6))
    if not zoom:
        ax.add_patch(plt.Circle((-C.MU, 0), 6378.0 / C.L_UNIT_KM, color=EARTH,
                                zorder=6))
        ax.text(-C.MU, -0.052, "Earth", color=EARTH, ha="center", fontsize=8.5)
    ax.text(1 - C.MU, -M.R_HILL - 0.028 if not zoom else -M.R_HILL - 0.012,
            "Moon", color="#6b7280", ha="center", fontsize=8.5)
    for nm in ("L1", "L2"):
        ax.plot(lp[nm], 0, "+", color=INK, ms=8, mew=1.4, zorder=7)
        ax.annotate(nm, (lp[nm], 0), textcoords="offset points",
                    xytext=(3, 5), fontsize=8, color=INK)


def fig_capture(cap):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9))
    for ax, zoom in zip(axes, (False, True)):
        _draw_bodies(ax, cap["lp"], zoom)
        # manifold tube — faint bundle
        for k in range(cap["tube"].shape[1]):
            tk = cap["tube"][:, k, :]
            ax.plot(tk[:, 0], tk[:, 1], color=TUBE, lw=0.35, alpha=0.28, zorder=2)
        # L2 Lyapunov orbit
        o = cap["orbit"]
        ax.plot(o[:, 0], o[:, 1], color=ORBIT, lw=1.8, zorder=4)
        # best ballistic capture arc + its bound lunar orbit
        arc = cap["best"]["arc"]
        ax.plot(arc[:, 0], arc[:, 1], color=CAPTURE, lw=1.7, zorder=5)
        ax.plot(cap["bound"][:, 0], cap["bound"][:, 1], color=BOUND, lw=1.4,
                zorder=5)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_xlim(-0.15, 1.32); axes[0].set_ylim(-0.42, 0.42)
    axes[0].set_title("Earth–Moon rotating frame: L2 manifold → the Moon",
                      fontsize=10)
    z = 1 - C.MU
    axes[1].set_xlim(z - 0.26, z + 0.20); axes[1].set_ylim(-0.23, 0.23)
    axes[1].set_title(
        f"ballistic capture: {cap['closest_km']:.0f} km, E_moon={cap['E']:+.2f} "
        f"(bound), {cap['revs']:.1f} revs / ~{cap['days']:.0f} d", fontsize=10)
    leg = [Line2D([0], [0], color=ORBIT, lw=2, label="L2 Lyapunov orbit"),
           Line2D([0], [0], color=TUBE, lw=2, alpha=0.6, label="stable manifold"),
           Line2D([0], [0], color=CAPTURE, lw=2, label="capture transfer arc"),
           Line2D([0], [0], color=BOUND, lw=2, label="bound lunar orbit (ballistic)")]
    axes[1].legend(handles=leg, loc="upper right", fontsize=7.5, frameon=False)
    fig.suptitle("Differentiable CR3BP: manifold-seeded ballistic capture at the "
                 "Moon (no capture burn)", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "cr3bp_capture.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("cr3bp_capture.png done", flush=True)


def gif_capture(cap):
    arc = cap["best"]["arc"]; bound = cap["bound"]
    path = np.vstack([arc[:, :2], bound[:, :2]])
    n_arc = len(arc)
    step = max(1, len(path) // 90)
    idx = list(range(0, len(path), step)) + [len(path) - 1]
    z = 1 - C.MU
    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=70)

    def draw(fi):
        ax.clear()
        _draw_bodies(ax, cap["lp"], zoom=True)
        ax.plot(cap["orbit"][:, 0], cap["orbit"][:, 1], color=ORBIT, lw=1.4,
                alpha=0.7, zorder=4)
        i = idx[fi]
        # draw the arc portion in capture colour, the bound portion in bound colour
        ax.plot(arc[:min(i + 1, n_arc), 0], arc[:min(i + 1, n_arc), 1],
                color=CAPTURE, lw=1.6, zorder=5)
        if i >= n_arc:
            ax.plot(bound[:i - n_arc + 1, 0], bound[:i - n_arc + 1, 1],
                    color=BOUND, lw=1.6, zorder=5)
        col = CAPTURE if i < n_arc else BOUND
        ax.scatter(path[i, 0], path[i, 1], color=col, s=30, zorder=8)
        phase = "transfer arc" if i < n_arc else "bound at the Moon"
        ax.set_title(f"ballistic capture — {phase}", fontsize=10)
        ax.set_xlim(z - 0.26, z + 0.20); ax.set_ylim(-0.23, 0.23)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    anim = FuncAnimation(fig, draw, frames=len(idx))
    anim.save(MEDIA / "cr3bp_capture.gif", writer=PillowWriter(fps=20))
    plt.close(fig)
    print("cr3bp_capture.gif done", flush=True)


# ------------------------------------------------------------------- J2 the beat
def compute_j2(alt=1500.0, inc=51.6, draan=30.0, days=8.0, dt=120.0,
               harm=8, amax=2e-5, steps=700, lr=2e-2):
    """Run a short j2_policy optimization to get the DISCOVERED dive, plus the
    passive coast, both recording altitude(t) and RAAN(t). Shorter/coarser than a
    campaign run (fast figure regen) but reproduces the qualitative dive+beat."""
    import jax
    import jax.numpy as jnp
    import j2_policy as JP

    a0 = JP.R_BODY + alt
    s0 = jnp.asarray(JP.orbit_state(a0, 0.0, inc, 0.0))
    n = round(days * JP.DAY / dt)
    draan_signed = -np.sign(np.cos(np.radians(inc))) * abs(draan)
    tgt = (float(a0), 0.0, float(np.radians(inc)), float(np.radians(draan_signed)))
    r_boost = 42164.0
    gfn = jax.jit(jax.value_and_grad(
        lambda x: JP.objective(x, s0, dt, n, amax, tgt, r_boost), has_aux=True))
    best_x, _ = JP.adam(gfn, np.zeros((3, 2, harm)), steps, lr, log_every=10**9)

    def record(coeffs):
        a_rtn = JP.rtn_profile(jnp.asarray(coeffs), n, amax)

        def st(rv, a):
            r, v = rv[:3], rv[3:]
            rh = r / jnp.sqrt(r @ r)
            h = jnp.cross(r, v); nh = h / jnp.sqrt(h @ h)
            th = jnp.cross(nh, rh)
            rv = rv.at[3:].add((a[0] * rh + a[1] * th + a[2] * nh) * dt)
            rv2 = JP.rk4(rv, dt)
            el = JP.elements(rv2)
            return rv2, jnp.array([el[0], el[3]])         # (a, raan)
        _, out = jax.lax.scan(st, s0, a_rtn)
        return np.asarray(out)

    active = record(best_x)
    passive = record(np.zeros((3, 2, harm)))
    t = np.arange(n) * dt / JP.DAY
    tgt_deg = draan_signed % 360.0
    return dict(t=t, act_alt=active[:, 0] - JP.R_BODY,
                pas_alt=passive[:, 0] - JP.R_BODY,
                act_raan=np.degrees(active[:, 1]), pas_raan=np.degrees(passive[:, 1]),
                tgt_deg=tgt_deg, draan=draan_signed)


# Verified campaign frontier (Build L, .rnd/campaign-2026-07-07-j2-eccentric-sweep):
# active vs passive-J2 total Δv ratio at ~80%-passive-coverage budgets.
FRONTIER = [(30, 0.876), (60, 0.449), (90, 0.313)]


def fig_j2(j2):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    t = j2["t"]
    # (1) altitude schedule — the discovered dive vs passive flat
    ax = axes[0]
    ax.plot(t, j2["pas_alt"], color="#9aa0a8", lw=1.8, label="passive (coast)")
    ax.plot(t, j2["act_alt"], color=EARTH, lw=1.8, label="diff-sim policy")
    ax.set_xlabel("time (days)"); ax.set_ylabel("altitude (km)")
    ax.set_title("the policy DIVES to speed J2 drift", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    # (2) RAAN drift-from-start; the policy reaches target, passive falls short
    ax = axes[1]
    def drift(raan_deg):
        d = np.degrees(np.unwrap(np.radians(raan_deg)))
        return d - d[0]
    pas, act = drift(j2["pas_raan"]), drift(j2["act_raan"])
    ax.axhline(j2["draan"], color=CAPTURE, ls="--", lw=1.2, label="target ΔΩ")
    ax.plot(t, pas, color="#9aa0a8", lw=1.8, label="passive (coast)")
    ax.plot(t, act, color=EARTH, lw=1.8, label="diff-sim policy")
    # the shortfall passive must pay the plane change for
    ax.annotate("", xy=(t[-1], j2["draan"]), xytext=(t[-1], pas[-1]),
                arrowprops=dict(arrowstyle="<->", color=CAPTURE, lw=1.1))
    ax.text(t[-1] - 0.15, 0.5 * (pas[-1] + j2["draan"]), "passive\nshortfall",
            ha="right", va="center", fontsize=7.5, color=CAPTURE)
    ax.set_xlabel("time (days)"); ax.set_ylabel("node drift from start (deg)")
    ax.set_title("policy reaches target; passive falls short", fontsize=10)
    ax.set_ylim(j2["draan"] - 4, 3)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    # (3) frontier — beat grows with node angle
    ax = axes[2]
    dO = [d for d, _ in FRONTIER]; r = [x for _, x in FRONTIER]
    ax.axhline(1.0, color="#bbb", ls="--", lw=1.0)
    ax.plot(dO, r, "-o", color=BOUND, lw=2.0, ms=8, zorder=3)
    for d, x in FRONTIER:
        ax.annotate(f"{(1-x)*100:.0f}% cheaper", (d, x), textcoords="offset points",
                    xytext=(9, 6), fontsize=8.5, color=BOUND, fontweight="bold")
    ax.set_xlabel("node change ΔΩ (deg)")
    ax.set_ylabel("Δv vs passive-J2  (ratio)")
    ax.set_title("beat grows with node angle", fontsize=10)
    ax.set_ylim(0.2, 1.12); ax.set_xlim(20, 100); ax.set_xticks(dO)
    ax.text(90, 1.02, "passive-J2 baseline", ha="right", fontsize=7.5, color="#999")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("J2 node change: a diff-sim policy beats the J2-aware analytic "
                 "optimum by exploiting nodal drift", fontsize=11)
    fig.tight_layout()
    fig.savefig(MEDIA / "j2_beat.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("j2_beat.png done", flush=True)


if __name__ == "__main__":
    print("computing CR3BP capture (float64 numpy, ~30-60 s)...", flush=True)
    cap = compute_capture()
    fig_capture(cap)
    gif_capture(cap)
    print("computing J2 policy dive (short JAX optimize)...", flush=True)
    j2 = compute_j2()
    fig_j2(j2)
    print("done.", flush=True)
