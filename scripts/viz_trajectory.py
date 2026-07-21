#!/usr/bin/env python3
"""The realized pump+crank tour as a trajectory picture (R-N36 → R-N38), for the README and website.

  pump_crank_path.png   two panels of the SAME flown path, sampled from the f&g propagator:
                        (a) top-down ecliptic view — the pump spiral through Venus/Earth flybys, legs
                        colored by arrival v∞; (b) edge-on view — the crank physically tilting the
                        orbit out of the ecliptic toward the arcsin(v∞/v_P) ceiling, legs colored by
                        inclination.

  --gif OUT.gif         the same two panels ANIMATED — craft and planets moving, the path revealed
                        leg by leg with a phase/day readout. NOT committed anywhere: the website
                        build renders it straight into its deploy directory (no blobs in git).

Unlike `viz_tour.py` (charts of the recorded scalars), this script RE-SOLVES the tour with the merged
machinery — `beam_constrained_tour.run_search` for the greedy pump chain and `crank_walk`'s dense
`crank_continuations` for the greedy max-inclination walk, both deterministic — then forward-samples
every leg with `fgprop.fg_propagate` and draws the actual heliocentric path. The replayed endpoints are
checked against the recorded R-N37/R-N38 values before the figure is written.

    uv run --with jax --with astroquery --with astropy --with matplotlib python scripts/viz_trajectory.py
    uv run --with jax --with astroquery --with astropy --with matplotlib --with pillow \
        python scripts/viz_trajectory.py --gif /path/to/out.gif --cache .rnd/tour_path_cache.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import beam_constrained_tour as B        # noqa: E402  (R-N37: run_search — the greedy pump chain)
import constrained_tour_discovery as C   # noqa: E402  (R-N36: rv_p, rodrigues, dmax_of, close_launch)
import crank_walk as CW                  # noqa: E402  (R-N38: crank_continuations, _ang_deg)
from fgprop import fg_propagate          # noqa: E402

MEDIA = Path("docs/media")
N_SAMP = 240                             # path samples per leg
# recorded endpoints the replay must reproduce (R-N37 / R-N38 verify outputs)
REC_FINAL_VINF = 16.27
REC_FINAL_I = 27.09
# the public nine-flyby tour = 4 pumping flybys + 5 cranks (the walk's remaining half-rev returns
# refine the inclination by < 0.05 deg and only clutter the drawing)
DRAW_CRANKS = 5


def sample_leg(dep, jd, v_helio, tof_d, n=N_SAMP):
    """Positions (n, 3) km along a ballistic leg from planet `dep` at jd with heliocentric velocity v_helio."""
    r0, _ = C.rv_p(dep, jd)
    st0 = jnp.concatenate([r0, v_helio])
    ts = jnp.linspace(0.0, tof_d * C.DAY, n)
    states = jax.vmap(lambda dt: fg_propagate(st0, dt, mu=C.MU_S, iters=12))(ts)
    return np.asarray(states[:, 0:3])


def replay_tour(t0):
    """Re-solve the pump chain + crank walk and sample every leg. Returns (legs, encounters, meta)."""
    print("  [replaying the greedy pump chain (R-N37)]", flush=True)
    best, _, root = B.run_search(t0, 1)
    if root is None:
        raise SystemExit("no launch leg closed — cannot replay the tour")

    # the launch leg: recover the launch v∞ vector at the chain's chosen tof
    lt = root["launch_tof"]
    u_l, _, seed_v, _ = C.close_launch(t0, lt, "venus")
    v_launch = u_l[2] * C.unit_dir(u_l[0], u_l[1])

    legs = []            # {pos, kind: "launch"|"pump"|"crank", val: arr v∞ or i_out}
    encounters = []      # {name, r, jd}
    rE, vE = C.rv_p("earth", t0)
    legs.append({"pos": sample_leg("earth", t0, vE + v_launch, lt), "kind": "launch",
                 "val": float(jnp.linalg.norm(root["vin"]))})
    encounters.append({"name": "launch", "r": np.asarray(rE), "jd": t0})

    # pump legs up to the saturated node (same last-pumping-leg rule as crank_walk.saturated_node)
    prev = root["mag"]
    sat_idx = 0
    for i, lg in enumerate(best["legs"]):
        if lg["arr_mag"] > prev + 0.05:
            sat_idx = i
        prev = lg["arr_mag"]

    at, jd, vin = "venus", root["jd"], root["vin"]
    for lg in best["legs"][:sat_idx + 1]:
        rP, vP = C.rv_p(at, jd)
        encounters.append({"name": at, "r": np.asarray(rP), "jd": jd})
        dm = C.dmax_of(at, jnp.linalg.norm(vin))
        vout = C.rodrigues(vin, dm * jnp.tanh(lg["u"][0]), lg["u"][1])
        legs.append({"pos": sample_leg(at, jd, vP + vout, lg["tof"]), "kind": "pump",
                     "val": lg["arr_mag"]})
        jd, vin, at = jd + lg["tof"], lg["vinf_arr"], lg["to"]
        print(f"    pump leg -> {at}: tof {lg['tof']:.0f} d, arrival v inf {lg['arr_mag']:.2f}", flush=True)

    vmag = float(jnp.linalg.norm(vin))
    print(f"  saturated node: {at}, v inf {vmag:.2f} (recorded {REC_FINAL_VINF})", flush=True)

    # crank walk — the same greedy max-inclination selection as crank_walk.crank_walk, but keeping
    # each chosen basin's u so the leg can be re-sampled
    print("  [replaying the crank walk (R-N38)]", flush=True)
    i_last = None
    for k in range(CW.MAX_CRANKS):
        rV, vV = C.rv_p(at, jd)
        sols = CW.crank_continuations(at, jd, vin)
        if not sols:
            break
        best_s = None
        for s in sols:
            vout = C.rodrigues(vin, s["dmax"] * jnp.tanh(s["u"][0]), s["u"][1])
            i_out = CW._ang_deg(jnp.cross(rV, vV + vout), jnp.cross(rV, vV))
            if best_s is None or i_out > best_s[0]:
                best_s = (i_out, s, vout)
        i_out, s, vout = best_s
        encounters.append({"name": at, "r": np.asarray(rV), "jd": jd})
        legs.append({"pos": sample_leg(at, jd, vV + vout, s["tof"]), "kind": "crank", "val": i_out})
        jd, vin = jd + s["tof"], s["vinf_arr"]
        i_last = i_out
        print(f"    crank {k + 1}: tof {s['tof']:.0f} d, i_rel -> {i_out:.2f} deg", flush=True)
    rV, _ = C.rv_p(at, jd)
    encounters.append({"name": at, "r": np.asarray(rV), "jd": jd})

    if abs(vmag - REC_FINAL_VINF) > 0.05 or i_last is None or abs(i_last - REC_FINAL_I) > 0.3:
        raise SystemExit(f"replay drifted from the recorded tour (v inf {vmag:.2f} vs {REC_FINAL_VINF}, "
                         f"i {i_last} vs {REC_FINAL_I}) — refusing to draw an unverified path")
    return legs, encounters, {"t0": t0, "jd_end": jd, "seed_v": seed_v}


def planet_orbit(p, jd0, jd1, n=400):
    jds = np.linspace(jd0, jd1, n)
    return np.array([np.asarray(C.rv_p(p, j)[0]) for j in jds])


INK = "#333333"
GRID = "#d8d8d8"


def _i_rel_of_leg(lg, enc):
    """Inclination (deg) of a sampled leg's orbit plane vs Venus's, from the leg's first two samples."""
    hc = np.cross(lg["pos"][0], lg["pos"][1])
    rV, vV = C.rv_p("venus", enc["jd"])
    hV = np.cross(np.asarray(rV), np.asarray(vV))
    c = float(np.dot(hc, hV) / (np.linalg.norm(hc) * np.linalg.norm(hV)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def draw(legs, encounters, meta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    # trim to the public nine-flyby tour; panel (a) colors by arrival v∞ (constant REC_FINAL_VINF on
    # the v∞-neutral cranks), panel (b) by i_rel vs Venus
    legs, encounters = _trim(legs, encounters)

    span = encounters[-1]["jd"] - meta["t0"]
    orbE = planet_orbit("earth", meta["t0"], meta["t0"] + min(span, 370.0)) / C.AU
    orbV = planet_orbit("venus", meta["t0"], meta["t0"] + min(span, 230.0)) / C.AU

    v_norm = colors.Normalize(vmin=5.0, vmax=17.0)
    i_norm = colors.Normalize(vmin=0.0, vmax=28.0)
    v_cm, i_cm = cm.plasma, cm.viridis

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.4, 5.4), width_ratios=[1.0, 1.15])

    for ax, (i0, i1), cmap, norm, key in ((a, (0, 1), v_cm, v_norm, "v_col"),
                                          (b, (0, 2), i_cm, i_norm, "i_col")):
        ax.plot(orbE[:, i0], orbE[:, i1], color="#9db4c8", lw=0.9, ls=":", zorder=1)
        ax.plot(orbV[:, i0], orbV[:, i1], color="#c8a97e", lw=0.9, ls=":", zorder=1)
        ax.scatter([0], [0], s=60, color="#f2b134", edgecolor=INK, lw=0.5, zorder=4)
        for lg in legs:
            p = lg["pos"] / C.AU
            ax.plot(p[:, i0], p[:, i1], color=cmap(norm(lg[key])), lw=1.5, alpha=0.9, zorder=3)
        for e in encounters:
            r = e["r"] / C.AU
            ax.scatter([r[i0]], [r[i1]], s=12, color=INK, zorder=5)
        ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    r0 = encounters[0]["r"] / C.AU
    a.annotate("launch", xy=(r0[0], r0[1]), xytext=(r0[0] + 0.25, r0[1] + 0.25),
               fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    a.set_xlabel("x (AU)")
    a.set_ylabel("y (AU)")
    a.set_aspect("equal")
    a.set_title("top down: four pumping flybys of Venus and Earth\nwalk v∞ from 5.95 to 16.27 km/s — zero Δv",
                fontsize=10)

    b.axhline(0.0, color=INK, lw=0.7, ls="--", alpha=0.6)
    b.text(0.98, 0.03, "ecliptic plane (z stretched)", transform=b.transAxes, ha="right", fontsize=8, color=INK)
    b.set_xlabel("x (AU)")
    b.set_ylabel("z (AU)")
    b.set_title("edge on: five crank flybys at Venus tilt the orbit\nout of the plane to 27.1° — 97% of the ceiling",
                fontsize=10)

    cb1 = fig.colorbar(cm.ScalarMappable(norm=v_norm, cmap=v_cm), ax=a, fraction=0.046, pad=0.03)
    cb1.set_label("arrival v∞ (km/s)", fontsize=8)
    cb2 = fig.colorbar(cm.ScalarMappable(norm=i_norm, cmap=i_cm), ax=b, fraction=0.046, pad=0.03)
    cb2.set_label("inclination vs Venus's orbit (deg)", fontsize=8)

    fig.tight_layout()
    MEDIA.mkdir(parents=True, exist_ok=True)
    fig.savefig(MEDIA / "pump_crank_path.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote pump_crank_path.png")


def _trim(legs, encounters):
    """The public nine-flyby tour, with both per-leg color values (see draw)."""
    n_draw = sum(1 for lg in legs if lg["kind"] != "crank") + DRAW_CRANKS
    legs, encounters = legs[:n_draw], encounters[:n_draw + 1]
    for lg, enc in zip(legs, encounters):
        lg["v_col"] = REC_FINAL_VINF if lg["kind"] == "crank" else lg["val"]
        lg["i_col"] = lg["val"] if lg["kind"] == "crank" else _i_rel_of_leg(lg, enc)
    return legs, encounters


def draw_gif(legs, encounters, meta, out, frames=140, fps=18):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.animation import FuncAnimation, PillowWriter

    legs, encounters = _trim(legs, encounters)
    t0, t_end = meta["t0"], encounters[-1]["jd"]
    leg_t = [np.linspace(encounters[i]["jd"], encounters[i + 1]["jd"], N_SAMP) for i in range(len(legs))]
    orbE = planet_orbit("earth", t0, t0 + 370.0) / C.AU
    orbV = planet_orbit("venus", t0, t0 + 230.0) / C.AU

    v_norm = colors.Normalize(vmin=5.0, vmax=17.0)
    i_norm = colors.Normalize(vmin=0.0, vmax=28.0)
    leg_col = {"a": [cm.plasma(v_norm(lg["v_col"])) for lg in legs],
               "b": [cm.viridis(i_norm(lg["i_col"])) for lg in legs]}

    ft = np.linspace(t0, t_end, frames)
    pE = np.array([np.asarray(C.rv_p("earth", j)[0]) for j in ft]) / C.AU
    pV = np.array([np.asarray(C.rv_p("venus", j)[0]) for j in ft]) / C.AU

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 4.1), width_ratios=[1.0, 1.12], dpi=62)
    trails, crafts, planets = {}, {}, {}
    for ax, pk, (i0, i1) in ((a, "a", (0, 1)), (b, "b", (0, 2))):
        ax.plot(orbE[:, i0], orbE[:, i1], color="#9db4c8", lw=0.8, ls=":", zorder=1)
        ax.plot(orbV[:, i0], orbV[:, i1], color="#c8a97e", lw=0.8, ls=":", zorder=1)
        ax.scatter([0], [0], s=45, color="#f2b134", edgecolor=INK, lw=0.5, zorder=4)
        trails[pk] = [ax.plot([], [], color=leg_col[pk][i], lw=1.4, alpha=0.9, zorder=3)[0]
                      for i in range(len(legs))]
        planets[pk] = {"earth": ax.plot([], [], "o", ms=5, color="#4a7ba6", zorder=5)[0],
                       "venus": ax.plot([], [], "o", ms=4, color="#b8860b", zorder=5)[0]}
        crafts[pk] = ax.plot([], [], "o", ms=4, color=INK, zorder=6)[0]
        ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7)
    all_xy = np.concatenate([lg["pos"] for lg in legs]) / C.AU
    a.set_xlim(all_xy[:, 0].min() - 0.1, all_xy[:, 0].max() + 0.1)
    a.set_ylim(all_xy[:, 1].min() - 0.1, all_xy[:, 1].max() + 0.1)
    a.set_aspect("equal")
    b.set_xlim(all_xy[:, 0].min() - 0.1, all_xy[:, 0].max() + 0.1)
    b.set_ylim(all_xy[:, 2].min() - 0.06, all_xy[:, 2].max() + 0.06)
    b.axhline(0.0, color=INK, lw=0.6, ls="--", alpha=0.5)
    a.set_title("top down: the pump", fontsize=9)
    b.set_title("edge on: the crank", fontsize=9)
    hud = fig.suptitle("", fontsize=9, y=0.995)

    def phase_of(i):
        lg = legs[i]
        if lg["kind"] == "launch":
            return "launch — v∞ 5.95 km/s"
        if lg["kind"] == "pump":
            return f"pumping — v∞ → {lg['v_col']:.1f} km/s (zero Δv)"
        return f"cranking — inclination → {lg['i_col']:.1f}° (ceiling 27.9°)"

    def update(f):
        t = ft[f]
        cur = None
        for i, lg in enumerate(legs):
            ts = leg_t[i]
            if t >= ts[-1]:
                n = N_SAMP
            elif t <= ts[0]:
                n = 0
            else:
                n = int(np.searchsorted(ts, t))
                cur = i
            p = lg["pos"][:n] / C.AU
            for pk, i1 in (("a", 1), ("b", 2)):
                trails[pk][i].set_data(p[:, 0], p[:, i1])
            if n and cur == i:
                for pk, i1 in (("a", 1), ("b", 2)):
                    crafts[pk].set_data([p[-1, 0]], [p[-1, i1]])
        if cur is None:
            cur = len(legs) - 1
        for pk, i1 in (("a", 1), ("b", 2)):
            planets[pk]["earth"].set_data([pE[f, 0]], [pE[f, i1 if i1 == 1 else 2]])
            planets[pk]["venus"].set_data([pV[f, 0]], [pV[f, i1 if i1 == 1 else 2]])
        hud.set_text(f"the flown pump+crank tour — day {t - t0:4.0f}   |   {phase_of(cur)}")
        return []

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)

    # re-encode with a shared 63-color palette + inter-frame deltas (unchanged pixels -> transparent
    # index, disposal=keep): only the trail tip, the moving dots, and the readout change per frame,
    # so this is the difference between a multi-MB file and a few hundred KB
    from PIL import Image, ImageSequence
    im = Image.open(out)
    rgb = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
    pal = rgb[-1].quantize(colors=63, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    qs = [np.array(f.quantize(palette=pal, dither=Image.Dither.NONE)) for f in rgb]
    TRANS = 63
    fr = [Image.fromarray(qs[0], mode="P")]
    for prev, cur in zip(qs, qs[1:]):
        d = cur.copy()
        d[cur == prev] = TRANS
        fr.append(Image.fromarray(d, mode="P"))
    for f in fr:
        f.putpalette(pal.getpalette())
    fr[0].save(out, save_all=True, append_images=fr[1:], duration=int(1000 / fps), loop=0,
               transparency=TRANS, disposal=1, optimize=False)
    print(f"wrote {out} ({Path(out).stat().st_size / 1e6:.2f} MB, {len(fr)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None, help="npz path: reuse the sampled path if present, write it after solving")
    ap.add_argument("--gif", default=None, help="render the animated tour to this path instead of the static PNG")
    args = ap.parse_args()

    if not C.F._require_cache():
        return
    for p in ("earth", "venus"):
        C._tab(p)
    if args.cache and Path(args.cache).exists():
        d = np.load(args.cache, allow_pickle=True)
        legs, encounters, meta = list(d["legs"]), list(d["encounters"]), d["meta"].item()
        print(f"loaded cached path from {args.cache}")
    else:
        t0 = C.F._start_jd() + 400.0
        legs, encounters, meta = replay_tour(t0)
        if args.cache:
            np.savez(args.cache, legs=np.array(legs, dtype=object),
                     encounters=np.array(encounters, dtype=object), meta=np.array(meta, dtype=object))
            print(f"cached path to {args.cache}")
    if args.gif:
        draw_gif(legs, encounters, meta, args.gif)
    else:
        draw(legs, encounters, meta)


if __name__ == "__main__":
    main()
