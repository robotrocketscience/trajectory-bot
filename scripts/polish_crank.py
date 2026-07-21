#!/usr/bin/env python3
"""Does end-to-end GRADIENT POLISH improve the CRANK (inclination)? (Build N, R-N42).

R-N41 polished the PUMP objective (final v inf) and found gradient polish on the continuous launch freedom
pays (+0.58 km/s via a +8.9 d launch-epoch walk, frozen basins). This is the sibling question for the
CRANK: does the same 2-D line-searched ascent, with the objective swapped to the final INCLINATION i_rel,
raise it past the fraction of the arcsin(v inf/v_P) ceiling that R-N38's coarse greedy crank walk reached —
or is the re-closure tax a hard CONTINUOUS wall? ONE knob vs R-N41: objective = final i_rel (not v inf),
through the full pump+crank chain, frozen basin inits, continuous freedom = (launch epoch, launch tof).

A measure-first probe (recorded) established the well-posed objective: R-N38's crank raises inclination in
its LEADING TIGHTLY-CLOSING legs (miss < 0.1 SOI, i_rel 1.16 -> 25.72 = 92% of the ceiling); the final push
to R-N38's 27.09 (97%) comes from WEAK outer-SOI grazes (miss up to 0.8 SOI) whose patched-conic arrival
v inf is not conserved when threaded, so they cannot enter a differentiable v inf-conserving chain. This
round therefore polishes the inclination of the tightly-closed crank (the legs that do the real flyby work).

  H-N42a  gradients FLOW: d(final i_rel)/d(t0, ltof) through the pump+crank unrolled-GN chain is finite and
          FD-sign-matched (h=0.05 d) in both components.
  H-N42b  crank polish PAYS: line-searched ascent raises final i_rel by >= 0.5 deg over the built chain,
          every encounter still sub-SOI, turns <= dmax.
  H-N42c  the gain is GENUINE crank improvement, not a higher ceiling: the FRACTION i_rel/arcsin(v inf/v_P)
          INCREASES at the polished point. REFUTE-BY: the fraction does not rise (> +0.5% of the built) —
          inclination polish acts only through pumping v inf (raising the ceiling), not the re-closure tax.

Frozen basin inits (gradients cannot hop basins, R-N7). lax.scan GN (R-N41) keeps the graph small.
Mechanism/DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/polish_crank.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import beam_constrained_tour as B        # noqa: E402  (R-N37 pump chain)
import constrained_tour_discovery as C   # noqa: E402  (R-N36 architecture)
import crank_walk as CW                  # noqa: E402  (R-N38 crank continuations)
from polish_tour import gn_scan          # noqa: E402  (R-N41 scan-based Levenberg GN)

STEP_SIZES = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
MAX_STEPS = 25
GAIN_BAR = 0.5          # deg, H-N42b
FRAC_BAR = 0.5          # %, H-N42c
FD_H = 0.05
TIGHT = 0.1             # crank legs kept if miss < TIGHT * SOI (threadable, v inf-conserving)


def build_frozen(t0_ref):
    """Greedy pump chain to the saturated node + greedy max-i crank walk; return frozen inits (launch,
    pump legs to the saturated node, tightly-closing crank legs) and reference scalars."""
    best, _, root = B.run_search(t0_ref, 1)
    lt0 = float(root["launch_tof"])
    u_l0 = C.close_launch(t0_ref, lt0, "venus")[0]
    all_pump_u = [jnp.asarray(lg["u"]) for lg in best["legs"]]
    all_pairs = list(zip(["venus"] + [lg["to"] for lg in best["legs"]][:-1],
                         [lg["to"] for lg in best["legs"]]))
    prev = root["mag"]; sat = 0
    for i, lg in enumerate(best["legs"]):
        if lg["arr_mag"] > prev + 0.05:
            sat = i
        prev = lg["arr_mag"]
    n_pump = sat + 1
    pump_u, pump_pairs = all_pump_u[:n_pump], all_pairs[:n_pump]      # truncate to the saturated node
    at, jd, vin = "venus", root["jd"], root["vin"]
    for lg in best["legs"][:n_pump]:
        jd, vin, at = jd + lg["tof"], lg["vinf_arr"], lg["to"]

    crank_u = []
    jc, vc = jd, vin
    for _ in range(CW.MAX_CRANKS):
        rV, vV = C.rv_p("venus", jc)
        sols = CW.crank_continuations("venus", jc, vc)
        if not sols:
            break
        bb = None
        for s in sols:
            vo = C.rodrigues(vc, s["dmax"] * jnp.tanh(s["u"][0]), s["u"][1])
            io = CW._ang_deg(jnp.cross(rV, vV + vo), jnp.cross(rV, vV))
            if bb is None or io > bb[0]:
                bb = (io, s)
        crank_u.append(jnp.asarray(bb[1]["u"]))
        jc, vc = jc + bb[1]["tof"], bb[1]["vinf_arr"]

    # keep only leading tightly-closing crank legs (threadable)
    jc2, vc2, n_tight = jd, vin, 0
    for u0 in crank_u:
        dm = C.dmax_of("venus", jnp.linalg.norm(vc2))
        vo = C.rodrigues(vc2, dm * jnp.tanh(u0[0]), u0[1])
        miss, va = C.shoot("venus", "venus", jc2, vo, u0[2])
        if float(jnp.linalg.norm(miss)) < TIGHT * C.SOI_KM["venus"]:
            n_tight += 1; jc2, vc2 = jc2 + float(u0[2]), va
        else:
            break
    crank_u = crank_u[:n_tight]
    return dict(lt0=lt0, u_l0=u_l0, pump_pairs=pump_pairs, pump_u=pump_u, crank_u=crank_u,
                n_pump=n_pump, n_crank=len(crank_u))


def make_irel_fn(fr):
    """(t0, ltof) -> (final i_rel deg, v inf entering the last crank, ceiling deg, all-leg misses km)."""
    def chain(t0, ltof):
        def res_l(u):
            miss, _ = C.shoot("earth", "venus", t0, u[2] * C.unit_dir(u[0], u[1]), ltof)
            return miss / 1e6
        u_l = gn_scan(res_l, fr["u_l0"], 15, step_max=(0.4, 0.2, 1.5))
        misses = [jnp.linalg.norm(res_l(u_l)) * 1e6]
        _, va = C.shoot("earth", "venus", t0, u_l[2] * C.unit_dir(u_l[0], u_l[1]), ltof)
        vin, jd = va, t0 + ltof
        for (dep, arr), u0 in zip(fr["pump_pairs"], fr["pump_u"]):
            dm = C.dmax_of(dep, jnp.linalg.norm(vin))
            def rf(u, vin=vin, dm=dm, dep=dep, arr=arr, jd=jd):
                vo = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1]); m, _ = C.shoot(dep, arr, jd, vo, u[2]); return m / 1e6
            u = gn_scan(rf, u0, 15)
            misses.append(jnp.linalg.norm(rf(u)) * 1e6)
            vo = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1]); _, va = C.shoot(dep, arr, jd, vo, u[2])
            jd, vin = jd + u[2], va
        last_vout, last_jd, last_vin = None, jd, vin
        for u0 in fr["crank_u"]:
            dm = C.dmax_of("venus", jnp.linalg.norm(vin))
            def rf(u, vin=vin, dm=dm, jd=jd):
                vo = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1]); m, _ = C.shoot("venus", "venus", jd, vo, u[2]); return m / 1e6
            u = gn_scan(rf, u0, 15)
            misses.append(jnp.linalg.norm(rf(u)) * 1e6)
            vo = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1]); _, va = C.shoot("venus", "venus", jd, vo, u[2])
            last_vout, last_jd, last_vin = vo, jd, vin
            jd, vin = jd + u[2], va
        rV, vV = C.rv_p("venus", last_jd)
        hc = jnp.cross(rV, vV + last_vout); hv = jnp.cross(rV, vV)
        cang = jnp.dot(hc, hv) / (jnp.linalg.norm(hc) * jnp.linalg.norm(hv))
        irel = jnp.degrees(jnp.arccos(jnp.clip(cang, -1 + 1e-7, 1 - 1e-7)))
        vmag = jnp.linalg.norm(last_vin)
        ceil = jnp.degrees(jnp.arcsin(jnp.clip(vmag / jnp.linalg.norm(vV), 0.0, 1.0)))
        return irel, vmag, ceil, jnp.stack(misses)
    return chain


def _closed(misses, arrs):
    return all(float(m) < C.SOI_KM[p] for m, p in zip(misses, arrs))


def verify(args):
    print("=== R-N42: does end-to-end GRADIENT POLISH improve the CRANK (inclination)? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus"):
        C._tab(p)
    t0_ref = sjd + 400.0
    print("  R-N41 machinery, objective swapped v inf -> final i_rel; frozen basins; continuous (t0, ltof).")
    print("  Crank measured on its tightly-closing legs (the loose outer-SOI grazes to R-N38's 97% are not")
    print("  v inf-conserving when threaded — a measure-first finding, recorded).\n")

    print("  [build pump+crank chain]", flush=True)
    fr = build_frozen(t0_ref)
    arrs = ["venus"] + [p[1] for p in fr["pump_pairs"]] + ["venus"] * fr["n_crank"]
    fv = jax.jit(make_irel_fn(fr))
    i0, vm0, ceil0, m0 = fv(jnp.float64(t0_ref), jnp.float64(fr["lt0"]))
    i0, vm0, ceil0 = float(i0), float(vm0), float(ceil0)
    frac0 = 100 * i0 / ceil0
    print(f"  {fr['n_pump']} pump legs + {fr['n_crank']} tight crank legs; built i_rel {i0:.3f} deg, "
          f"v inf {vm0:.3f}, ceiling {ceil0:.3f}, fraction {frac0:.2f}%\n")

    # H-N42a
    g = jax.jit(jax.grad(lambda a, b: fv(a, b)[0], argnums=(0, 1)))(jnp.float64(t0_ref), jnp.float64(fr["lt0"]))
    g = (float(g[0]), float(g[1]))
    fd = []
    for dt, dl in ((FD_H, 0.0), (0.0, FD_H)):
        vp = float(fv(jnp.float64(t0_ref + dt), jnp.float64(fr["lt0"] + dl))[0])
        vm = float(fv(jnp.float64(t0_ref - dt), jnp.float64(fr["lt0"] - dl))[0])
        fd.append((vp - vm) / (2 * FD_H))
    finite = bool(np.isfinite(g).all())
    signs = [bool(np.sign(g[k]) == np.sign(fd[k])) for k in range(2)]
    a_ok = finite and all(signs)
    print(f"  grad d(i_rel)/d(t0, ltof) = ({g[0]:+.4f}, {g[1]:+.4f}) deg/day (finite={finite})")
    print(f"  central FD (h={FD_H})      = ({fd[0]:+.4f}, {fd[1]:+.4f})  sign match: t0={signs[0]}, ltof={signs[1]}\n")

    # H-N42b: ascent on i_rel, step accepted only if it improves AND every leg stays sub-SOI
    print("  [line-searched gradient ascent on i_rel]", flush=True)
    g_fn = jax.jit(jax.grad(lambda a, b: fv(a, b)[0], argnums=(0, 1)))
    t0, lt = float(t0_ref), float(fr["lt0"])
    i_cur = i0
    path = []
    for _ in range(MAX_STEPS):
        gg = np.array(g_fn(jnp.float64(t0), jnp.float64(lt)))
        gn = np.linalg.norm(gg)
        if not np.isfinite(gn) or gn == 0:
            break
        d = gg / gn
        best = None
        for s in STEP_SIZES:
            it, vt, ct, mt = fv(jnp.float64(t0 + s * d[0]), jnp.float64(lt + s * d[1]))
            if float(it) > i_cur and _closed(mt, arrs):
                best = (s, float(it), float(vt), float(ct)); break
        if best is None:
            break
        s, i_cur, v_cur, c_cur = best
        t0, lt = t0 + s * d[0], lt + s * d[1]
        path.append((t0 - t0_ref, lt - fr["lt0"], i_cur, v_cur, 100 * i_cur / c_cur))
    for k, (dt, dl, ii, vv, ff) in enumerate(path, 1):
        print(f"    step {k:2d}: t0 {dt:+7.2f} d  ltof {dl:+6.2f} d  i_rel {ii:.3f}  v inf {vv:.3f}  frac {ff:.2f}%")
    i_p, v_p, ceil_p = (i_cur, path[-1][3], i_cur / (path[-1][4] / 100)) if path else (i0, vm0, ceil0)
    frac_p = 100 * i_p / ceil_p
    gain = i_p - i0
    frac_gain = frac_p - frac0
    b_ok = gain >= GAIN_BAR
    c_ok = frac_gain > FRAC_BAR
    print(f"\n  polished: i_rel {i0:.3f} -> {i_p:.3f} deg (gain {gain:+.3f}); v inf {vm0:.3f} -> {v_p:.3f} "
          f"(ceiling {ceil0:.3f} -> {ceil_p:.3f}); fraction {frac0:.2f}% -> {frac_p:.2f}% ({frac_gain:+.2f} pts)\n")

    print(f"  → H-N42a {'SUPPORTED' if a_ok else 'REFUTED'}: the crank-objective gradient "
          f"{'FLOWS' if a_ok else 'fails'} — finite={finite}, FD sign match {signs}.")
    print(f"  → H-N42b {'SUPPORTED' if b_ok else 'REFUTED'}: crank polish "
          f"{'raises inclination' if b_ok else 'does NOT raise inclination'} by {gain:+.3f} deg "
          f"{'≥' if b_ok else '<'} the {GAIN_BAR} deg bar.")
    print(f"  → H-N42c {'SUPPORTED' if c_ok else 'REFUTED'}: the gain is "
          f"{'GENUINE crank improvement' if c_ok else 'NOT genuine crank improvement'} — fraction of ceiling "
          f"{frac_gain:+.2f} pts {'>' if c_ok else '≤'} {FRAC_BAR}%"
          + ("." if c_ok else " — inclination polish acts through PUMPING v inf (raising the ceiling), not by "
             "beating the re-closure tax; the tax is a near-hard continuous wall on the FRACTION."))

    print(f"\n  → verdicts: H-N42a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N42b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N42c {'SUPPORTED' if c_ok else 'REFUTED'}")
    if b_ok and not c_ok:
        print("  NET: gradient polish RAISES the absolute inclination but the gain rides the PUMP — the launch")
        print("    walk that raises v inf raises the arcsin(v inf/v_P) ceiling, and the crank tracks it at a")
        print("    roughly FIXED fraction. The re-closure tax that pins the crank below its ceiling is a")
        print("    near-hard CONTINUOUS wall: gradients cannot buy back the phasing the encounters spend. So")
        print("    'search chooses basins, gradients polish' completes for the PUMP (R-N41: real v inf gain)")
        print("    but for the CRANK the continuous freedom only moves inclination THROUGH v inf — the discrete")
        print("    basin choice sets the crank fraction. Judged against the pre-registered falsifier.")
    elif b_ok and c_ok:
        print("  NET: gradient polish genuinely improves the crank — the fraction of the ceiling RISES, so the")
        print("    re-closure tax is NOT a hard wall; continuous launch tuning buys back some of the phasing.")
    else:
        print("  NET: the coarse crank already sits at a continuous local optimum in inclination — the discrete")
        print("    basin choice fully determines the crank outcome; gradients buy nothing. (Probe-consistent.)")
    print("    Scope: one epoch, tightly-closing crank legs (loose outer-SOI grazes to R-N38's 97% excluded as")
    print("    non-threadable), frozen basins. Mechanism/DISCOVERY study, never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
