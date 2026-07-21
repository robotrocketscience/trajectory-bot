#!/usr/bin/env python3
"""Does end-to-end GRADIENT POLISH improve the full searched pump chain? (Build N, R-N41).

The north-star claim since R-N36 is "coarse search chooses the discrete basins, gradients polish the
continuous freedom" — but the polish half was only ever probed: one gradient wrt launch tof at depth 2,
coarse steps. This round runs the polish for real on R-N37's full 6-flyby chain. The continuous freedoms
are launch epoch t0 and launch tof (everything else is GN-determined given the frozen basin inits); the
outer loop is 2-D line-searched gradient ascent on final v inf, with a step ACCEPTED only if it both
improves the objective and keeps every encounter GN-closed sub-SOI — closure is enforced at every step,
never traded for objective (the R-N34 lesson, structurally).

ONE knob vs R-N37: add the outer differentiable polish loop on top of the UNCHANGED searched chain.

  H-N41a  gradients FLOW at full depth: d(final v inf)/d(t0, ltof) through the 6-flyby unrolled-GN chain
          is finite and agrees in SIGN with central finite differences (h=0.05 d) in both components.
  H-N41b  polish PAYS: line-searched ascent (steps 2.0 down to 0.01 d, <= 25 accepted steps) improves
          final v inf by >= +0.05 km/s over the searched chain.
  H-N41c  polish PRESERVES feasibility: at the polished (t0, ltof) every encounter re-closes sub-SOI with
          turns <= dmax under the independent 40-iteration solver (|v inf| conserved by construction).

Measure-first probe (recorded): the frozen-init unrolled chain reproduces the searched tour to 2e-10; the
gradient is finite and FD-sign-consistent; the fine ltof landscape is SMOOTH at 0.02-d resolution but only
worth +0.013 within +/-1 d — the gain, if any, lives in multi-day walking along the stronger t0 direction.
Engineering: the GN inner loop here is lax.scan-based (same Levenberg math as C.gn_close) because the
python-unrolled graph cost an 11-minute XLA compile; a reproduction assert guards the equivalence.

Sun-only two-body legs, patched-conic flybys, real cached-JPL ephemeris, frozen basin inits (gradients
cannot hop basins, R-N7). Mechanism/DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/polish_tour.py --verify
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp
from jax import jacfwd, lax

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import beam_constrained_tour as B        # noqa: E402  (R-N37: run_search, pair — the searched chain)
import constrained_tour_discovery as C   # noqa: E402  (R-N36: shoot, rodrigues, dmax_of, close_launch)

STEP_SIZES = (2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)   # days, tried largest-first each ascent step
MAX_STEPS = 25
GAIN_BAR = 0.05                                            # km/s, the pre-registered H-N41b bar
FD_H = 0.05                                                # days, central-difference step for H-N41a


def gn_scan(res_fn, u0, iters, lam=1e-3, step_max=(0.3, 0.3, 15.0)):
    """C.gn_close's Levenberg-damped Gauss-Newton, carried by lax.scan so the XLA graph stays small and
    reverse-differentiable (the python-unrolled version compiles for ~11 minutes at chain depth)."""
    sm = jnp.asarray(step_max)
    J_fn = jacfwd(res_fn)

    def body(u, _):
        r = res_fn(u)
        J = J_fn(u)
        JTJ = J.T @ J
        JTJ = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + 1e-12 * jnp.eye(3)
        du = jnp.linalg.solve(JTJ, J.T @ r)
        return u - jnp.clip(du, -sm, sm), None

    u, _ = lax.scan(body, u0, None, length=iters)
    return u


def make_chain_fn(u_l0, u_f0s, seq_pairs, gn_iters=15):
    """(t0, ltof) -> (final v inf, per-leg misses in km) through the frozen-init unrolled-GN chain."""
    def chain(t0, ltof):
        def res_l(u):
            miss, _ = C.shoot("earth", "venus", t0, u[2] * C.unit_dir(u[0], u[1]), ltof)
            return miss / 1e6
        u_l = gn_scan(res_l, u_l0, gn_iters, step_max=(0.4, 0.2, 1.5))
        miss_l = jnp.linalg.norm(res_l(u_l)) * 1e6
        _, va = C.shoot("earth", "venus", t0, u_l[2] * C.unit_dir(u_l[0], u_l[1]), ltof)
        vin, jd = va, t0 + ltof
        misses = [miss_l]
        for (dep, arr), u0 in zip(seq_pairs, u_f0s):
            dm = C.dmax_of(dep, jnp.linalg.norm(vin))

            def res_f(u, vin=vin, dm=dm, dep=dep, arr=arr, jd=jd):
                vout = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1])
                miss, _ = C.shoot(dep, arr, jd, vout, u[2])
                return miss / 1e6

            u = gn_scan(res_f, u0, gn_iters)
            misses.append(jnp.linalg.norm(res_f(u)) * 1e6)
            vout = C.rodrigues(vin, dm * jnp.tanh(u[0]), u[1])
            _, va = C.shoot(dep, arr, jd, vout, u[2])
            jd = jd + u[2]
            vin = va
        return jnp.linalg.norm(vin), jnp.stack(misses)
    return chain


def closed(misses, seq_arrs):
    return all(float(m) < C.SOI_KM[p] for m, p in zip(misses, seq_arrs))


def ascend(fv, t0, lt, seq_arrs):
    """Line-searched 2-D gradient ascent; a step is accepted only if it improves final v inf AND every
    encounter stays GN-closed sub-SOI. Returns (t0, lt, value, path)."""
    g_fn = jax.jit(jax.grad(lambda a, b: fv(a, b)[0], argnums=(0, 1)))
    v, m = fv(t0, lt)
    v = float(v)
    path = [(0.0, 0.0, v)]
    for _ in range(MAX_STEPS):
        g = np.array(g_fn(jnp.float64(t0), jnp.float64(lt)))
        gn = np.linalg.norm(g)
        if not np.isfinite(gn) or gn == 0.0:
            break
        d = g / gn
        best = None
        for s in STEP_SIZES:
            v_try, m_try = fv(jnp.float64(t0 + s * d[0]), jnp.float64(lt + s * d[1]))
            if float(v_try) > v and closed(m_try, seq_arrs):
                best = (s, float(v_try))
                break
        if best is None:
            break
        s, v = best
        t0, lt = t0 + s * d[0], lt + s * d[1]
        path.append((float(t0), float(lt), v))
    return float(t0), float(lt), v, path


def verify(args):
    print("=== R-N41: does end-to-end GRADIENT POLISH improve the full searched pump chain? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus"):
        C._tab(p)
    t0_ref = sjd + 400.0
    print("  R-N36/R-N37 architecture verbatim; ONE knob: an outer differentiable polish loop — 2-D")
    print("  line-searched gradient ascent on (launch epoch, launch tof), frozen basin inits, a step")
    print("  accepted ONLY if it improves final v inf AND every encounter re-closes sub-SOI.\n")

    print("  [rebuild the searched chain]", flush=True)
    best, _, root = B.run_search(t0_ref, 1)
    if root is None:
        print("  no chain — aborting.")
        return
    lt0 = float(root["launch_tof"])
    v_search = float(best["mag"])
    u_l0 = C.close_launch(t0_ref, lt0, "venus")[0]
    u_f0s = [jnp.asarray(lg["u"]) for lg in best["legs"]]
    seq_arrs = ["venus"] + [lg["to"] for lg in best["legs"]]
    seq_pairs = list(zip(seq_arrs[:-1], seq_arrs[1:]))
    print(f"  chain venus->{'-'.join(seq_arrs[1:])}, final v inf {v_search:.4f}, launch tof {lt0:.0f} d")

    fv = jax.jit(make_chain_fn(u_l0, u_f0s, seq_pairs))
    v0, m0 = fv(jnp.float64(t0_ref), jnp.float64(lt0))
    v0 = float(v0)
    assert abs(v0 - v_search) < 1e-6, f"scan-GN chain does not reproduce the searched tour: {v0} vs {v_search}"
    print(f"  frozen-init scan-GN chain reproduces the searched tour: {v0:.4f} (diff {abs(v0 - v_search):.1e})\n")

    # ---- H-N41a: gradient vs central finite differences ----
    g = jax.jit(jax.grad(lambda a, b: fv(a, b)[0], argnums=(0, 1)))(jnp.float64(t0_ref), jnp.float64(lt0))
    g = (float(g[0]), float(g[1]))
    fd = []
    for k, (dt, dl) in enumerate(((FD_H, 0.0), (0.0, FD_H))):
        vp = float(fv(jnp.float64(t0_ref + dt), jnp.float64(lt0 + dl))[0])
        vm = float(fv(jnp.float64(t0_ref - dt), jnp.float64(lt0 - dl))[0])
        fd.append((vp - vm) / (2 * FD_H))
    finite = bool(np.isfinite(g).all())
    signs = [bool(np.sign(g[k]) == np.sign(fd[k])) for k in range(2)]
    a_ok = finite and all(signs)
    print(f"  grad d(v_f)/d(t0, ltof) = ({g[0]:+.5f}, {g[1]:+.5f}) km/s per day (finite={finite})")
    print(f"  central FD (h={FD_H} d)  = ({fd[0]:+.5f}, {fd[1]:+.5f})  sign match: t0={signs[0]}, ltof={signs[1]}\n")

    # ---- H-N41b: the polish itself ----
    print("  [line-searched gradient ascent]", flush=True)
    t0_p, lt_p, v_p, path = ascend(fv, t0_ref, lt0, seq_arrs)
    for k, (a, b_, v) in enumerate(path[1:], 1):
        print(f"    step {k:2d}: t0 {a - t0_ref:+7.2f} d  ltof {b_ - lt0:+6.2f} d  v inf {v:.4f}")
    gain = v_p - v0
    b_ok = gain >= GAIN_BAR
    print(f"  polished: v inf {v0:.4f} -> {v_p:.4f} (gain {gain:+.4f} km/s) after {len(path) - 1} accepted steps,"
          f" moved (Δt0 {t0_p - t0_ref:+.2f} d, Δltof {lt_p - lt0:+.2f} d)")

    # launch-v inf disclosure (the R-N39 lesson: an uncapped launch can BUY v inf) — the launch vmag is a
    # GN unknown, so the polish may partly enlarge the launch; report the pumped gain net of that
    seed_s = C.close_launch(t0_ref, lt0, "venus")[2]
    seed_p = C.close_launch(t0_p, lt_p, "venus")[2]
    net_gain = (v_p - seed_p) - (v0 - seed_s)
    print(f"  launch disclosure: seed v inf {seed_s:.4f} -> {seed_p:.4f} (+{seed_p - seed_s:.4f}); pumped gain "
          f"net of launch {net_gain:+.4f} km/s ({v0 - seed_s:.3f} -> {v_p - seed_p:.3f})\n")

    # ---- H-N41c: independent re-closure at the polished point (the 40-iteration solver) ----
    def res_launch(u):
        miss, _ = C.shoot("earth", "venus", jnp.float64(t0_p), u[2] * C.unit_dir(u[0], u[1]), jnp.float64(lt_p))
        return miss / 1e6
    u_l40 = gn_scan(res_launch, u_l0, 40, step_max=(0.4, 0.2, 1.5))
    _, va_l = C.shoot("earth", "venus", jnp.float64(t0_p),
                      u_l40[2] * C.unit_dir(u_l40[0], u_l40[1]), jnp.float64(lt_p))
    vin, jd = va_l, t0_p + lt_p
    miss_l40 = float(jnp.linalg.norm(res_launch(u_l40))) * 1e6
    legs_ok, worst = [miss_l40 < C.SOI_KM["venus"]], miss_l40
    for (dep, arr), u0 in zip(seq_pairs, u_f0s):
        scan_miss, gn40, leg_out = B.pair(dep, arr)
        u, miss = gn40(u0, vin, jnp.float64(jd))
        va, turn, dm = leg_out(u, vin, jnp.float64(jd))
        ok = float(miss) < C.SOI_KM[arr] and abs(float(turn)) <= float(dm) + 1e-12
        legs_ok.append(ok)
        worst = max(worst, float(miss))
        vin, jd = va, jd + float(u[2])
    c_ok = all(legs_ok)
    v_c = float(jnp.linalg.norm(vin))
    print(f"  independent 40-iter re-closure at the polished point: all legs closed={c_ok} "
          f"(worst miss {worst:.1e} km), final v inf {v_c:.4f}\n")

    # ---- verdicts vs pre-registered REFUTE-BYs ----
    print(f"  → H-N41a {'SUPPORTED' if a_ok else 'REFUTED'}: the full-depth gradient "
          f"{'FLOWS' if a_ok else 'fails'} — finite={finite}, FD sign match {signs}.")
    print(f"  → H-N41b {'SUPPORTED' if b_ok else 'REFUTED'}: polish "
          f"{'PAYS' if b_ok else 'does NOT pay'} — gain {gain:+.4f} km/s "
          f"{'≥' if b_ok else '<'} the pre-registered {GAIN_BAR} bar"
          + ("." if b_ok else " — the searched chain already sits at a continuous local optimum within its"
         " basin; the basin choice, not the continuous freedom, determines the outcome."))
    print(f"  → H-N41c {'SUPPORTED' if c_ok else 'REFUTED'}: the polished chain "
          f"{'re-closes' if c_ok else 'FAILS to re-close'} under the independent solver "
          f"(worst miss {worst:.1e} km; |v inf| conserved by construction).")

    print(f"\n  → verdicts: H-N41a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N41b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N41c {'SUPPORTED' if c_ok else 'REFUTED'}")
    if b_ok:
        print("  NET: end-to-end gradient polish IMPROVES the full searched chain — the R-N36 'search chooses,")
        print(f"    gradients polish' split completes at depth: +{gain:.3f} km/s of final v inf (of which "
              f"{net_gain:+.3f} is")
        print(f"    genuinely PUMPED, {seed_p - seed_s:+.3f} from a modestly larger solved launch — disclosed) from")
        print("    walking the continuous launch freedom inside the frozen basins, every encounter still exactly")
        print("    ballistic.")
    else:
        print("  NET: the searched chain is already at (or within {:.0f} m/s of) a continuous local optimum —".format(gain * 1000))
        print("    the coarse basin search leaves almost nothing on the table for gradients at this epoch. That")
        print("    is a REAL structural finding, judged against the pre-registered falsifier: in this architecture")
        print("    the discrete basin choice determines the outcome, and the continuous freedom is nearly")
        print("    exhausted by GN closure itself. (Probe-consistent: the fine landscape is smooth but flat.)")
    print("    Scope: one epoch, pump objective only (crank-inclination polish is future work), frozen basin")
    print("    inits (gradients cannot hop basins, R-N7), steps bounded by every-leg sub-SOI closure.")
    print("    Mechanism/DISCOVERY study, never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
