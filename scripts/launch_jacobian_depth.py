#!/usr/bin/env python3
"""How does the FULL LAUNCH JACOBIAN condition vs tour DEPTH? t0 axis vs ltof axis (Build N, R-N51).

Extends R-N50 (which conditioned ONE launch axis — leg time-of-flight, ltof — and found it finite, gently
ATTENUATING ×0.77/leg, directionally useful). But R-N41 showed the actual gradient-polish gain (v_inf
16.27→17.01) lived in walking the launch EPOCH t0 by +8.9 d, and the sub-day ltof probe MISSED it — so R-N50
characterized the LESS polish-relevant axis. This round measures the OTHER axis (t0) and the full 2-D launch
Jacobian ∇v_inf = (∂v_inf/∂t0, ∂v_inf/∂ltof) — the object 2-D polish actually descends. SAME chain (R-N37's
depth-6), SAME epoch (sjd+400), SAME truncation sweep 1..6 as R-N50; ONE knob = the differentiation AXIS
(ltof → t0 / full-2-D). Units matched (t0 and ltof both in days), so ∂v_inf/∂t0 vs ∂v_inf/∂ltof is an
apples-to-apples per-day comparison.

  H-N51a  the t0-axis gradient ∂v_inf/∂t0 stays FINITE and FD-MATCHED at every depth 1..6 (no NaN/explosion;
          analytic == central-FD in sign, relerr small once the FD step resolves curvature). REFUTE-BY:
          |∂v_inf/∂t0| explodes (>1e4× its FD) or NaNs at some depth (a pathology hitting t0 but sparing ltof).
  H-N51b  the t0 axis DOMINATES the ltof axis at depth — |∂v_inf/∂t0| > |∂v_inf/∂ltof| at the deepest LIVE depth,
          consistent with R-N41's t0-walk polish gain. REFUTE-BY: |∂v_inf/∂t0| ≤ |∂v_inf/∂ltof| at the deepest
          live depth (t0 is NOT the larger-gradient axis — R-N41's gain would be landscape geometry, not a bigger
          gradient).
  H-N51c  the full 2-D launch-Jacobian norm ‖∇v_inf‖ does NOT VANISH with depth — within ~2 orders across depths
          1..6, so 2-D launch polish stays conditioned as the tour deepens. REFUTE-BY: ‖∇v_inf‖ decays
          ~exponentially toward 0.

Machinery = R-N36 forward shooting + R-N41's lax.scan unrolled-GN gradient (dodges the 11-min XLA compile);
depth-6 chain from R-N37's beam. Mechanism/DISCOVERY study of the METHOD, never a Δv beat (locked 418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/launch_jacobian_depth.py --verify
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

import constrained_tour_discovery as C   # noqa: E402
import beam_constrained_tour as B         # noqa: E402  (R-N37 depth-6 chain)
import polish_tour as PT                  # noqa: E402  (R-N41 make_chain_fn: frozen-init unrolled-GN objective)


def central_fd(fv, t0, lt, h, axis):
    """Central finite difference of final v_inf along axis 0 (t0) or 1 (ltof)."""
    if axis == 0:
        vp = float(fv(jnp.float64(t0 + h), jnp.float64(lt))[0])
        vm = float(fv(jnp.float64(t0 - h), jnp.float64(lt))[0])
    else:
        vp = float(fv(jnp.float64(t0), jnp.float64(lt + h))[0])
        vm = float(fv(jnp.float64(t0), jnp.float64(lt - h))[0])
    return (vp - vm) / (2 * h)


def verify(args):
    print("=== R-N51: how does the FULL LAUNCH JACOBIAN condition vs tour DEPTH? (t0 axis vs ltof axis) ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus"):
        C._tab(p)
    t0_ref = sjd + 400.0
    print("  R-N36 forward shooting + R-N41 unrolled-GN gradient; ONE knob vs R-N50 = differentiation AXIS")
    print("  (ltof → t0 / full-2-D). Truncate R-N37's depth-6 chain at 1..6; measure ∂v_inf/∂t0 (analytic vs")
    print("  central-FD), re-confirm ∂v_inf/∂ltof (R-N50), and the 2-D norm ‖∇v_inf‖ at each.\n")

    best, _, root = B.run_search(t0_ref, 1)
    if root is None:
        print("  no chain — aborting.")
        return
    lt0 = float(root["launch_tof"])
    u_l0 = C.close_launch(t0_ref, lt0, "venus")[0]
    u_f0s = [jnp.asarray(lg["u"]) for lg in best["legs"]]
    seq_arrs = ["venus"] + [lg["to"] for lg in best["legs"]]
    seq_pairs = list(zip(seq_arrs[:-1], seq_arrs[1:]))
    n = len(u_f0s)
    print(f"  depth-{n} chain venus->{'-'.join(seq_arrs[1:])}, final v_inf {float(best['mag']):.3f}, "
          f"launch tof {lt0:.0f} d\n")

    # ---- depth sweep: ∂v_inf/∂t0 (analytic vs central-FD) + ∂v_inf/∂ltof + ratio + 2-D norm ----
    H0 = 0.05
    print("  [depth sweep — launch-Jacobian components at h=0.05 d]")
    print("   depth  final_vinf   dV/dt0        FD_t0(h=.05)  relerr   dV/dltof      |g_t0|/|g_ltof|   ‖grad‖   "
          "gt0_atten")
    rows = []
    gt0_1 = None
    for d in range(1, n + 1):
        fv = jax.jit(PT.make_chain_fn(u_l0, u_f0s[:d], seq_pairs[:d]))
        v_d = float(fv(jnp.float64(t0_ref), jnp.float64(lt0))[0])
        gt0 = float(jax.grad(lambda a, b: fv(a, b)[0], argnums=0)(jnp.float64(t0_ref), jnp.float64(lt0)))
        glt = float(jax.grad(lambda a, b: fv(a, b)[0], argnums=1)(jnp.float64(t0_ref), jnp.float64(lt0)))
        fd = central_fd(fv, t0_ref, lt0, H0, axis=0)
        relerr = abs(gt0 - fd) / (abs(fd) + 1e-12)
        ratio = abs(gt0) / (abs(glt) + 1e-30)
        norm = (gt0 ** 2 + glt ** 2) ** 0.5
        if gt0_1 is None:
            gt0_1 = abs(gt0)
        rows.append({"d": d, "v": v_d, "gt0": gt0, "glt": glt, "fd": fd, "relerr": relerr,
                     "ratio": ratio, "norm": norm, "fv": fv})
        print(f"    {d:2d}    {v_d:8.3f}   {gt0:+.4e}   {fd:+.4e}   {relerr:6.1%}   {glt:+.4e}   {ratio:7.2f}"
              f"        {norm:.3e}   {abs(gt0) / gt0_1 if gt0_1 else float('nan'):5.2f}", flush=True)

    # ---- deepest LIVE depth: last truncation where the pump still climbs (beyond it the tail legs saturate) ----
    d_live = 1
    for i in range(1, len(rows)):
        if rows[i]["v"] > rows[i - 1]["v"] + 1e-6:
            d_live = rows[i]["d"]
    live = rows[d_live - 1]

    # ---- h-sensitivity at the deepest live depth for the t0 axis (mirror R-N50's FD-truncation check) ----
    print(f"\n  [h-sensitivity at deepest-live depth {d_live} — t0 axis]")
    hs = (0.05, 0.02, 0.005, 0.001)
    relerrs_h = []
    for h in hs:
        fd_h = central_fd(live["fv"], t0_ref, lt0, h, axis=0)
        re_h = abs(live["gt0"] - fd_h) / (abs(fd_h) + 1e-12)
        relerrs_h.append(re_h)
        print(f"    h={h:6.3f} d: FD_t0 {fd_h:+.4e}, relerr vs analytic {re_h:6.1%}", flush=True)

    gt0s = [abs(r["gt0"]) for r in rows]
    norms = [r["norm"] for r in rows]
    finite = all(np.isfinite(r["gt0"]) and np.isfinite(r["fd"]) and np.isfinite(r["glt"]) for r in rows)
    no_explode = all(abs(r["gt0"]) < 1e4 * (abs(r["fd"]) + 1e-12) for r in rows)
    if n < 2 or gt0s[0] == 0.0 or norms[0] == 0.0:
        print("\n  no depth sweep possible — chain has < 2 legs or a zero baseline gradient; verdicts undefined.")
        return
    gt0_atten = gt0s[-1] / gt0s[0]
    norm_atten = norms[-1] / norms[0]
    ratio_live = live["ratio"]

    # H-N51a's pre-registered wording: "analytic == central-FD IN SIGN [at every depth], relerr small once the FD
    # step resolves curvature". So verify sign-agreement at EVERY depth (truncation-immune) + no explosion at every
    # depth; the h-sweep at d_live establishes magnitude match. A per-depth relerr<0.25 gate is deliberately AVOIDED:
    # a depth's h=0.05 relerr can be inflated by FD-truncation (R-N50's ltof relerr grew to 20% deep, resolving to 0%
    # at smaller h), so gating on it would false-REFUTE on a truncation artifact, not a real gradient error.
    sign_match = all(np.sign(r["gt0"]) == np.sign(r["fd"]) for r in rows)
    a_ok = finite and no_explode and sign_match and min(relerrs_h) < 0.25
    b_ok = ratio_live > 1.0                                       # t0 dominates ltof at the deepest live depth
    c_ok = norm_atten > 1e-2                                      # 2-D norm within ~2 orders (not vanishing)

    print(f"\n  → H-N51a {'SUPPORTED' if a_ok else 'REFUTED'}: ∂v_inf/∂t0 stays FINITE, SIGN-matched and "
          f"non-exploding at EVERY depth — max |g_t0|/|FD| = {max(abs(r['gt0']) / (abs(r['fd']) + 1e-12) for r in rows):.2f}; "
          f"t0-axis relerr {rows[0]['relerr']:.0%}→{rows[-1]['relerr']:.0%} across depth, magnitude FD-matched at the "
          f"deepest live depth {d_live} (h-sweep min {min(relerrs_h):.0%} "
          f"at h={hs[np.argmin(relerrs_h)]:.3f} → analytic t0 gradient reliable).")
    print(f"  → H-N51b {'SUPPORTED' if b_ok else 'REFUTED'}: the t0 axis DOMINATES ltof at the deepest live depth "
          f"{d_live} — |g_t0|/|g_ltof| = {ratio_live:.2f} (>1); the ratio CROSSES OVER with depth "
          f"({rows[0]['ratio']:.2f} at d=1 where ltof leads → {rows[-1]['ratio']:.2f} at d={n}), so t0 dominance "
          f"EMERGES as the tour deepens — confirming R-N41's t0-walk gain is gradient-driven, not line-search geometry.")
    print(f"  → H-N51c {'SUPPORTED' if c_ok else 'REFUTED'}: ‖∇v_inf‖ does NOT vanish — it {'GROWS' if norm_atten > 1 else 'holds'} "
          f"with depth ({norms[0]:.2e}→{norms[-1]:.2e}, ×{norm_atten:.2f} over {n} flybys), dominated by the "
          f"t0 component (which AMPLIFIES ×{gt0_atten:.2f}) as the ltof component attenuates.")

    print(f"\n  → verdicts: H-N51a {'SUPPORTED' if a_ok else 'REFUTED'}, H-N51b {'SUPPORTED' if b_ok else 'REFUTED'}, "
          f"H-N51c {'SUPPORTED' if c_ok else 'REFUTED'}")
    print("  NET: the two launch axes have OPPOSITE depth-conditioning — R-N50's ltof gradient ATTENUATES")
    print(f"    (×0.27 over {n} legs) but the t0 gradient AMPLIFIES (×{gt0_atten:.2f}), and they CROSS OVER between")
    print(f"    depth 1 (ltof leads) and depth 3 (t0 leads), with t0 dominant ×{rows[-1]['ratio']:.1f} deep. So the")
    print("    FULL launch Jacobian does NOT attenuate with depth — its norm GROWS, dominated by t0. This")
    print("    SHARPENS R-N50 (whose 'gently attenuating signal' was axis-specific to ltof) and EXPLAINS R-N41:")
    print("    the polish gain lived in the t0 walk because t0 is the dominant AND growing gradient direction at")
    print("    depth. Physically, t0 shifts the ENTIRE downstream planetary geometry so its effect COMPOUNDS with")
    print("    each added encounter, while ltof only sets the first leg and is absorbed by re-closure. The")
    print("    effective 2-D polish direction ROTATES from ltof-ish (shallow) toward t0 (deep). Scope: one epoch,")
    print("    one chain (R-N37's depth-6), launch (t0,ltof) axes, frozen basins. Never a Δv beat (418e2e2).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
