#!/usr/bin/env python3
"""Multiple-shooting (Sims-Flanagan) low-thrust Earth→Mars on real ephemerides (Build N, R-N5).

R-N4 showed naive SINGLE-shooting cannot economize a long-arc low-thrust transfer: the 250-day
terminal-miss landscape is too stiff for plain Adam. The fix is a transcription that makes every
propagated arc SHORT. Split the TOF into M segments; the segment NODE states x_k=(r,v) are decision
variables; each segment applies a bounded impulse u_k (Sims-Flanagan: |u_k| ≤ a_max·h, one segment's
worth of continuous Δv) then coasts under the N-body field for h; enforce CONTINUITY defects
d_k = propagate(x_k + u_k) − x_{k+1} = 0. Each defect spans only h = TOF/M, so its gradient is
well-conditioned; the optimizer places the intermediate states directly and minimizes the total
Δv = Σ|u_k| under defect + boundary penalties. Warm-started on the impulsive Lambert arc (defects
start ~0). Impulsive Lambert Δv is the FLOOR a low-thrust arc cannot beat.

    uv run --with jax python scripts/nbody_collocation.py --verify         # Sun-only, offline
    uv run --with jax python scripts/nbody_collocation.py --fetch --mission mro   # real Sun+Jupiter
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM               # noqa: E402
import nbody_sim as NB               # noqa: E402
import nbody_transfer as NT          # noqa: E402

MU_SUN = NB.GM["sun"]
AU = NB.AU
DAY = NB.DAY
SOI_MARS = 5.77e5
# canonical units so the node decision variables are O(1) (essential for Adam: its per-parameter
# step is ~lr regardless of scale, so raw km / km·s⁻¹ nodes at O(1e8)/O(30) never move enough).
LU = AU                                  # length unit (km)
VU = float(np.sqrt(MU_SUN / AU))         # velocity unit ≈ 29.78 km/s
SCALE = jnp.array([LU, LU, LU, VU, VU, VU])


# ---------- segment propagation (batched over segments) ----------
def propagate_segments(seg_start, impulses, body_seg, gm, dt_sub, soft):
    """seg_start (M,6) segment-start states; impulses (M,3) applied to velocity; body_seg
    (M,n_sub,K,3) held-per-substep body positions. Returns end-of-segment states (M,6)."""
    def one(x0, u, bseq):
        rv = x0.at[3:].add(u)
        rvT, _ = NB.rollout(rv, bseq, gm, dt_sub, soft=soft)
        return rvT
    return jax.vmap(one)(seg_start, impulses, body_seg)


def bound_impulses(raw, cap):
    """Magnitude-bounded impulses (M,3), |u_k| ≤ cap. eps inside the sqrt keeps the gradient
    finite at raw=0 (the R-N4 NaN lesson)."""
    mag = jnp.sqrt(jnp.sum(raw ** 2, axis=1, keepdims=True) + 1e-12)
    return cap * jnp.tanh(mag) * raw / mag


def objective(params, x0_fixed, r2, body_seg, gm, dt_sub, cap, w_def, w_bc, soft):
    nodes_s, raw = params                        # nodes_s: (M,6) SCALED = x_1..x_M ; raw: (M,3)
    nodes = nodes_s * SCALE                       # → physical km, km/s
    u = bound_impulses(raw, cap)
    seg_start = jnp.concatenate([x0_fixed[None], nodes[:-1]], axis=0)   # (M,6) = x_0..x_{M-1}
    prop = propagate_segments(seg_start, u, body_seg, gm, dt_sub, soft)  # (M,6)
    defect = (prop - nodes) / SCALE                # continuity defect in canonical units (O(1))
    # eps inside the sqrt — a bare norm of a zero impulse has a NaN gradient (R-N4 lesson).
    dv = jnp.sum(jnp.sqrt(jnp.sum(u ** 2, axis=1) + 1e-12))    # total Δv (km/s)
    def_pen = jnp.sum(defect ** 2)
    bc = jnp.sum(((nodes[-1, :3] - r2) / LU) ** 2)             # terminal position → Mars (canonical)
    L = dv + w_def * def_pen + w_bc * bc
    max_def_km = jnp.max(jnp.linalg.norm((prop - nodes)[:, :3], axis=1))
    term_miss = jnp.linalg.norm(nodes[-1, :3] - r2)
    return L, (dv, max_def_km, term_miss)


def lambert_arc_nodes(r1, v1_L, body_full, gm, dt_sub, n_sub, M, soft):
    """Propagate the impulsive Lambert arc under the N-body field and sample node states at the
    M segment boundaries → init nodes x_1..x_M (M,6). body_full: (M*n_sub, K, 3)."""
    rvT, traj = NB.rollout(jnp.asarray(np.concatenate([r1, v1_L])), body_full, gm, dt_sub, soft=soft)
    # traj[j] = state after substep j (j=0..M*n_sub-1); node x_k (k=1..M) = state after k*n_sub steps
    idx = np.arange(1, M + 1) * n_sub - 1
    return jnp.asarray(np.asarray(traj)[idx])      # (M,6)


def solve(x0_fixed, r2, v1_L, body_full, gm, dt_sub, n_sub, M, a_max, h,
          iters=4000, lr=1e-2, soft=1.0):
    """Multiple-shooting solve. Returns (dv, max_defect_km, term_miss_km)."""
    body_seg = body_full.reshape(M, n_sub, body_full.shape[1], 3)
    cap = a_max * h
    nodes0 = lambert_arc_nodes(np.asarray(x0_fixed[:3]), v1_L, body_full, gm, dt_sub, n_sub, M, soft)
    # departure impulse u_0 = Lambert Δv (clipped to cap by tanh); rest ~0
    dv_dep = np.asarray(v1_L) - np.asarray(x0_fixed[3:])
    raw0 = np.zeros((M, 3))
    frac = min(0.98, float(np.linalg.norm(dv_dep)) / cap)
    raw0[0] = np.arctanh(frac) * dv_dep / (np.linalg.norm(dv_dep) + 1e-12)
    params = (nodes0 / SCALE, jnp.asarray(raw0))    # nodes optimized in canonical units

    vg = jax.jit(jax.value_and_grad(
        lambda p, wd: objective(p, x0_fixed, r2, body_seg, gm, dt_sub, cap, wd, wd, soft),
        has_aux=True))
    m = [jnp.zeros_like(p) for p in params]
    v = [jnp.zeros_like(p) for p in params]
    b1, b2, eps = 0.9, 0.999, 1e-12
    best = (float("inf"), None)
    t = 0
    for stage, wd in enumerate((3e2, 3e3, 3e4, 3e5)):      # anneal the defect/boundary weight up
        for _ in range(iters // 4):
            t += 1
            (L, (dv, mdef, miss)), g = vg(params, wd)
            # feasibility-aware best: smallest Δv among iterates with defects closed to ≪ SOI & reached
            if float(mdef) < 0.05 * SOI_MARS and float(miss) < 0.1 * SOI_MARS and float(dv) < best[0]:
                best = (float(dv), (float(dv), float(mdef), float(miss)))
            params = list(params)
            for i in (0, 1):
                m[i] = b1 * m[i] + (1 - b1) * g[i]
                v[i] = b2 * v[i] + (1 - b2) * (g[i] * g[i])
                params[i] = params[i] - lr * (m[i] / (1 - b1 ** t)) / (jnp.sqrt(v[i] / (1 - b2 ** t)) + eps)
            params = tuple(params)
    if best[1] is not None:
        return best[1]
    (L, (dv, mdef, miss)), _ = vg(params, 3e5)
    return float(dv), float(mdef), float(miss)


# ---------- offline Sun-only verification (CI-safe) ----------
def verify(args):
    print("=== R-N5: multiple-shooting low-thrust — offline Sun-only (H-N5a/b) ===")
    mu = MU_SUN
    r_e, r_m = 1.0 * AU, 1.5237 * AU
    v_ce = np.sqrt(mu / r_e)
    r1 = np.array([r_e, 0.0, 0.0])
    v_earth = np.array([0.0, v_ce, 0.0])
    sweep = np.radians(150.0)
    r2 = r_m * np.array([np.cos(sweep), np.sin(sweep), 0.0])
    tof = args.tof * DAY
    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=mu)
    v1_L = np.asarray(v1_L)
    dv_imp = float(np.linalg.norm(v1_L - v_earth))
    M, n_sub = args.segments, args.substeps
    h = tof / M
    dt_sub = h / n_sub
    a_mean = dv_imp / tof
    body_full = jnp.zeros((M * n_sub, 1, 3))
    gm = jnp.array([mu])
    x0 = jnp.asarray(np.concatenate([r1, v_earth]))
    print(f"  endpoints: 1.000→1.524 AU @150° sweep, TOF={args.tof:.0f} d, M={M} segments; "
          f"impulsive Lambert Δv={dv_imp:.4f} km/s (FLOOR); mean-accel={a_mean:.2e} km/s²")
    print(f"  {'a_max/mean':>10} {'a_max(km/s²)':>13} {'Δv(km/s)':>9} {'dvr':>7} "
          f"{'max_defect(km)':>14} {'miss(km)':>10}")
    for k in (10.0, 5.0, 3.0, 2.0):
        a_max = k * a_mean
        dv, mdef, miss = solve(x0, r2, v1_L, body_full, gm, dt_sub, n_sub, M, a_max, h,
                               iters=args.iters, soft=0.0)
        print(f"  {k:10.0f} {a_max:13.2e} {dv:9.4f} {dv/dv_imp:7.3f} {mdef:14.0f} {miss:10.0f}")
    print("  H-N5a: defects close (max_defect ≪ SOI) & reaches at a_max where single-shooting failed. "
          "H-N5b: dvr ≥ 1, → ~1 at high a_max, rising smoothly as a_max drops (TRUSTWORTHY, unlike R-N4).")


# ---------- real ephemerides ----------
def fetch(args):
    from scripts.missions import MISSIONS
    from astropy.time import Time
    m = MISSIONS[args.mission]
    launch = m.launch + "T00:00:00"
    arrival = m.arrival + "T00:00:00"
    tof = float((Time(arrival) - Time(launch)).sec)
    tof_days = tof / DAY
    r1, v_earth = NT.body_state("earth", launch)
    r2, _ = NT.body_state("mars", arrival)
    v1_L, _ = LAM.lambert(jnp.asarray(r1), jnp.asarray(r2), tof, mu=MU_SUN)
    v1_L = np.asarray(v1_L)
    dv_imp = float(np.linalg.norm(v1_L - v_earth))
    M, n_sub = args.segments, args.substeps
    h = tof / M
    dt_sub = h / n_sub
    a_mean = dv_imp / tof
    perturbers = ["sun", "jupiter"]
    gm = jnp.array([NB.GM[b] for b in perturbers])
    # sample bodies at every substep (held-per-substep): M*n_sub rows
    body_np, _ = NT.sample_fine(perturbers, launch, tof_days, dt_sub)   # (2N+1, K, 3) at half-steps
    body_full = jnp.asarray(body_np[0:2 * M * n_sub:2])                 # (M*n_sub, K, 3) step-starts
    x0 = jnp.asarray(np.concatenate([r1, v_earth]))
    print(f"=== R-N5 real ephemerides: {m.name} (earth→mars) multiple-shooting low-thrust ===")
    print(f"  launch {m.launch}, arrival {m.arrival}, TOF={tof_days:.1f} d, M={M}; "
          f"impulsive Lambert Δv={dv_imp:.4f} km/s (FLOOR); Sun+Jupiter")
    print(f"  {'a_max/mean':>10} {'a_max(km/s²)':>13} {'Δv(km/s)':>9} {'dvr':>7} "
          f"{'max_defect(km)':>14} {'miss(km)':>10}")
    for k in (10.0, 5.0, 3.0, 2.0):
        a_max = k * a_mean
        dv, mdef, miss = solve(x0, r2, v1_L, body_full, gm, dt_sub, n_sub, M, a_max, h,
                               iters=args.iters, soft=10.0)
        print(f"  {k:10.0f} {a_max:13.2e} {dv:9.4f} {dv/dv_imp:7.3f} {mdef:14.0f} {miss:10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--mission", type=str, default="mro")
    ap.add_argument("--tof", type=float, default=250.0)
    ap.add_argument("--segments", type=int, default=24)
    ap.add_argument("--substeps", type=int, default=15)
    ap.add_argument("--iters", type=int, default=4000)
    args = ap.parse_args()
    if args.verify:
        verify(args)
    if args.fetch:
        fetch(args)


if __name__ == "__main__":
    main()
