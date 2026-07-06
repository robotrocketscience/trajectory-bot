#!/usr/bin/env python3
"""Plane-change strategy in the averaged model (Build E) — does diff-sim rediscover
raise-to-plane-change for large Δi?

Edelbaum is the PROVABLE optimum for the averaged combined transfer, so there's no
"beat" here. But the closed form for a large PURE plane change (Vf = V0) hides a
strategy: plane changes are cheaper where V is small (di/dt ∝ f/V), so the optimum
DIPS in altitude (V drops below V0) to turn the plane cheaply, then returns — the
continuous analog of the 3-impulse bi-elliptic-inclination trick. A naive
constant-altitude plane change (β=90°) costs (π/2)·V0·Δi; the Edelbaum optimum
2·V0·sin(π/4·Δi) is strictly less for large Δi, and the gap IS that altitude dip.

Test: train a diff-sim yaw+throttle law on pure plane changes and check it (a)
matches 2·V0·sin(π/4·Δi) (beating the naive constant-V law for large Δi) and (b)
its V-trajectory DIPS — i.e. it rediscovers raise-to-plane-change. Reuses
edelbaum_sim's averaged dynamics.

    uv run --with jax python scripts/planechange_sim.py --train --eval
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

sys.path.insert(0, "scripts")
import edelbaum_sim as E  # noqa: E402

R_E = E.R_E


def sample_pc(key, B, di_lo=5.0, di_hi=60.0):
    """Pure plane change: Vf = V0 (same altitude), i0 large, target equatorial."""
    k = random.split(key, 3)
    r0 = R_E + (300.0 + 700.0 * random.uniform(k[0], (B,)))
    V0 = jnp.sqrt(E.MU_EARTH / r0)
    Vf = V0                                                 # same altitude
    i0 = np.radians(di_lo) + (np.radians(di_hi) - np.radians(di_lo)) * random.uniform(k[1], (B,))
    return V0, Vf, i0, jnp.zeros(B)


def naive_constant_v(V0, i0, i_tgt):
    """(π/2)·V0·Δi — the constant-altitude (β=90°) plane change (no altitude dip)."""
    dI = np.abs(np.asarray(i0) - np.asarray(i_tgt))
    return 0.5 * np.pi * np.asarray(V0) * dI


# FULL yaw β ∈ [-π, π] (edelbaum_sim caps at ±π/2 → cos β ≥ 0, altitude can only
# RISE). The return leg of an up-then-down excursion needs cos β < 0 (retro in-plane
# thrust to come back down to V0), so a pure plane change must use the full range.
def _integrate(params, V0, Vf, i0, i_tgt, n, record):
    B = V0.shape[0]

    def step(carry, _):
        V, i, dv = carry
        act = E.policy(params, E.observe(V, i, V0, Vf, i_tgt))
        beta = np.pi * jnp.tanh(act[:, 0])                 # full yaw
        thr = 0.5 * (1.0 + jnp.tanh(act[:, 1]))
        dvs = thr * E.F_THRUST * E.DT
        V = jnp.clip(V - dvs * jnp.cos(beta), 0.5, None)   # cos β can be < 0 → V rises again
        i = i + thr * E.TWO_PI * (E.F_THRUST / V) * jnp.sin(beta) * E.DT
        dv = dv + dvs
        return (V, i, dv), (V if record else None)
    (V, i, dv), Vtraj = lax.scan(step, (V0, i0, jnp.zeros(B)), None, length=n)
    return (V, i, dv, Vtraj) if record else (V, i, dv)


def integrate_pc(params, V0, Vf, i0, i_tgt, n=E.N_STEPS):
    return _integrate(params, V0, Vf, i0, i_tgt, n, record=False)


def integrate_record(params, V0, Vf, i0, i_tgt, n=E.N_STEPS):
    return _integrate(params, V0, Vf, i0, i_tgt, n, record=True)   # (V, i, dv, Vtraj)


def make_loss_pc(w_v=2000.0, w_i=2000.0):
    def loss(params, V0, Vf, i0, i_tgt):
        V, i, dv = integrate_pc(params, V0, Vf, i0, i_tgt)
        return dv.mean() + w_v * (((V - Vf) / Vf) ** 2).mean() + w_i * ((i - i_tgt) ** 2).mean()
    return loss


def train(args):
    params = E.init_params(random.PRNGKey(1))
    loss_fn = make_loss_pc()         # full-β dv + w_v*(V-Vf)^2 + w_i*(i-i_tgt)^2
    vg = jax.jit(jax.value_and_grad(loss_fn))
    opt = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), 0)

    @jax.jit
    def step(params, opt, V0, Vf, i0, it, lr):
        l, g = vg(params, V0, Vf, i0, it)
        gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        g = tree_map(lambda x: x * jnp.minimum(1.0, 10.0 / jnp.maximum(gn, 1e-9)), g)
        m, v, t = opt; t = t + 1
        m = tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g)
        v = tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, g)
        mh = tree_map(lambda m_: m_ / (1 - 0.9 ** t), m)
        vh = tree_map(lambda v_: v_ / (1 - 0.999 ** t), v)
        params = tree_map(lambda p_, m_, v_: p_ - lr * m_ / (jnp.sqrt(v_) + 1e-8), params, mh, vh)
        return params, (m, v, t), l

    key = random.PRNGKey(args.seed)
    for it in range(args.iters):
        V0, Vf, i0, itg = sample_pc(random.fold_in(key, it), args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))
        params, opt, l = step(params, opt, V0, Vf, i0, itg, jnp.float32(lr))
        if (it + 1) % args.eval_every == 0 or it == 0:
            print(f"it={it:4d} loss={float(l):.4f}", flush=True)
    E.save(args.save, params)
    print(f"saved {args.save}", flush=True)
    evaluate(params)


def evaluate(params, seed=31337, episodes=4096):
    V0, Vf, i0, itg = sample_pc(random.PRNGKey(seed), episodes)
    V, i, dv, Vtraj = integrate_record(params, V0, Vf, i0, itg)
    V = np.asarray(V); i = np.asarray(i); dv = np.asarray(dv); Vtraj = np.asarray(Vtraj)
    V0n = np.asarray(V0); i0n = np.asarray(i0)
    edel = E.edelbaum_ref(V0, Vf, i0, itg)                  # 2 V0 sin(pi/4 dI)
    naive = naive_constant_v(V0, i0, itg)                   # (pi/2) V0 dI
    v_err = np.abs(V - V0n) / V0n                           # must return to V0
    i_err = np.degrees(np.abs(i))
    ok = (v_err < 0.02) & (i_err < 1.0)
    Vdip = 1.0 - Vtraj.min(axis=0) / V0n                    # fractional altitude dip (V below V0)
    print(f"pure-plane-change eval ({episodes} tasks, Δi {np.degrees(i0n.min()):.0f}-"
          f"{np.degrees(i0n.max()):.0f}°):")
    print(f"  returned to V0 & reached i_tgt = {100*ok.mean():.1f}%")
    if ok.any():
        print(f"  median Δv/edelbaum      = {np.median(dv[ok]/edel[ok]):.3f}  (1.0 = optimum)")
        print(f"  median Δv/naive-const-V = {np.median(dv[ok]/naive[ok]):.3f}  "
              f"(<1 = beats the constant-altitude plane change)")
        print(f"  median mid-maneuver V-dip = {100*np.median(Vdip[ok]):.1f}% below V0 "
              f"(>0 = raised altitude to plane-change = the rediscovered strategy)")
    # large-Δi slice: where the dip should be biggest
    big = ok & (np.degrees(i0n) > 35.0)
    if big.any():
        print(f"  Δi>35° slice: Δv/edelbaum={np.median(dv[big]/edel[big]):.3f} "
              f"Δv/naive={np.median(dv[big]/naive[big]):.3f} "
              f"V-dip={100*np.median(Vdip[big]):.1f}%  (n={big.sum()})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--init", type=str, default="models/planechange.npz")
    ap.add_argument("--save", type=str, default="models/planechange.npz")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.train:
        train(args)
    if args.eval and not args.train:
        evaluate(E.load(args.init))


if __name__ == "__main__":
    main()
