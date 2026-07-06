#!/usr/bin/env python3
"""Orbit-averaged Edelbaum spiral (Build C) — diff-sim yaw law vs the closed form.

The true SEP LEO->GEO spiral is weeks / hundreds of revs — un-RK4-able. This
collapses each revolution into the secular drift of (V, i) [Edelbaum 1961], so the
whole transfer is a short 2-D time integration, and asks whether a diff-sim
yaw+throttle law recovers the closed-form optimum edelbaum_dv (it is PROVABLY
optimal for this averaged model, so diff-sim can only MATCH — a correctness check
that validates the averaged-dynamics engine).

Averaged EOM (V = sqrt(mu/a) circular speed, i inclination, yaw beta, thrust f):
    dV/dt = -throttle f cos(beta)
    di/dt =  throttle (2/pi)(f/V) sin(beta)
    dv    += throttle f dt
Closed form (tbot.orbital3d.edelbaum_dv): sqrt(V0^2 - 2 V0 Vf cos(pi/2 dI) + Vf^2).
Anchor: LEO(7.73)->GEO(3.07) @ 28.5deg = 5.96 km/s.

    uv run --with jax python scripts/edelbaum_sim.py --sanity
    uv run --with jax python scripts/edelbaum_sim.py --train --eval
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random
from jax.tree_util import tree_map

sys.path.insert(0, ".")
from tbot.orbital import MU_EARTH, speed_circular          # noqa: E402
from tbot.orbital3d import edelbaum_dv                     # noqa: E402

R_E = 6378.137
F_THRUST = 1.0            # normalized thrust accel [km/s per time-unit]; Δv = f*T
DT = 0.05                 # time step [tu]; N*DT*f caps Δv
N_STEPS = 220             # ~11 km/s of Δv budget at full throttle
HID = 32
TWO_PI = 2.0 / np.pi


def init_params(key):
    ks = random.split(key, 3)

    def layer(k, nin, nout, s=1.0):
        return (random.normal(k, (nin, nout)) * (s / np.sqrt(nin)), jnp.zeros((nout,)))
    return [layer(ks[0], 4, HID), layer(ks[1], HID, HID), layer(ks[2], HID, 2, s=0.1)]


def policy(params, obs):
    x = obs
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b                      # (…,2) raw -> (beta_raw, throttle_raw)


def observe(V, i, V0, Vf, i_tgt):
    return jnp.stack([(V - Vf) / V0, V / V0, i, i_tgt], axis=-1)


def integrate(params, V0, Vf, i0, i_tgt, n=N_STEPS):
    """Roll the averaged dynamics; return (V_final, i_final, dv)."""
    B = V0.shape[0]

    def step(carry, _):
        V, i, dv = carry
        act = policy(params, observe(V, i, V0, Vf, i_tgt))
        beta = 0.5 * np.pi * jnp.tanh(act[:, 0])
        thr = 0.5 * (1.0 + jnp.tanh(act[:, 1]))        # throttle in (0,1)
        dvs = thr * F_THRUST * DT
        V = V - dvs * jnp.cos(beta)
        V = jnp.clip(V, 0.5, None)                     # keep V positive/finite
        i = i + thr * TWO_PI * (F_THRUST / V) * jnp.sin(beta) * DT
        dv = dv + dvs
        return (V, i, dv), None

    (V, i, dv), _ = lax.scan(step, (V0, i0, jnp.zeros(B)), None, length=n)
    return V, i, dv


def sample_tasks(key, B):
    k = random.split(key, 4)

    def u(kk, lo, hi):
        return lo + (hi - lo) * random.uniform(kk, (B,))
    r0 = R_E + u(k[0], 300.0, 1000.0)
    rf = r0 * u(k[1], 4.0, 7.0)                         # LEO->GEO-ish ratio
    V0 = jnp.sqrt(MU_EARTH / r0)
    Vf = jnp.sqrt(MU_EARTH / rf)
    i0 = u(k[2], 0.0, np.radians(30.0))
    i_tgt = jnp.zeros(B)                                # target equatorial
    return V0, Vf, i0, i_tgt


def edelbaum_ref(V0, Vf, i0, i_tgt):
    dI = np.abs(np.asarray(i0) - np.asarray(i_tgt))
    V0 = np.asarray(V0); Vf = np.asarray(Vf)
    return np.sqrt(np.maximum(V0 ** 2 - 2 * V0 * Vf * np.cos(0.5 * np.pi * dI) + Vf ** 2, 0.0))


def make_loss(w_v=2000.0, w_i=2000.0):
    # Δv ~ 5 km/s; a 5% Vf-miss is ((0.05))^2=0.0025 and a 5deg i-miss is
    # (0.087)^2=0.0076, so weights must be ~O(10^3) for reaching the target to
    # dominate the Δv-minimization (else the policy just undershoots — R-C2a).
    def loss(params, V0, Vf, i0, i_tgt):
        V, i, dv = integrate(params, V0, Vf, i0, i_tgt)
        vmiss = ((V - Vf) / Vf) ** 2
        imiss = (i - i_tgt) ** 2
        return (dv.mean() + w_v * vmiss.mean() + w_i * imiss.mean())
    return loss


def train(args):
    params = init_params(random.PRNGKey(1))
    loss_fn = make_loss()
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def adam_init(p):
        return (tree_map(jnp.zeros_like, p), tree_map(jnp.zeros_like, p), 0)
    opt = adam_init(params)

    @jax.jit
    def step(params, opt, V0, Vf, i0, it, lr):
        l, g = vg(params, V0, Vf, i0, it)
        gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        scale = jnp.minimum(1.0, 10.0 / jnp.maximum(gn, 1e-9))
        g = tree_map(lambda x: x * scale, g)
        m, v, t = opt; t = t + 1
        m = tree_map(lambda m_, g_: 0.9 * m_ + 0.1 * g_, m, g)
        v = tree_map(lambda v_, g_: 0.999 * v_ + 0.001 * g_ * g_, v, g)
        mh = tree_map(lambda m_: m_ / (1 - 0.9 ** t), m)
        vh = tree_map(lambda v_: v_ / (1 - 0.999 ** t), v)
        params = tree_map(lambda p_, m_, v_: p_ - lr * m_ / (jnp.sqrt(v_) + 1e-8),
                          params, mh, vh)
        return params, (m, v, t), l

    key = random.PRNGKey(args.seed)
    for it in range(args.iters):
        V0, Vf, i0, itg = sample_tasks(random.fold_in(key, it), args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))
        params, opt, l = step(params, opt, V0, Vf, i0, itg, jnp.float32(lr))
        if (it + 1) % args.eval_every == 0 or it == 0:
            print(f"it={it:4d} loss={float(l):.4f}", flush=True)
    save(args.save, params)
    print(f"saved {args.save}", flush=True)
    evaluate(params)


def save(path, params):
    d = {}
    for i, (w, b) in enumerate(params):
        d[f"w{i}"] = np.asarray(w); d[f"b{i}"] = np.asarray(b)
    np.savez(path, **d)


def load(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def evaluate(params, seed=31337, episodes=4096):
    V0, Vf, i0, itg = sample_tasks(random.PRNGKey(seed), episodes)
    V, i, dv = (np.asarray(x) for x in integrate(params, V0, Vf, i0, itg))
    edel = edelbaum_ref(V0, Vf, i0, itg)
    v_err = np.abs(V - np.asarray(Vf)) / np.asarray(Vf)
    i_err = np.degrees(np.abs(i - np.asarray(itg)))
    # only score maneuvers that actually reached the target box (V and i)
    ok = (v_err < 0.02) & (i_err < 1.0)
    ratio = dv[ok] / edel[ok]
    print(f"averaged-Edelbaum eval ({episodes} tasks):")
    print(f"  reached target box (V<2%, i<1deg) = {100*ok.mean():.1f}%")
    if ok.any():
        print(f"  median Δv/edelbaum = {np.median(ratio):.4f}  "
              f"(1.0 = matches the analytic optimum; <1 impossible)")
        print(f"  mean   Δv/edelbaum = {np.mean(ratio):.4f}  "
              f"p90={np.percentile(ratio,90):.4f}")
    # the marquee anchor: LEO->GEO @ 28.5deg
    V0a = jnp.array([speed_circular(R_E + 300.0)])
    Vfa = jnp.array([speed_circular(42164.0)])
    i0a = jnp.array([np.radians(28.5)]); ita = jnp.array([0.0])
    Va, ia, dva = (np.asarray(x) for x in integrate(params, V0a, Vfa, i0a, ita))
    ea = edelbaum_ref(V0a, Vfa, i0a, ita)
    print(f"  ANCHOR LEO->GEO@28.5: Δv={float(dva[0]):.3f} vs edelbaum {float(ea[0]):.3f} "
          f"km/s (ratio {float(dva[0]/ea[0]):.3f}); Vf err "
          f"{100*abs(float(Va[0])-float(Vfa[0]))/float(Vfa[0]):.1f}% i err "
          f"{np.degrees(abs(float(ia[0]))):.2f}deg")


def sanity():
    """R-C1: fixed-schedule integrator checks vs the closed form."""
    B = 1
    V0 = jnp.array([speed_circular(R_E + 300.0)])
    Vf = jnp.array([speed_circular(42164.0)])

    def const_policy(beta, thr):
        return [(jnp.zeros((4, 1)), jnp.array([np.arctanh(np.clip(2 * beta / np.pi, -.999, .999))])),
                (jnp.zeros((1, 1)), jnp.zeros((1,))),
                (jnp.zeros((1, 2)), jnp.array([0.0, np.arctanh(2 * thr - 1)]))]
    # β=0, near-full throttle, enough steps -> pure altitude Δv = |V0-Vf|, i unchanged
    p0 = const_policy(0.0, 0.999)
    V, i, dv = integrate(p0, V0, Vf, jnp.array([np.radians(20.0)]), jnp.zeros(B), n=400)
    print(f"R-C1 pure altitude (β=0): reached Vf? V={float(V[0]):.3f} target {float(Vf[0]):.3f}; "
          f"Δv={float(dv[0]):.3f} vs |V0-Vf|={float(V0[0]-Vf[0]):.3f}; "
          f"Δi={np.degrees(abs(float(i[0])-np.radians(20.0))):.3f}deg (want ~0)")
    print("  (Δv exceeds |V0-Vf| here only because full-throttle overshoots Vf; the "
          "policy learns to coast. Check: pre-overshoot Δv tracks |V0-Vf|.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--init", type=str, default="models/edelbaum.npz")
    ap.add_argument("--save", type=str, default="models/edelbaum.npz")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.sanity:
        sanity()
    if args.train:
        train(args)
    if args.eval and not args.train:
        evaluate(load(args.init))


if __name__ == "__main__":
    main()
