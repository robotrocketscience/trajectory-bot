#!/usr/bin/env python3
"""Binding-cost eclipse (Build D) — can the transfer policy avoid shadow when it can
SEE the umbra and PAY for commanding into it?

B refuted the eclipse-avoidance gift: shadow-fraction stayed ~37% because
cancelled-in-shadow thrust costs no fuel AND the 13-obs policy is blind to shadow.
This gives the policy (a) a sun-gate OBSERVATION (14th input: ~1 sunlit, ~0 shadow)
and (b) a penalty on commanded-but-cancelled Δv in shadow, then asks whether it
learns to WITHHOLD throttle in shadow (shadow-fraction drops below ~37%).

Reuses transfer_sim's sampler + Edelbaum baseline and jaxsim's dynamics; adds the
sun-obs, a shadow-command accumulator in the rollout, and the shadow penalty.

    uv run --with "jax[cuda12]" python scripts/eclipse_cost_sim.py --train \
        --w-shadow 2.0 --init models/transfer.npz --save models/eclipse_cost.npz
    uv run ... eclipse_cost_sim.py --eval --init models/eclipse_cost.npz
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
import jaxsim as J  # noqa: E402
import transfer_sim as T  # noqa: E402


def observe14_sun(state, rt, fuel):
    """jaxsim's 13-obs + the sun-gate value (∈(0,1)): the minimal shadow observability."""
    base = J.observe(state, rt, fuel)                    # (B,13), clipped
    sun = J.thrust_gate(state[:, 0:3])[:, None]          # ~1 sunlit, ~0 shadow (ECLIPSE on)
    return jnp.concatenate([base, sun], axis=1)          # (B,14)


def rollout_ec(params, s0, rt, H):
    """Transfer rollout with a 14-obs policy + shadow-command accounting."""
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),         # state, fuel, dv
             jnp.zeros((B,)), jnp.zeros((B,), bool), jnp.zeros((B,)))  # crash, latch, shadow_cmd

    def dec(c, _):
        st, fuel, dv, crash, latch, shc = c
        act = J.policy(params, observe14_sun(st, rt, jnp.clip(fuel, 0.0, None)))
        coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

        def sub(c2, _):
            st, fuel, dv, crash, shc = c2
            t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
            d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
            d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
            omega = J.point_rate(st[:, 6:10], d)
            sun = J.thrust_gate(st[:, 0:3])
            cmd = throttle * (fuel > 0).astype(jnp.float32)      # commanded (pre-shadow)
            thr = cmd * sun                                      # only sunlit flies
            dv_sub = thr * J.A_THRUST * J.DT
            shadow_sub = cmd * (1.0 - sun) * J.A_THRUST * J.DT   # commanded-but-cancelled
            fuel = fuel - dv_sub; dv = dv + dv_sub; shc = shc + shadow_sub
            st = J.rk4(st, omega, thr)
            crash = crash + jnp.clip((J.R_BODY - J.snorm(st[:, 0:3], axis=1)) / J.R_BODY, 0.0, None) ** 2
            return (st, fuel, dv, crash, shc), None

        (st, fuel, dv, crash, shc), _ = lax.scan(sub, (st, fuel, dv, crash, shc),
                                                 None, length=J.REPEAT)
        ae, e = J.a_err_e(st, rt)
        latch = latch | ((ae < J.A_TOL) & (e < J.E_TOL))
        dead = c[4]
        st = jnp.where(dead[:, None], c[0], st); fuel = jnp.where(dead, c[1], fuel)
        dv = jnp.where(dead, c[2], dv); crash = jnp.where(dead, c[3], crash)
        latch = jnp.where(dead, c[4], latch); shc = jnp.where(dead, c[5], shc)
        return (st, fuel, dv, crash, latch, shc), None

    (st, fuel, dv, crash, latch, shc), _ = lax.scan(dec, carry, None, length=H)
    return st, fuel, dv, crash, latch, shc


def make_loss_ec(H, w_orbit=4.0, w_dv=0.05, w_crash=5.0, w_shape=1.0, w_well=1.0,
                 w_shadow=0.0, sigma=0.15):
    def loss(params, state, rt):
        phi0 = J.potential(state, rt)
        st, fuel, dv, crash, latch, shc = rollout_ec(params, state, rt, H)
        oe_T = J.orbit_err(st, rt)
        shape = J.potential(st, rt) - phi0
        well = -jnp.exp(-oe_T / sigma)
        return (w_orbit * oe_T.mean() + w_dv * dv.mean() + w_crash * crash.mean()
                - w_shape * shape.mean() + w_well * well.mean() + w_shadow * shc.mean())
    return loss


def load_padded(path):
    d = np.load(path)
    w0 = np.asarray(d["w0"])
    if w0.shape[0] == 13:
        w0 = np.concatenate([w0, np.zeros((1, w0.shape[1]), w0.dtype)], axis=0)
    return [(jnp.asarray(w0), jnp.asarray(d["b0"]))] + \
           [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in (1, 2)]


def save_npz(path, params):
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"w{i}"] = np.asarray(w); out[f"b{i}"] = np.asarray(b)
    np.savez(path, **out)


def evaluate(params, episodes, H, seed):
    s0, rt = T.sample_transfer(random.PRNGKey(seed), episodes)
    roll = jax.jit(lambda p, s, r: rollout_ec(p, s, r, H))   # 4096-batch eclipse eval is XLA-safe
    st, fuel, dv, crash, latch, shc = roll(params, s0, rt)
    edel = T.edelbaum_coplanar(s0, rt)
    dvn = np.asarray(dv); shcn = np.asarray(shc)
    latch = np.asarray(latch); crash = np.asarray(crash)
    ok = latch & (crash == 0.0)
    frac = float(np.sum(shcn) / max(np.sum(dvn + shcn), 1e-9))
    print(f"eclipse-cost eval: {episodes} eps, H={H}, eclipse=True")
    print(f"  success = {100*latch.mean():.1f}%   shadow% of commanded Δv = {100*frac:.1f}%")
    if ok.any():
        print(f"  median Δv/edelbaum (delivered) = {np.median(dvn[ok]/edel[ok]):.3f}  (n={ok.sum()})")


def train(args):
    params = load_padded(args.init)
    loss_fn = make_loss_ec(args.horizon, w_shadow=args.w_shadow)
    base_vg = jax.value_and_grad(lambda p, s1, r1: loss_fn(p, s1[None], r1[None]))
    vg_ep = jax.vmap(base_vg, in_axes=(None, 0, 0))

    @jax.jit
    def train_step(params, opt, s0, rt, lr):
        B = s0.shape[0]
        losses, grads = vg_ep(params, s0, rt)
        grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        losses = jnp.where(jnp.isfinite(losses), losses, 0.0)
        norms = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
                             for g in jax.tree_util.tree_leaves(grads)))
        cutoff = jnp.sort(norms)[max(B - args.trim_ep - 1, 0)]   # guard trim_ep >= batch
        scale = (norms <= cutoff).astype(jnp.float32)
        scale = scale * jnp.minimum(1.0, args.clip_ep / jnp.maximum(norms, 1e-12))
        kept = jnp.maximum(jnp.sum((scale > 0).astype(jnp.float32)), 1.0)
        grads = tree_map(
            lambda g: jnp.sum(g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), axis=0) / kept,
            grads)
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        new_p, new_o = T._adam_apply(params, grads, opt, lr)
        ok = jnp.isfinite(losses.mean()) & jnp.isfinite(gnorm)
        params = tree_map(lambda o, n: jnp.where(ok, n, o), params, new_p)
        (mo, vo, to), (mn, vn, tn) = opt, new_o
        opt = (tree_map(lambda o, n: jnp.where(ok, n, o), mo, mn),
               tree_map(lambda o, n: jnp.where(ok, n, o), vo, vn),
               jnp.where(ok, tn, to))
        return params, opt, losses.mean(), norms.max()

    opt = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), jnp.array(0.0))
    ema_p = params; tau = args.ema
    key = random.PRNGKey(args.seed)
    ps, prt = T.sample_transfer(random.PRNGKey(999_983), 512)
    pedel = T.edelbaum_coplanar(ps, prt)
    roll = jax.jit(lambda p, s, r: rollout_ec(p, s, r, args.horizon), backend="cpu")
    best = None; best_s = -1.0
    for it in range(args.iters):
        s0, rt = T.sample_transfer(random.fold_in(key, it), args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))
        params, opt, l, gmax = train_step(params, opt, s0, rt, jnp.float32(lr))
        ema_p = tree_map(lambda a, b: tau * a + (1 - tau) * b, ema_p, params)
        if (it + 1) % args.eval_every == 0 or it == 0:
            st, fuel, dv, crash, latch, shc = roll(ema_p, ps, prt)
            latch = np.asarray(latch); dv = np.asarray(dv); shc = np.asarray(shc)
            crash = np.asarray(crash); ok = latch & (crash == 0.0)
            s = 100 * latch.mean()
            frac = 100 * float(np.sum(shc) / max(np.sum(dv + shc), 1e-9))
            vr = float(np.median(dv[ok] / pedel[ok])) if ok.any() else float("nan")
            star = ""
            if s > best_s:
                best_s = s; best = tree_map(np.asarray, ema_p); star = " *"
            print(f"it={it:4d} loss={float(l):+.3f} succ={s:.1f}% shadow%={frac:.1f} "
                  f"Δv/edel={vr:.3f}{star}", flush=True)
    save_npz(args.save, best)
    print(f"saved {args.save} (best succ={best_s:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--init", type=str, default="models/transfer.npz")
    ap.add_argument("--save", type=str, default="models/eclipse_cost.npz")
    ap.add_argument("--w-shadow", type=float, default=0.0, help="penalty on shadow-commanded Δv")
    ap.add_argument("--a-thrust", type=float, default=5e-4)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=480)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--trim-ep", type=int, default=16)
    ap.add_argument("--clip-ep", type=float, default=20000.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=31337)
    args = ap.parse_args()
    J.DV_BUDGET = args.budget; J.ABSORB = True; J.PHI_DV = True; J.D_EPS = 1e-4
    J.A_THRUST = args.a_thrust; J.ECLIPSE = True     # Build D always studies eclipse
    print(f"jax devices: {jax.devices()}  a_thrust={J.A_THRUST:.1e} w_shadow={args.w_shadow} "
          f"budget={J.DV_BUDGET} H={args.horizon}", flush=True)
    if args.train:
        train(args)
    if args.eval and not args.train:
        evaluate(load_padded(args.init) if np.load(args.init)["w0"].shape[0] == 13
                 else [(jnp.asarray(np.load(args.init)[f"w{i}"]),
                        jnp.asarray(np.load(args.init)[f"b{i}"])) for i in range(3)],
                 args.episodes, args.horizon, args.eval_seed)


if __name__ == "__main__":
    main()
