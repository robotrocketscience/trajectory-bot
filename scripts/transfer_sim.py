#!/usr/bin/env python3
"""Low-thrust multi-rev altitude-raise transfer (Build B) — the eclipse-gift vehicle.

Circular r1 -> circular r2 (r2 = r1 * ratio) at low thrust, over many revolutions.
Unlike circularize-at-own-apoapsis (E2, no burn-location freedom), a raise spiral
lets the policy thrust on any arc of any rev, so it CAN defer thrust to sunlit arcs
when the eclipse gate is on — the freedom E2 said the gift needs.

Reuse: an altitude raise IS jaxsim's circularize objective (a->rt, e->0) from a
circular start with rt=r2 >> r1, so this drives jaxsim's dynamics / _decision_step
/ make_loss / eclipse gate verbatim; only sample_transfer and the Edelbaum baseline
are new. Baseline: coplanar Edelbaum/low-thrust limit |v_circ(r1) - v_circ(r2)|
(the continuous-thrust optimum; RK4 finite-time exceeds it = the loss).

    uv run --with "jax[cuda12]" python scripts/transfer_sim.py --train \
        --a-thrust 5e-4 --save models/transfer.npz
    uv run ... transfer_sim.py --eval  --a-thrust 5e-4 --init models/transfer.npz
    uv run ... transfer_sim.py --shadow --a-thrust 5e-4 --init A.npz B.npz   # gift probe
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

R1_ALT = (400.0, 1500.0)      # start circular altitude band [km]
RATIO = (1.3, 1.8)            # r2/r1 (kept so |v1-v2| < budget at r1~7000)


def sample_transfer(key, B, r1_alt=R1_ALT, ratio=RATIO, inc_max=J.INC_MAX):
    """Circular orbit at r1 (inclined), target rt = r2 = r1*ratio. e=0 start."""
    k = random.split(key, 6)

    def u(kk, lo, hi):
        return lo + (hi - lo) * random.uniform(kk, (B,))
    r1 = J.R_BODY + u(k[0], *r1_alt)
    r2 = r1 * u(k[1], *ratio)
    v1 = jnp.sqrt(J.MU / r1)
    nu = u(k[2], 0.0, 2 * np.pi)
    z = jnp.zeros_like(r1)
    pf = jnp.stack([r1 * jnp.cos(nu), r1 * jnp.sin(nu), z], axis=1)
    pfv = jnp.stack([-v1 * jnp.sin(nu), v1 * jnp.cos(nu), z], axis=1)
    inc = u(k[3], 0.0, inc_max); raan = u(k[4], 0.0, 2 * np.pi)
    ci, si = jnp.cos(inc), jnp.sin(inc); cr, sr = jnp.cos(raan), jnp.sin(raan)

    def rot(vec):                       # incl about x, then RAAN about z (jaxsim form)
        y = vec[:, 1] * ci - vec[:, 2] * si
        zc = vec[:, 1] * si + vec[:, 2] * ci
        x = vec[:, 0]
        return jnp.stack([x * cr - y * sr, x * sr + y * cr, zc], axis=1)
    r_vec = rot(pf); v_vec = rot(pfv)
    q0 = J.qnorm(random.normal(k[5], (B, 4)))
    w0 = jnp.zeros((B, 3))
    return jnp.concatenate([r_vec, v_vec, q0, w0], axis=1), r2


def edelbaum_coplanar(s0, rt):
    """|v_circ(r1) - v_circ(r2)| per episode — the coplanar low-thrust Δv bound."""
    r1 = np.linalg.norm(np.asarray(s0[:, 0:3]), axis=1)
    rt = np.asarray(rt)
    return np.abs(np.sqrt(J.MU / r1) - np.sqrt(J.MU / rt))


def rollout(params, s0, rt, H):
    """Drive jaxsim's _decision_step (13-obs, scalar A_THRUST, eclipse gate) over H."""
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))

    def scanfn(c, _):
        c2, _ = J._decision_step(params, c, rt)
        return c2, None
    (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
    return state, fuel, dv, crash, latch


def load(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def save_npz(path, params):
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"w{i}"] = np.asarray(w); out[f"b{i}"] = np.asarray(b)
    np.savez(path, **out)


def evaluate(params, episodes, H, seed):
    s0, rt = sample_transfer(random.PRNGKey(seed), episodes)
    roll = jax.jit(lambda p, s, r: rollout(p, s, r, H))
    state, fuel, dv, crash, latch = roll(params, s0, rt)
    ae, e = (np.asarray(x) for x in J.a_err_e(state, rt))
    dv = np.asarray(dv); latch = np.asarray(latch); crash = np.asarray(crash)
    edel = edelbaum_coplanar(s0, rt)
    clean = crash == 0.0; ok = latch & clean
    print(f"transfer fresh eval: {episodes} eps, a_thrust={J.A_THRUST:.1e}, H={H}, "
          f"eclipse={J.ECLIPSE}")
    print(f"  success (a,e) = {100*latch.mean():.1f}%   crash-free = {100*clean.mean():.1f}%")
    if ok.any():
        print(f"  median Δv          = {np.median(dv[ok]):.3f} km/s  (n={ok.sum()})")
        print(f"  median edelbaum    = {np.median(edel[ok]):.3f} km/s")
        print(f"  median Δv/edelbaum = {np.median(dv[ok]/edel[ok]):.3f}  "
              f"(>1 = finite-time loss)")


# --- eclipse-avoidance probe on transfer starts (shadow-fraction of commanded Δv) --
def shadow_probe(params, episodes, H, seed):
    J.ECLIPSE = True
    s0, rt = sample_transfer(random.PRNGKey(seed), episodes)

    def probe(p, s, r):
        B = s.shape[0]
        carry = (s, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),   # sun Δv
                 jnp.zeros((B,)), jnp.zeros((B,), bool))            # shadow Δv, latch

        def dec(c, _):
            st, fuel, sdv, shdv, latch = c
            act = J.policy(p, J.observe(st, r, jnp.clip(fuel, 0.0, None)))
            coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

            def sub(c2, _):
                st, fuel, sdv, shdv = c2
                t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
                d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
                d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
                omega = J.point_rate(st[:, 6:10], d)
                sun = J.thrust_gate(st[:, 0:3])
                cmd = throttle * (fuel > 0).astype(jnp.float32)
                cmd_dv = cmd * J.A_THRUST * J.DT
                sdv = sdv + cmd_dv * sun
                shdv = shdv + cmd_dv * (1.0 - sun)
                thr = cmd * sun
                fuel = fuel - cmd_dv * sun
                st = J.rk4(st, omega, thr)
                return (st, fuel, sdv, shdv), None
            (st, fuel, sdv, shdv), _ = lax.scan(sub, (st, fuel, sdv, shdv),
                                                None, length=J.REPEAT)
            ae, e = J.a_err_e(st, r)
            latch = latch | ((ae < J.A_TOL) & (e < J.E_TOL))
            dead = c[4]
            st = jnp.where(dead[:, None], c[0], st); fuel = jnp.where(dead, c[1], fuel)
            sdv = jnp.where(dead, c[2], sdv); shdv = jnp.where(dead, c[3], shdv)
            latch = jnp.where(dead, c[4], latch)
            return (st, fuel, sdv, shdv, latch), None
        (st, fuel, sdv, shdv, latch), _ = lax.scan(dec, carry, None, length=H)
        return sdv, shdv, latch
    roll = jax.jit(probe)
    sdv, shdv, latch = (np.asarray(x) for x in roll(params, s0, rt))
    tot = sdv + shdv
    frac = float(np.sum(shdv) / max(np.sum(tot), 1e-9))
    return np.mean(sdv), np.mean(shdv), 100 * frac, 100 * latch.mean()


# --- diff-sim training (reuse jaxsim.make_loss; per-episode trim grads) ------------
def _adam_apply(params, grads, opt, lr):
    m, v, t = opt
    t = t + 1.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, grads)
    v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, grads)
    bc = jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = tree_map(lambda p_, m_, v_: p_ - lr * bc * m_ / (jnp.sqrt(v_) + eps),
                      params, m, v)
    return params, (m, v, t)


def train(args):
    params = load(args.init) if args.init else J.init_params(random.PRNGKey(1), 0.01)
    loss_fn = J.make_loss(args.horizon, w_orbit=args.w_orbit, w_dv=args.w_dv,
                          w_crash=args.w_crash)
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
        cutoff = jnp.sort(norms)[B - args.trim_ep - 1]
        scale = (norms <= cutoff).astype(jnp.float32)
        scale = scale * jnp.minimum(1.0, args.clip_ep / jnp.maximum(norms, 1e-12))
        kept = jnp.maximum(jnp.sum((scale > 0).astype(jnp.float32)), 1.0)
        grads = tree_map(
            lambda g: jnp.sum(g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), axis=0) / kept,
            grads)
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        new_p, new_o = _adam_apply(params, grads, opt, lr)
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
    ps, prt = sample_transfer(random.PRNGKey(999_983), 512)
    pedel = edelbaum_coplanar(ps, prt)
    roll = jax.jit(lambda p, s, r: rollout(p, s, r, args.horizon))
    best = None; best_s = -1.0
    for it in range(args.iters):
        s0, rt = sample_transfer(random.fold_in(key, it), args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))
        params, opt, l, gmax = train_step(params, opt, s0, rt, jnp.float32(lr))
        ema_p = tree_map(lambda a, b: tau * a + (1 - tau) * b, ema_p, params)
        if (it + 1) % args.eval_every == 0 or it == 0:
            state, fuel, dv, crash, latch = roll(ema_p, ps, prt)
            latch = np.asarray(latch); dv = np.asarray(dv); crash = np.asarray(crash)
            ok = latch & (crash == 0.0)
            s = 100 * latch.mean()
            vr = float(np.median(dv[ok] / pedel[ok])) if ok.any() else float("nan")
            star = ""
            if s > best_s:
                best_s = s; best = tree_map(np.asarray, ema_p); star = " *"
            print(f"it={it:4d} loss={float(l):+.3f} gmax={float(gmax):.1e} "
                  f"succ={s:.1f}% Δv/edel={vr:.3f}{star}", flush=True)
    save_npz(args.save, best)
    save_npz(args.save.replace(".npz", "_final.npz"), tree_map(np.asarray, ema_p))
    print(f"saved {args.save} (best succ={best_s:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--shadow", action="store_true", help="R-B3 shadow-fraction probe")
    ap.add_argument("--init", type=str, nargs="*", default=[])
    ap.add_argument("--save", type=str, default="models/transfer.npz")
    ap.add_argument("--a-thrust", type=float, default=5e-4)
    ap.add_argument("--eclipse", action="store_true")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=480)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-orbit", type=float, default=4.0)
    ap.add_argument("--w-dv", type=float, default=0.05)
    ap.add_argument("--w-crash", type=float, default=5.0)
    ap.add_argument("--trim-ep", type=int, default=16)
    ap.add_argument("--clip-ep", type=float, default=20000.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=31337)
    args = ap.parse_args()
    J.DV_BUDGET = args.budget; J.ABSORB = True; J.PHI_DV = True; J.D_EPS = 1e-4
    J.A_THRUST = args.a_thrust; J.ECLIPSE = args.eclipse
    print(f"jax devices: {jax.devices()}  a_thrust={J.A_THRUST:.1e} eclipse={J.ECLIPSE} "
          f"budget={J.DV_BUDGET} H={args.horizon}", flush=True)
    if args.train:
        args.init = args.init[0] if args.init else ""
        train(args)
    if args.eval:
        evaluate(load(args.init[0]), args.episodes, args.horizon, args.eval_seed)
    if args.shadow:
        print(f"shadow-fraction probe (transfer, a_thrust={J.A_THRUST:.1e}, H={args.horizon})")
        print(f"{'ckpt':>26} {'sunΔv':>7} {'shadΔv':>7} {'shadow%':>8} {'succ':>7}")
        for path in args.init:
            sdv, shdv, frac, succ = shadow_probe(load(path), args.episodes,
                                                 args.horizon, args.eval_seed)
            print(f"{path.split('/')[-1]:>26} {sdv:>7.3f} {shdv:>7.3f} "
                  f"{frac:>7.1f}% {succ:>6.1f}%", flush=True)


if __name__ == "__main__":
    main()
