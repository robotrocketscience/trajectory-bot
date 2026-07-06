#!/usr/bin/env python3
"""Combined circularize + plane-change task (JAX) — the single-burn KSC step.

An inclined ellipse (apoapsis on the equatorial line of nodes, as the shared
sampler builds it) -> circular AND equatorial at its own apoapsis, in ONE
combined apoapsis burn. The naive textbook route circularizes, then pays a
separate plane change at circular speed; the combined burn folds both into one
vector Δv (~25% cheaper at 28.5°, see tbot.orbital3d.combined_circularize_plane_dv).
The thesis test: a diff-sim policy optimizing raw Δv should DISCOVER the combined
burn rather than the naive decomposition.

Reuses jaxsim's dynamics/constants verbatim (deriv/rk4/orbit_frame/point_rate/
elements); adds only a plane-aware observation (13->16), an inclination success
term, and a privileged combined-burn expert for DAgger. This module is the env +
expert + a forward-only validation of the expert (no training).

    uv run --with jax python scripts/combined_sim.py --validate --episodes 512

Experiment code (excluded from the strict-typed library).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

sys.path.insert(0, "scripts")
import jaxsim as J  # noqa: E402

MU = J.MU
Z_AXIS = jnp.array([0.0, 0.0, 1.0])

# task knobs
INC_TOL = np.radians(3.0)          # plane tolerance for the success latch
COS_INC_TOL = float(np.cos(INC_TOL))
W_PLANE = 4.0                      # weight on plane error in the shaping potential

# expert knobs (tuned by --validate)
K_EXPERT = 1.5                     # throttle gain on |Δv| / v_circ
GATE_RDOT = 0.35                   # [km/s] radial-speed half-width of the apo burn gate


# --- plane-aware observation (jaxsim's 13 + 3 equatorial-normal projections) ---
def observe16(state, rt, fuel):
    base = J.observe(state, rt, fuel)                  # 13, rt-normalized + clipped
    t, w, s = J.orbit_frame(state[:, 0:3], state[:, 3:6])
    # +z (the target equatorial normal) resolved in the orbit frame: z·w = cos(inc),
    # and z·t / z·s tell the policy which way to tilt the plane. Everything it needs
    # to null inclination, in the same frame it already acts in.
    plane = jnp.stack([(Z_AXIS * t).sum(1), (Z_AXIS * w).sum(1),
                       (Z_AXIS * s).sum(1)], axis=1)
    return jnp.concatenate([base, plane], axis=1)      # 16


def plane_err(state):
    """1 - cos(inc): ->0 at a prograde equatorial orbit, smooth (safe gradient)."""
    h = J.cross(state[:, 0:3], state[:, 3:6])
    cos_i = h[:, 2] / J.snorm(h, axis=1)
    return 1.0 - cos_i


def is_success(state, rt):
    ae, e = J.a_err_e(state, rt)
    h = J.cross(state[:, 0:3], state[:, 3:6])
    cos_i = h[:, 2] / J.snorm(h, axis=1)
    return (ae < J.A_TOL) & (e < J.E_TOL) & (cos_i > COS_INC_TOL)


def orbit_err_plane(state, rt):
    return J.orbit_err(state, rt) + W_PLANE * plane_err(state)


# --- 16-input MLP (identical forward to J.policy; only the init shape changes) ---
def init_params(key, final_scale=1.0):
    ks = random.split(key, 3)

    def layer(k, nin, nout, scale=1.0):
        return (random.normal(k, (nin, nout)) * (scale / np.sqrt(nin)),
                jnp.zeros((nout,)))
    return [layer(ks[0], 16, J.HID), layer(ks[1], J.HID, J.HID),
            layer(ks[2], J.HID, 4, scale=final_scale)]


# --- privileged combined-burn expert (full state; generates DAgger labels) ------
def expert_action(state):
    """One combined apoapsis burn: rotate+scale v toward the equatorial circular
    velocity at the current position. Gated to the apoapsis half near the apsis so
    the burn straddles the node; throttle scales with the remaining Δv so it stops
    when the orbit is circular+equatorial."""
    r = state[:, 0:3]; v = state[:, 3:6]
    rmag = J.snorm(r, axis=1, keepdims=True)
    a, e = J.elements(state)
    v_circ = jnp.sqrt(MU / rmag)                       # circular speed at current r
    rhat = r / rmag
    zc = jnp.cross(Z_AXIS, rhat)                        # ⊥r, in the equatorial plane
    zc = zc / J.snorm(zc, axis=1, keepdims=True)
    prog = jnp.sign((zc * v).sum(1, keepdims=True) + 1e-12)   # match motion sense
    v_tgt = v_circ * prog * zc                          # equatorial circular target v
    dv = v_tgt - v

    t, w, s = J.orbit_frame(r, v)
    coeffs = jnp.stack([(dv * t).sum(1), (dv * w).sum(1), (dv * s).sum(1)], axis=1)
    cdir = coeffs / J.snorm(coeffs, axis=1, keepdims=True, eps=1e-6)

    rdot = (r * v).sum(1) / rmag[:, 0]
    outer = (rmag[:, 0] > a).astype(jnp.float32)        # apoapsis half only
    gate = jnp.clip(1.0 - jnp.abs(rdot) / GATE_RDOT, 0.0, 1.0)
    dv_mag = J.snorm(dv, axis=1)
    throttle = jnp.clip(K_EXPERT * dv_mag / v_circ[:, 0], 0.0, 1.0) * outer * gate
    return jnp.concatenate([cdir, throttle[:, None]], axis=1)


# --- rollout: drive the shared substep dynamics with a chosen action fn ----------
def _decision(action_fn, params, carry, rt):
    """One decision with plane-aware obs/success. Mirrors J._decision_step's
    substep integration and Δv accounting exactly; action_fn(params, state)->act."""
    state, fuel, dv, crash, latch = carry
    act = action_fn(params, state, jnp.clip(fuel, 0.0, None))
    coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

    def substep(c2, _):
        st, fu, dvv, cr = c2
        t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
        omega_cmd = J.point_rate(st[:, 6:10], d)
        thr = throttle * (fu > 0).astype(jnp.float32)
        dv_sub = thr * J.A_THRUST * J.DT
        fu = fu - dv_sub; dvv = dvv + dv_sub
        st = J.rk4(st, omega_cmd, thr)
        rnow = J.snorm(st[:, 0:3], axis=1)
        cr = cr + jnp.clip((J.R_BODY - rnow) / J.R_BODY, 0.0, None) ** 2
        return (st, fu, dvv, cr), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash),
                                           None, length=J.REPEAT)
    latch = latch | is_success(state, rt)
    # absorbing success: episodes already latched BEFORE this step freeze, so
    # post-success exploration drift can't unlatch them or spend more Δv.
    dead = carry[4]
    state = jnp.where(dead[:, None], carry[0], state)
    fuel = jnp.where(dead, carry[1], fuel)
    dv = jnp.where(dead, carry[2], dv)
    crash = jnp.where(dead, carry[3], crash)
    latch = jnp.where(dead, carry[4], latch)
    return (state, fuel, dv, crash, latch), None


def rollout(action_fn, params, s0, rt, H=120, collect_states=False):
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))
    if collect_states:
        # record the visited (STATE, fuel): the expert is a function of state, not
        # obs, so DAgger relabeling needs states — obs16 and labels both derive from
        # them, with fuel feeding the obs budget feature.
        def scanfn(c, _):
            c2, _ = _decision(action_fn, params, c, rt)
            return c2, (c[0], c[1])
        (state, fuel, dv, crash, latch), (st_seq, fu_seq) = lax.scan(
            scanfn, carry, None, length=H)
        return (state, fuel, dv, crash, latch), (st_seq.reshape(-1, st_seq.shape[-1]),
                                                 fu_seq.reshape(-1))

    def scanfn(c, _):
        c2, _ = _decision(action_fn, params, c, rt)
        return c2, None
    (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
    return (state, fuel, dv, crash, latch), None


# --- analytic per-episode baseline (vectorized combined_circularize_plane_dv) ---
def baseline_dv(s0):
    r = np.asarray(s0[:, 0:3]); v = np.asarray(s0[:, 3:6])
    rmag = np.linalg.norm(r, axis=1); vmag = np.linalg.norm(v, axis=1)
    energy = 0.5 * vmag ** 2 - MU / rmag
    a = -MU / (2.0 * energy)
    h = np.cross(r, v); hmag = np.linalg.norm(h, axis=1)
    e = np.sqrt(np.clip(1.0 - hmag ** 2 / (MU * a), 0.0, None))
    r_a = a * (1.0 + e); r_p = a * (1.0 - e)
    inc = np.arccos(np.clip(h[:, 2] / hmag, -1.0, 1.0))
    a_ell = 0.5 * (r_p + r_a)
    v_apo = np.sqrt(MU * (2.0 / r_a - 1.0 / a_ell))
    v_circ = np.sqrt(MU / r_a)
    naive = np.abs(v_circ - v_apo) + 2.0 * v_circ * np.abs(np.sin(inc / 2.0))
    combined = np.sqrt(v_apo ** 2 + v_circ ** 2 - 2.0 * v_apo * v_circ * np.cos(inc))
    return naive, combined, inc, r_a


def validate(episodes, H, seed):
    key = random.PRNGKey(seed)
    s0, rt = J.sample_orbits(key, episodes)     # inc∈[0,40°], rt=r_a
    (state, fuel, dv, crash, latch), _ = rollout(expert_action_wrap, None, s0, rt, H=H)
    naive, combined, inc, r_a = baseline_dv(s0)
    dv = np.asarray(dv); latch = np.asarray(latch); crash = np.asarray(crash)
    clean = (crash == 0.0)
    ok = latch & clean
    print(f"expert over {episodes} episodes (H={H}):")
    print(f"  success (a,e,inc<{np.degrees(INC_TOL):.0f}°) = {100*latch.mean():.1f}%"
          f"   crash-free = {100*clean.mean():.1f}%")
    if ok.sum() > 0:
        print(f"  measured Δv on clean successes: median={np.median(dv[ok]):.3f} km/s"
              f"  (n={ok.sum()})")
        print(f"  analytic combined (same eps):   median={np.median(combined[ok]):.3f} km/s")
        print(f"  analytic naive    (same eps):   median={np.median(naive[ok]):.3f} km/s")
        print(f"  measured/combined ratio median = {np.median(dv[ok]/combined[ok]):.3f}")
        print(f"  measured/naive    ratio median = {np.median(dv[ok]/naive[ok]):.3f}")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 41)]:
        m = (np.degrees(inc) >= lo) & (np.degrees(inc) < hi)
        if m.sum():
            print(f"  inc [{lo:2d},{hi:2d})°: succ={100*latch[m].mean():5.1f}%"
                  f"  n={m.sum():4d}  median Δv={np.median(dv[m & clean]) if (m&clean).any() else float('nan'):.3f}")


def expert_action_wrap(_params, state, _fuel):
    return expert_action(state)


def _mlp_action_for(rt):
    def act(params, state, fuel):
        return J.policy(params, observe16(state, rt, fuel))
    return act


# --- DAgger: clone the combined-burn expert into the 16-input MLP ---------------
from jax.tree_util import tree_map  # noqa: E402


def loss_fn(params, obs, tgt, wts):
    pred = J.policy(params, obs)
    per = ((pred - tgt) ** 2).mean(axis=1)
    return (wts * per).sum() / jnp.maximum(wts.sum(), 1.0)


@jax.jit
def adam_step(params, opt, obs, tgt, wts, lr):
    m, v, t = opt
    _, g = jax.value_and_grad(loss_fn)(params, obs, tgt, wts)
    t = t + 1.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
    v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
    bc = jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = tree_map(lambda p_, m_, v_: p_ - lr * bc * m_ / (jnp.sqrt(v_) + eps),
                      params, m, v)
    return params, (m, v, t)


def bc_fit(params, obs, tgt, epochs, key, batch=4096, lr=5e-4, burn_w=15.0):
    wts = 1.0 + burn_w * (tgt[:, 3] > 0.05).astype(jnp.float32)   # weight burn labels
    opt = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params),
           jnp.array(0.0))
    n = obs.shape[0]
    for _ in range(epochs):
        key, k = random.split(key)
        perm = random.permutation(k, n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            params, opt = adam_step(params, opt, obs[idx], tgt[idx], wts[idx], lr)
    return params


def inc_band_eval(params, episodes=1024, H=120, seed=31337):
    s0, rt = J.sample_orbits(random.PRNGKey(seed), episodes)
    (state, fuel, dv, crash, latch), _ = rollout(_mlp_action_for(rt), params, s0, rt, H=H)
    naive, combined, inc, _ = baseline_dv(s0)
    dv = np.asarray(dv); latch = np.asarray(latch); crash = np.asarray(crash)
    clean = crash == 0.0; ok = latch & clean
    succ = 100 * latch.mean()
    ratio = float(np.median(dv[ok] / naive[ok])) if ok.any() else float("nan")
    return succ, ratio


def dagger(args):
    key = random.PRNGKey(0)
    params = init_params(random.PRNGKey(1), final_scale=0.01)
    all_obs = all_tgt = None
    best = None; best_succ = -1.0
    for it in range(args.iters):
        k = random.fold_in(key, it)
        s0, rt = J.sample_orbits(k, args.episodes)
        drv = expert_action_wrap if it == 0 else _mlp_action_for(rt)
        _, (st_flat, _fu) = rollout(drv, params, s0, rt, H=args.horizon,
                                    collect_states=True)
        rt_flat = jnp.tile(rt, args.horizon)
        obs_flat = observe16(st_flat, rt_flat, _fu)
        tgt_flat = expert_action(st_flat)
        all_obs = obs_flat if all_obs is None else jnp.concatenate([all_obs, obs_flat])
        all_tgt = tgt_flat if all_tgt is None else jnp.concatenate([all_tgt, tgt_flat])
        params = bc_fit(params, all_obs, all_tgt, args.bc_epochs,
                        random.fold_in(key, 1000 + it))
        succ, ratio = inc_band_eval(params, H=args.horizon)
        star = ""
        if succ > best_succ:                        # bank the best-success checkpoint
            best_succ = succ; best = tree_map(lambda x: np.asarray(x), params); star = " *"
        print(f"dagger it={it}  data={all_obs.shape[0]:>7d}  succ={succ:5.1f}%"
              f"  median Δv/naive={ratio:.3f}{star}", flush=True)
    d = {}
    for i, (w, b) in enumerate(best):
        d[f"w{i}"] = np.asarray(w); d[f"b{i}"] = np.asarray(b)
    np.savez(args.save, **d)
    print(f"saved {args.save}  (best succ={best_succ:.1f}%)", flush=True)


# --- diff-sim refinement: optimize raw Δv through the plane-aware rollout -------
def make_loss_plane(H, w_orbit=4.0, w_dv=0.05, w_crash=5.0, w_shape=1.0,
                    w_well=1.0, sigma=0.15):
    def loss(params, state, rt):
        act_fn = _mlp_action_for(rt)
        B = state.shape[0]
        carry = (state, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        phi0 = -orbit_err_plane(state, rt)
        def scanfn(c, _):
            c2, _ = _decision(act_fn, params, c, rt)
            return c2, None
        (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
        oe_T = orbit_err_plane(state, rt)
        shape = (-orbit_err_plane(state, rt)) - phi0      # Ng-1999 potential (telescoped)
        well = -jnp.exp(-oe_T / sigma)
        return (w_orbit * oe_T.mean() + w_dv * dv.mean() + w_crash * crash.mean()
                - w_shape * shape.mean() + w_well * well.mean())
    return loss


def diag(params, episodes=1024, H=120, seed=31337):
    s0, rt = J.sample_orbits(random.PRNGKey(seed), episodes)
    (state, fuel, dv, crash, latch), _ = rollout(_mlp_action_for(rt), params, s0, rt, H=H)
    naive, combined, inc, _ = baseline_dv(s0)
    dv = np.asarray(dv); latch = np.asarray(latch); crash = np.asarray(crash)
    clean = crash == 0.0; ok = latch & clean
    succ = 100 * latch.mean()
    vn = float(np.median(dv[ok] / naive[ok])) if ok.any() else float("nan")
    vc = float(np.median(dv[ok] / combined[ok])) if ok.any() else float("nan")
    return succ, vn, vc


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


def load_npz(path):
    d = np.load(path)
    n = sum(1 for k in d.files if k.startswith("w"))
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(n)]


def train(args):
    params = load_npz(args.init)
    loss_fn_ds = make_loss_plane(args.horizon)
    # per-episode grads (heavy-tailed through rk4): trim the top-k monsters, then the
    # trimmed mean — the recipe from the circularize campaign (Yin 2018 / R14-R19).
    base_vg = jax.value_and_grad(lambda p, s1, r1: loss_fn_ds(p, s1[None], r1[None]))
    vg_ep = jax.vmap(base_vg, in_axes=(None, 0, 0))

    @jax.jit
    def train_step(params, opt, s0, rt, lr):
        B = s0.shape[0]
        losses, grads = vg_ep(params, s0, rt)
        grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        losses = jnp.where(jnp.isfinite(losses), losses, 0.0)
        norms = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
                             for g in jax.tree_util.tree_leaves(grads)))
        scale = jnp.ones_like(norms)
        cutoff = jnp.sort(norms)[B - args.trim_ep - 1]
        scale = scale * (norms <= cutoff).astype(jnp.float32)
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
    best = None; best_succ = -1.0
    for it in range(args.iters):
        s0, rt = J.sample_orbits(random.fold_in(key, it), args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))  # cosine
        params, opt, l, gmax = train_step(params, opt, s0, rt, jnp.float32(lr))
        ema_p = tree_map(lambda a, b: tau * a + (1 - tau) * b, ema_p, params)
        if (it + 1) % args.eval_every == 0 or it == 0:
            succ, vn, vc = diag(ema_p, H=args.horizon)
            star = ""
            if succ > best_succ:
                best_succ = succ; best = tree_map(np.asarray, ema_p); star = " *"
            print(f"it={it:4d} loss={float(l):+.3f} gmax={float(gmax):.1e} "
                  f"succ={succ:5.1f}% Δv/naive={vn:.3f} Δv/comb={vc:.3f}{star}", flush=True)
    d = {}
    for i, (w, b) in enumerate(best):
        d[f"w{i}"] = np.asarray(w); d[f"b{i}"] = np.asarray(b)
    np.savez(args.save, **d)
    print(f"saved {args.save}  (best succ={best_succ:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dagger", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--trim-ep", type=int, default=16)
    ap.add_argument("--clip-ep", type=float, default=20000.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--init", type=str, default="models/dagger_combined.npz")
    ap.add_argument("--save", type=str, default="models/dagger_combined.npz")
    args = ap.parse_args()
    J.DV_BUDGET = args.budget
    J.D_EPS = 1e-4          # cap the coast-direction gradient seed (jaxsim R15 fix)
    if args.validate:
        validate(args.episodes, args.horizon, args.seed)
    if args.dagger:
        dagger(args)
    if args.train:
        train(args)


if __name__ == "__main__":
    main()
