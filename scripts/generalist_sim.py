#!/usr/bin/env python3
"""Thrust-conditioned generalist (Build A) — one circularize policy across the
low-thrust band, told its own thrust in the observation.

E1 found the low-thrust control law is thrust-specific (a 2e-4 specialist craters
at chemical thrust). This trains ONE 14-input policy (jaxsim's 13 obs + normalized
log-thrust) with A_THRUST sampled per episode over 5e-3→2e-4, and tests whether it
matches the per-thrust E1 specialists. jaxsim's A_THRUST is a scalar module global,
so per-episode variation needs a rollout that threads a per-episode `a_thrust` (B,)
array through the dynamics — hence this separate module (mirrors combined_sim). It
reuses jaxsim's constants/quaternion/elements/observe/sample_orbits verbatim.

    uv run --with "jax[cuda12]" python scripts/generalist_sim.py --train \
        --init models/warm_r28_ema_final.npz --save models/generalist.npz
    uv run ... generalist_sim.py --eval --init models/generalist.npz
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

THRUST_LO, THRUST_HI = 2e-4, 5e-3          # the E1 band
THRUST_REF = 1e-3                          # log-feature reference (decades from 1e-3)
EVAL_THRUSTS = [(5e-3, 120), (2e-3, 120), (1e-3, 180), (5e-4, 300), (2e-4, 480)]


# --- thrust-conditioned observation (jaxsim's 13 + normalized log-thrust) --------
def thrust_feature(a_thrust):
    return jnp.log10(a_thrust / THRUST_REF)          # ≈ decades from 1e-3


def observe14(state, rt, fuel, a_thrust):
    base = J.observe(state, rt, fuel)                # (B,13), already clipped
    f = jnp.clip(thrust_feature(a_thrust)[:, None], -10.0, 10.0)
    return jnp.concatenate([base, f], axis=1)        # (B,14)


# --- dynamics with a per-episode thrust (B,1). Mirrors J.deriv/J.rk4 exactly, but
#     the thrust magnitude is per-episode instead of the scalar global A_THRUST.
#     (J2 omitted: J2_COEF is 0 in these experiments; two-body path only.) ----------
def deriv_g(state, omega_cmd, throttle, a_thrust):
    r = state[:, 0:3]; v = state[:, 3:6]; q = state[:, 6:10]; w = state[:, 10:13]
    rmag = jnp.maximum(J.snorm(r, axis=1, keepdims=True), J.R_BODY)
    grav = -J.MU * r / rmag ** 3
    b_hat = jnp.zeros_like(v).at[:, 0].set(1.0)
    tdir = J.qrotate(q, b_hat)
    acc = grav + (throttle * a_thrust)[:, None] * tdir
    z = jnp.zeros((w.shape[0], 1))
    qdot = 0.5 * J.qmul(q, jnp.concatenate([z, w], axis=1))
    wdot = J.RATE_GAIN * (omega_cmd - w)
    return jnp.concatenate([v, acc, qdot, wdot], axis=1)


def rk4_g(state, omega_cmd, throttle, a_thrust):
    k1 = deriv_g(state, omega_cmd, throttle, a_thrust)
    k2 = deriv_g(state + 0.5 * J.DT * k1, omega_cmd, throttle, a_thrust)
    k3 = deriv_g(state + 0.5 * J.DT * k2, omega_cmd, throttle, a_thrust)
    k4 = deriv_g(state + J.DT * k3, omega_cmd, throttle, a_thrust)
    s = state + (J.DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    q = J.qnorm(s[:, 6:10])
    return jnp.concatenate([s[:, 0:6], q, s[:, 10:13]], axis=1)


def _decision(params, carry, rt, a_thrust):
    state, fuel, dv, crash, latch = carry
    obs = observe14(state, rt, jnp.clip(fuel, 0.0, None), a_thrust)
    act = J.policy(params, obs)
    coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

    def substep(c2, _):
        st, fu, dvv, cr = c2
        t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
        omega_cmd = J.point_rate(st[:, 6:10], d)
        gate = (fu > 0).astype(jnp.float32) * J.thrust_gate(st[:, 0:3])
        thr = throttle * gate
        dv_sub = thr * a_thrust * J.DT
        fu = fu - dv_sub; dvv = dvv + dv_sub
        st = rk4_g(st, omega_cmd, thr, a_thrust)
        rnow = J.snorm(st[:, 0:3], axis=1)
        cr = cr + jnp.clip((J.R_BODY - rnow) / J.R_BODY, 0.0, None) ** 2
        return (st, fu, dvv, cr), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash),
                                           None, length=J.REPEAT)
    ae, e = J.a_err_e(state, rt)
    latch = latch | ((ae < J.A_TOL) & (e < J.E_TOL))
    dead = carry[4]                                   # absorbing success (env-style)
    state = jnp.where(dead[:, None], carry[0], state)
    fuel = jnp.where(dead, carry[1], fuel)
    dv = jnp.where(dead, carry[2], dv)
    crash = jnp.where(dead, carry[3], crash)
    latch = jnp.where(dead, carry[4], latch)
    return (state, fuel, dv, crash, latch), None


def rollout(params, s0, rt, a_thrust, H):
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))

    def scanfn(c, _):
        c2, _ = _decision(params, c, rt, a_thrust)
        return c2, None
    (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
    return state, fuel, dv, crash, latch


def sample_thrust(key, B):
    lo, hi = np.log10(THRUST_LO), np.log10(THRUST_HI)
    return 10.0 ** (lo + (hi - lo) * random.uniform(key, (B,)))


# --- R-A3 distillation: clone the 5 E1 specialists into ONE conditioned net ------
# (thrust, horizon, 13-input specialist checkpoint) — the E1 diagonal winners.
SPECIALISTS = [
    (5e-3, 120, "models/warm_r28_ema_final.npz"),
    (2e-3, 120, "models/lt_2e-3.npz"),
    (1e-3, 180, "models/lt_1e-3.npz"),
    (5e-4, 300, "models/lt_5e-4.npz"),
    (2e-4, 480, "models/lt_2e-4.npz"),
]


def load13(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def collect_states(spec13, a_thrust, H, episodes, key):
    """Roll the 13-input specialist at its thrust; return its visited (state, fuel).
    The specialist is a function of the 13-obs, so DAgger relabelling needs states."""
    s0, rt = J.sample_orbits(key, episodes)
    at = jnp.full((episodes,), a_thrust)
    B = episodes
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))

    def decision(c, _):
        st, fu = c[0], c[1]
        act = J.policy(spec13, J.observe(st, rt, jnp.clip(fu, 0.0, None)))  # 13-in
        coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

        def sub(c2, _):
            s2, f2, d2, cr2 = c2
            t, w, s = J.orbit_frame(s2[:, 0:3], s2[:, 3:6])
            d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
            d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
            omega = J.point_rate(s2[:, 6:10], d)
            thr = throttle * (f2 > 0).astype(jnp.float32) * J.thrust_gate(s2[:, 0:3])
            dsub = thr * a_thrust * J.DT
            f2 = f2 - dsub; d2 = d2 + dsub
            s2 = rk4_g(s2, omega, thr, a_thrust)
            cr2 = cr2 + jnp.clip((J.R_BODY - J.snorm(s2[:, 0:3], axis=1)) / J.R_BODY, 0.0, None) ** 2
            return (s2, f2, d2, cr2), None

        (st2, fu2, dv2, cr2), _ = lax.scan(sub, (c[0], c[1], c[2], c[3]),
                                           None, length=J.REPEAT)
        latch = c[4] | ((lambda ae, e: (ae < J.A_TOL) & (e < J.E_TOL))(*J.a_err_e(st2, rt)))
        dead = c[4]
        st2 = jnp.where(dead[:, None], c[0], st2)
        fu2 = jnp.where(dead, c[1], fu2)
        return (st2, fu2, dv2, cr2, latch), (st, fu, rt)

    _, (st_seq, fu_seq, rt_seq) = lax.scan(decision, carry, None, length=H)
    return (st_seq.reshape(-1, st_seq.shape[-1]), fu_seq.reshape(-1),
            rt_seq.reshape(-1), jnp.full((H * episodes,), a_thrust))


def bc_loss(params, obs, tgt, wts):
    per = ((J.policy(params, obs) - tgt) ** 2).mean(axis=1)
    return (wts * per).sum() / jnp.maximum(wts.sum(), 1.0)


@jax.jit
def bc_step(params, opt, obs, tgt, wts, lr):
    m, v, t = opt
    _, g = jax.value_and_grad(bc_loss)(params, obs, tgt, wts)
    t = t + 1.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
    v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
    bc = jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = tree_map(lambda p_, m_, v_: p_ - lr * bc * m_ / (jnp.sqrt(v_) + eps),
                      params, m, v)
    return params, (m, v, t)


def distill(args):
    """Clone the 5 E1 specialists into one 14-input conditioned net (multi-expert BC).
    The proper test of H-A: does a single policy with thrust in its obs match the
    specialists when it is TAUGHT their per-thrust behaviour (vs diff-sim drifting to
    a compromise)? Mirrors the target-conditioning fix (conditioned expert -> DAgger)."""
    key = random.PRNGKey(args.seed)
    obs_all = tgt_all = None
    for i, (a_thrust, H, path) in enumerate(SPECIALISTS):
        spec = load13(path)
        st, fu, rt, at = collect_states(spec, a_thrust, H, args.distill_eps,
                                        random.fold_in(key, i))
        obs14 = observe14(st, rt, fu, at)
        tgt = J.policy(spec, J.observe(st, rt, fu))         # specialist's own action
        obs_all = obs14 if obs_all is None else jnp.concatenate([obs_all, obs14])
        tgt_all = tgt if tgt_all is None else jnp.concatenate([tgt_all, tgt])
        print(f"  collected {path.split('/')[-1]} @ {a_thrust:.0e}: "
              f"{obs14.shape[0]} states", flush=True)
    # start from the padded champion (a good prior at high thrust); BC pulls the rest
    params = load_padded(args.init)
    wts = 1.0 + 15.0 * (tgt_all[:, 3] > 0.05).astype(jnp.float32)   # weight burn labels
    opt = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), jnp.array(0.0))
    n = obs_all.shape[0]
    for ep in range(args.bc_epochs):
        k = random.fold_in(key, 1000 + ep)
        perm = random.permutation(k, n)
        for j in range(0, n, 4096):
            idx = perm[j:j + 4096]
            params, opt = bc_step(params, opt, obs_all[idx], tgt_all[idx], wts[idx],
                                  jnp.float32(args.lr))
        if (ep + 1) % max(1, args.bc_epochs // 8) == 0:
            print(f"  bc epoch {ep+1}/{args.bc_epochs} "
                  f"loss={float(bc_loss(params, obs_all[:8192], tgt_all[:8192], wts[:8192])):.4f}",
                  flush=True)
    save_npz(args.save, [(np.asarray(w), np.asarray(b)) for w, b in params])
    print(f"saved {args.save}", flush=True)


# --- diff-sim loss (circularize; identical shaping to jaxsim.make_loss) ----------
def make_loss_g(H, w_orbit=4.0, w_dv=0.05, w_crash=5.0, w_shape=1.0,
                w_well=1.0, sigma=0.15):
    def loss(params, state, rt, a_thrust):
        phi0 = J.potential(state, rt)
        state_f, fuel, dv, crash, latch = rollout(params, state, rt, a_thrust, H)
        oe_T = J.orbit_err(state_f, rt)
        shape = J.potential(state_f, rt) - phi0
        well = -jnp.exp(-oe_T / sigma)
        return (w_orbit * oe_T.mean() + w_dv * dv.mean() + w_crash * crash.mean()
                - w_shape * shape.mean() + w_well * well.mean())
    return loss


def make_diag_g(H):
    def diag(params, state, rt, a_thrust):
        dvgo0 = J.dv_to_go(state, rt)
        state_f, fuel, dv, crash, latch = rollout(params, state, rt, a_thrust, H)
        ae, e = J.a_err_e(state_f, rt)
        lf = latch.astype(jnp.float32)
        dvr = jnp.sum(lf * dv / jnp.maximum(dvgo0, 1e-3)) / jnp.maximum(lf.sum(), 1.0)
        return latch.mean(), dv.mean(), ae.mean(), e.mean(), (crash > 0).mean(), dvr
    return diag


# --- checkpoint I/O with 13->14 pad (zero column for the new thrust feature) ------
def load_padded(path):
    d = np.load(path)
    w0 = np.asarray(d["w0"])                          # (nin, 128)
    if w0.shape[0] == 13:                             # pad an input row of zeros
        w0 = np.concatenate([w0, np.zeros((1, w0.shape[1]), w0.dtype)], axis=0)
    params = [(jnp.asarray(w0), jnp.asarray(d["b0"]))]
    params += [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in (1, 2)]
    return params


def save_npz(path, params):
    out = {}
    for i, (w, b) in enumerate(params):
        out[f"w{i}"] = np.asarray(w); out[f"b{i}"] = np.asarray(b)
    np.savez(path, **out)


def evaluate(params, episodes, seed):
    """Fresh-set eval at each E1 thrust; the diagonal to compare to the specialists."""
    print(f"generalist fresh eval ({episodes} eps, seed {seed})")
    print(f"{'a_thrust':>10} {'H':>4} {'success':>8} {'dv':>7} {'a_err':>6} "
          f"{'e':>6} {'crash':>6} {'dvr':>6}")
    for a_thrust, h in EVAL_THRUSTS:
        s0, rt = J.sample_orbits(random.PRNGKey(seed), episodes)
        at = jnp.full((episodes,), a_thrust)
        diag = jax.jit(make_diag_g(h))
        s, dvu, ae, e, cr, dvr = (float(x) for x in diag(params, s0, rt, at))
        print(f"{a_thrust:>10.1e} {h:>4d} {100*s:>7.2f}% {dvu:>7.3f} {ae:>6.3f} "
              f"{e:>6.3f} {100*cr:>5.1f}% {dvr:>6.3f}", flush=True)


def action_control(params, seed=31337, episodes=256):
    """Assumption-2 control: same policy, same orbits, two thrusts — actions MUST
    differ, else the thrust feature is dead."""
    s0, rt = J.sample_orbits(random.PRNGKey(seed), episodes)
    fuel = jnp.full((episodes,), J.DV_BUDGET)
    a_lo = observe14(s0, rt, fuel, jnp.full((episodes,), THRUST_LO))
    a_hi = observe14(s0, rt, fuel, jnp.full((episodes,), THRUST_HI))
    act_lo = J.policy(params, a_lo); act_hi = J.policy(params, a_hi)
    d = float(jnp.abs(act_lo - act_hi).mean())
    print(f"action |Δ| between thrust {THRUST_LO:.0e} and {THRUST_HI:.0e} "
          f"on identical orbits = {d:.4f}  ({'USES thrust' if d > 1e-3 else 'DEAD feature'})")


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
    params = load_padded(args.init)
    loss_fn = make_loss_g(args.horizon)
    base_vg = jax.value_and_grad(
        lambda p, s1, r1, a1: loss_fn(p, s1[None], r1[None], a1[None]))
    vg_ep = jax.vmap(base_vg, in_axes=(None, 0, 0, 0))     # per-episode grads

    @jax.jit
    def train_step(params, opt, s0, rt, at, lr):
        B = s0.shape[0]
        losses, grads = vg_ep(params, s0, rt, at)
        grads = tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        losses = jnp.where(jnp.isfinite(losses), losses, 0.0)
        norms = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
                             for g in jax.tree_util.tree_leaves(grads)))
        cutoff = jnp.sort(norms)[B - args.trim_ep - 1]        # trim top-k monsters
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
    # DIAGONAL probe: success at each E1 thrust cell (the quantity H-A is about), so
    # best-save tracks band-wide skill, not a mixed batch biased toward the easy end.
    # a_thrust is a runtime arg to make_diag_g (not baked), so one jit per horizon is
    # reused across cells that share H (5e-3 & 2e-3 both H=120) — correct, unlike a
    # baked global.
    probe_orbits = {h: J.sample_orbits(random.PRNGKey(999_983), 512) for _, h in EVAL_THRUSTS}
    probe_diag = {h: jax.jit(make_diag_g(h)) for _, h in EVAL_THRUSTS}

    def diagonal(p):
        out = []
        for a_thrust, h in EVAL_THRUSTS:
            s0p, rtp = probe_orbits[h]
            atp = jnp.full((s0p.shape[0],), a_thrust)
            out.append(float(probe_diag[h](p, s0p, rtp, atp)[0]))
        return out

    best = None; best_m = -1.0
    for it in range(args.iters):
        k1, k2 = random.split(random.fold_in(key, it))
        s0, rt = J.sample_orbits(k1, args.batch)
        at = sample_thrust(k2, args.batch)
        lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * it / args.iters)))
        params, opt, l, gmax = train_step(params, opt, s0, rt, at, jnp.float32(lr))
        ema_p = tree_map(lambda a, b: tau * a + (1 - tau) * b, ema_p, params)
        if (it + 1) % args.eval_every == 0 or it == 0:
            succ = diagonal(ema_p)
            m = sum(succ) / len(succ)
            star = ""
            if m > best_m:
                best_m = m; best = tree_map(np.asarray, ema_p); star = " *"
            cells = " ".join(f"{100*s:.0f}" for s in succ)
            print(f"it={it:4d} loss={float(l):+.3f} diag_mean={100*m:.1f}% "
                  f"[{cells}]{star}", flush=True)
    save_npz(args.save, best)
    save_npz(args.save.replace(".npz", "_final.npz"), tree_map(np.asarray, ema_p))
    print(f"saved {args.save} (best diag_mean={100*best_m:.1f}%)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--distill", action="store_true",
                    help="R-A3: clone the 5 E1 specialists into one conditioned net")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--parity", action="store_true",
                    help="R-A1: padded 13->14 net at 5e-3 must ≈ the 13-input seed")
    ap.add_argument("--action-control", action="store_true")
    ap.add_argument("--distill-eps", type=int, default=256)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--init", type=str, default="models/warm_r28_ema_final.npz")
    ap.add_argument("--save", type=str, default="models/generalist.npz")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=480)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--trim-ep", type=int, default=16)
    ap.add_argument("--clip-ep", type=float, default=20000.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=31337)
    args = ap.parse_args()
    J.DV_BUDGET = 2.0; J.ABSORB = True; J.PHI_DV = True; J.D_EPS = 1e-4
    print(f"jax devices: {jax.devices()}", flush=True)
    if args.parity:
        params = load_padded(args.init)
        print("=== R-A1 parity: generalist@5e-3 vs seed ===")
        evaluate(params, 1024, args.eval_seed)
    if args.action_control:
        action_control(load_padded(args.init) if "generalist" not in args.init
                       else [(jnp.asarray(np.load(args.init)[f"w{i}"]),
                              jnp.asarray(np.load(args.init)[f"b{i}"])) for i in range(3)])
    if args.train:
        train(args)
    if args.distill:
        distill(args)
    if args.eval:
        d = np.load(args.init)
        params = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]
        evaluate(params, args.episodes, args.eval_seed)


if __name__ == "__main__":
    main()
