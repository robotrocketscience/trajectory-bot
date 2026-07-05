#!/usr/bin/env python3
"""Target-conditioned DAgger (JAX) — the R42 fix for the rt==r_a degeneracy.

The whole policy lineage bootstrapped from a FIXED-target scripted expert
(circularize at r_a only), so it never imitated target-tracking and every
checkpoint starts wrong-signed (slope -4.88 .. -2.0). This clones a
TARGET-CONDITIONED expert instead: a two-apse tangential controller that
drives BOTH apses to the commanded rt (prograde-default; retrograde only to
lower an apse; burn gated near apses). Measured standalone: slope +0.89,
band-mean 94% across [0.85,1.15]*r_a. DAgger it into jaxsim's 13->128->128->4
MLP across the jittered band, producing a target-conditioned init (slope ~+1)
for subsequent diff-sim refinement.

Expert is a pure function of the (rt-normalized) obs, so relabeling visited
states is exact. Output: models/dagger_target_jax.npz (jaxsim w/b format).

    uv run --with "jax[cuda12]" python scripts/dagger_target_jax.py \
        --iters 8 --episodes 512 --bc-epochs 40 --save models/dagger_target_jax.npz

Experiment code (excluded from the strict-typed library).
"""
import argparse
import numpy as np
import jax
from jax import lax, random
import jax.numpy as jnp
from jax.tree_util import tree_map
import sys
sys.path.insert(0, "scripts")
import jaxsim as J

J.DV_BUDGET = 2.0; J.ABSORB = True; J.ABSORB_CRASH = False; J.PHI_DV = True; J.D_EPS = 1e-4

K_EXPERT = 0.7
GATE_EXPERT = 0.10


def expert(obs):
    """Two-apse target-conditioned controller, from obs alone. slope +0.89 / 94%."""
    a_rt = obs[:, 6] + 1.0; e = obs[:, 7]; r_rt = obs[:, 8] + 1.0
    ra_rt = a_rt * (1.0 + e); rp_rt = a_rt * (1.0 - e)
    near_apo = r_rt > a_rt
    err = jnp.where(near_apo, 1.0 - rp_rt, 1.0 - ra_rt)   # +: apse below rt -> raise
    throttle = jnp.clip(K_EXPERT * jnp.abs(err), 0.0, 1.0)
    pos = obs[:, 0:3]; vel = obs[:, 3:6]
    rn = jnp.maximum(jnp.linalg.norm(pos, axis=1), 1e-6)
    rdot = (pos * vel).sum(1) / rn
    throttle = throttle * jnp.clip(1.0 - jnp.abs(rdot) / GATE_EXPERT, 0.0, 1.0)
    sign = jnp.where(err < 0.0, -1.0, 1.0)                # prograde default
    z = jnp.zeros_like(throttle)
    return jnp.stack([sign, z, z, throttle], axis=1)


POLICY = J.policy                 # jaxsim's real MLP forward, saved before any swap
expert_wrap = lambda params, obs: expert(obs)   # drop-in driver for _decision_step


def rollout_collect(policy_fn, params, s0, rt, H=120):
    """Drive jaxsim's exact _decision_step (via a J.policy swap) and record the
    visited obs. policy_fn(params, obs)->act is either POLICY or expert_wrap."""
    J.policy = policy_fn
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))

    def scanfn(carry, _):
        state, fuel = carry[0], carry[1]
        obs = J.observe(state, rt, jnp.clip(fuel, 0.0, None))
        newcarry, _ = J._decision_step(params, carry, rt)
        return newcarry, obs

    (state, fuel, dv, crash, latch), obs_seq = lax.scan(scanfn, carry, None, length=H)
    J.policy = POLICY
    return obs_seq.reshape(-1, 13), latch


def loss_fn(params, obs, tgt, wts):
    pred = J.policy(params, obs)
    per = ((pred - tgt) ** 2).mean(axis=1)
    return (wts * per).sum() / jnp.maximum(wts.sum(), 1.0)


@jax.jit
def adam_step(params, state, obs, tgt, wts, lr):
    m, v, t = state
    l, g = jax.value_and_grad(loss_fn)(params, obs, tgt, wts)
    t = t + 1.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
    v = tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
    bc = jnp.sqrt(1 - b2 ** t) / (1 - b1 ** t)
    params = tree_map(lambda p_, m_, v_: p_ - lr * bc * m_ / (jnp.sqrt(v_) + eps),
                      params, m, v)
    return params, (m, v, t)


def bc_fit(params, obs, tgt, epochs, key, batch=4096, lr=1e-3, burn_w=15.0):
    wts = 1.0 + burn_w * (tgt[:, 3] > 0).astype(jnp.float32)
    st = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), jnp.array(0.0))
    n = obs.shape[0]
    for ep in range(epochs):
        key, k = random.split(key)
        perm = random.permutation(k, n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            params, st = adam_step(params, st, obs[idx], tgt[idx], wts[idx], lr)
    return params


def band_eval(params):
    J.policy = POLICY
    s, ra = J.sample_orbits(random.PRNGKey(31_337), 1024); ra = jnp.asarray(ra)
    ratios = np.linspace(0.85, 1.15, 7)
    succ = []; afr = []
    for x in ratios:
        B = s.shape[0]
        carry = (s, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        def sf(c, _):
            c, _ = J._decision_step(params, c, ra * x); return c, None
        (fs, _, _, _, latch), _ = lax.scan(sf, carry, None, length=120)
        a, e = J.elements(fs)
        succ.append(float(jnp.mean(latch)) * 100)
        afr.append(float(jnp.mean(a / ra)))
    slope = np.polyfit(ratios, np.array(afr), 1)[0]
    return np.mean(succ), slope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--width", type=float, default=0.10)
    ap.add_argument("--save", type=str, default="models/dagger_target_jax.npz")
    args = ap.parse_args()

    key = random.PRNGKey(0)
    params = J.init_params(random.PRNGKey(1), final_scale=0.01)
    all_obs = None
    for it in range(args.iters):
        k = random.fold_in(key, it)
        s0, rt = J.sample_orbits(k, args.episodes, rt_jitter=args.width)
        if it == 0:
            obs_flat, _ = rollout_collect(expert_wrap, None, s0, rt)
        else:
            obs_flat, _ = rollout_collect(POLICY, params, s0, rt)
        all_obs = obs_flat if all_obs is None else jnp.concatenate([all_obs, obs_flat])
        tgt = expert(all_obs)
        params = bc_fit(params, all_obs, tgt, args.bc_epochs, random.fold_in(key, 1000 + it))
        succ, slope = band_eval(params)
        print(f"dagger it={it}  data={all_obs.shape[0]:>7d}  band-succ={succ:5.1f}%  slope={slope:+.2f}",
              flush=True)

    d = {}
    for i, (w, b) in enumerate(params):
        d[f"w{i}"] = np.asarray(w); d[f"b{i}"] = np.asarray(b)
    np.savez(args.save, **d)
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
