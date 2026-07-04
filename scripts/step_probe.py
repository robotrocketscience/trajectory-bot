#!/usr/bin/env python3
"""One-Adam-step success sensitivity at a checkpoint.

R21's iter-0 line evaled warm_r19.npz at 82.62% while the file itself scores
93.55% on the same eval set — and the loop updates BEFORE the iter-0 eval, so
the line is warm_r19 + exactly one lr=5e-5 step. Adam's first step is
elementwise-bounded at lr, so an 11pp drop from one step means success is
knife-edge sensitive to parameter jitter. Measure it: N independent one-step
trials (different batch draws), report the Δsuccess distribution.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad
import jaxsim as J

J.DV_BUDGET = 2.0
J.ABSORB = True
J.ABSORB_CRASH = False
J.PHI_DV = True
J.D_EPS = 1e-4

ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/warm_r19.npz"
lr = 5e-5
trim_ep = 5

d = np.load(ckpt)
mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]

diag = jax.jit(J.make_diag(120))
eval_state, eval_rt = J.sample_orbits(random.PRNGKey(999_983), 512)
s0 = float(diag(mlp, eval_state, eval_rt)[0])
print(f"ckpt={ckpt}  base success={s0:.2%}", flush=True)

# R21's loss config: chunk=60 == horizon=60 -> make_loss path, default weights
det = J.make_loss(60, w_orbit=4.0, w_dv=0.05, w_crash=5.0)
base_vg = value_and_grad(lambda p, s1, r1: det(p, s1[None], r1[None]))
vg_ep = jax.jit(jax.vmap(base_vg, in_axes=(None, 0, 0)))


def one_step(params, ks):
    state, rt = J.sample_orbits(ks, 256)
    losses, grads = vg_ep(params, state, rt)
    grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    norms = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
                         for g in jax.tree_util.tree_leaves(grads)))
    cutoff = jnp.sort(norms)[256 - trim_ep - 1]
    scale = (norms <= cutoff).astype(jnp.float32)
    kept = jnp.maximum(scale.sum(), 1.0)
    grads = jax.tree_util.tree_map(
        lambda g: jnp.sum(g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), axis=0) / kept,
        grads)
    opt = J.adam_init(params)
    new_p, _ = J.adam_step(params, grads, opt, lr=lr)
    return new_p


key = random.PRNGKey(0)
deltas = []
for i in range(8):
    key, ks = random.split(key)
    p1 = one_step(mlp, ks)
    s1 = float(diag(p1, eval_state, eval_rt)[0])
    deltas.append(s1 - s0)
    print(f"  trial {i}: after 1 step success={s1:.2%}  delta={100*(s1-s0):+.2f}pp",
          flush=True)
deltas = np.array(deltas)
print(f"one-step |delta|: mean={100*np.abs(deltas).mean():.2f}pp  "
      f"max={100*np.abs(deltas).max():.2f}pp", flush=True)
