#!/usr/bin/env python3
"""Per-episode gradient-norm distribution at a checkpoint.

R17 set --clip-ep 1.0 blind and cratered FASTER than no-clip (arm B improved to
84.57% first): if healthy episodes routinely carry norms >> 1, clip=1.0
unit-normalizes everything and destroys the magnitude structure. Measure the
distribution, then clip at ~p99 so only the heavy tail is cut.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, vmap
import jaxsim as J

J.ABSORB = True
J.PHI_DV = True
J.D_EPS = 1e-4

ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/dagger_jax.npz"
d = np.load(ckpt)
mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]

det = J.make_loss_tbptt(60, K=10, w_dv=0.05, w_crash=5.0, w_shape=4.0)
loss1 = lambda p, s1, r1: det(p, s1[None], r1[None])
vg = jax.jit(vmap(value_and_grad(loss1), in_axes=(None, 0, 0)))

norms_all = []
for i in range(8):
    s, rt = J.sample_orbits(random.PRNGKey(100 + i), 256)
    _, g = vg(mlp, s, rt)
    n = np.sqrt(sum(np.asarray((gg.reshape(gg.shape[0], -1) ** 2).sum(1))
                    for gg in jax.tree_util.tree_leaves(g)))
    norms_all.append(n)
n = np.concatenate(norms_all)
n_fin = n[np.isfinite(n)]
print(f"ckpt={ckpt}  episodes={n.size}  non-finite={np.size(n)-n_fin.size}")
for q in (50, 90, 95, 99, 99.9):
    print(f"  p{q:<5} {np.percentile(n_fin, q):.3e}")
print(f"  max   {n_fin.max():.3e}")
for thr in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5):
    print(f"  frac > {thr:g}: {(n_fin > thr).mean():.2%}")
