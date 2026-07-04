#!/usr/bin/env python3
"""Per-episode gradient-norm distribution at checkpoints (K=60 loss path).

Informs aggregation design (R17 lesson: measure before setting robust-stats
knobs). Key question after R21/R22 (same seed, trim-recovers vs clip-dies):
what fraction of episodes does clip-ep=100 flatten at each checkpoint? If a
large share of HEALTHY episodes sit above 100, the clip equalizes them with
monsters and the aggregate direction is no longer descent-like.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, value_and_grad, vmap
import jaxsim as J

J.DV_BUDGET = 2.0
J.ABSORB = True
J.ABSORB_CRASH = False
J.PHI_DV = True
J.D_EPS = 1e-4

det = J.make_loss(60, w_orbit=4.0, w_dv=0.05, w_crash=5.0)   # chunk=60 path, run parity
loss1 = lambda p, s1, r1: det(p, s1[None], r1[None])
vg = jax.jit(vmap(value_and_grad(loss1), in_axes=(None, 0, 0)))

for ckpt in (sys.argv[1:] or ["models/dagger_jax.npz"]):
    d = np.load(ckpt)
    mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]
    norms_all = []
    for i in range(8):
        s, rt = J.sample_orbits(random.PRNGKey(100 + i), 256)
        _, g = vg(mlp, s, rt)
        n = np.sqrt(sum(np.asarray((gg.reshape(gg.shape[0], -1) ** 2).sum(1))
                        for gg in jax.tree_util.tree_leaves(g)))
        norms_all.append(n)
    n = np.concatenate(norms_all)
    fin = n[np.isfinite(n)]
    qs = "  ".join(f"p{q}={np.percentile(fin, q):.2e}" for q in (50, 90, 95, 99))
    print(f"{ckpt:26s} nonfin={n.size - fin.size:3d}  {qs}  max={fin.max():.2e}", flush=True)
    print(f"{'':26s} frac>100: {(fin > 100).mean():6.2%}   frac>1e3: {(fin > 1e3).mean():6.2%}"
          f"   frac>1e6: {(fin > 1e6).mean():6.2%}", flush=True)
