#!/usr/bin/env python3
"""Canonical checkpoint scoreboard — one binary, fixed eval sets.

Two anomalies force this: (a) R21's iter-0 eval of warm_r19.npz read 82.62%
where R19's in-run best line said 93.55%, on the same PRNGKey(999_983)/512
eval set — either the saved weights aren't the iter-900 weights or the two
runs' binaries eval differently; (b) every in-run "best" is a max over ~20
noisy evals (winner's curse). Score every checkpoint with the CURRENT binary
on the standard set AND a fresh 4096-episode set. The fresh-set number is
the citable one.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import jaxsim as J

J.DV_BUDGET = 2.0
J.ABSORB = True
J.ABSORB_CRASH = False
J.PHI_DV = True
J.D_EPS = 1e-4

diag = jax.jit(J.make_diag(120))   # matches in-run --eval-horizon default

sets = {
    "std512": J.sample_orbits(random.PRNGKey(999_983), 512),
    "fresh4096": J.sample_orbits(random.PRNGKey(31_337), 4096),
}

ckpts = sys.argv[1:] or [
    "models/dagger_jax.npz", "models/warm_r18b.npz", "models/warm_r19.npz",
    "models/warm_r19_final.npz", "models/warm_r21.npz", "models/warm_r21_final.npz",
]

for path in ckpts:
    d = np.load(path)
    mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]
    for name, (s, rt) in sets.items():
        suc, dv, ae, e, crash, dvr = (float(x) for x in diag(mlp, s, rt))
        print(f"{path:34s} {name:9s} success={suc*100:6.2f}%  dv={dv:.3f}  "
              f"a_err={ae:.3f}  e={e:.3f}  crash={crash*100:.1f}%  dvr={dvr:.3f}",
              flush=True)
