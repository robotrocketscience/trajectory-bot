#!/usr/bin/env python3
"""Phase-blindness probe — is the learned policy's burn decision phase-aware?

Claim under test (R10 narrative): the trained policy fixes `a` with phase-blind
burning, which random-walks e. If true, its throttle output should be ~independent
of orbital phase (apoapsis half vs periapsis half). The apoapsis-half indicator is
linearly decodable from its own inputs: (r/rt-1) - (a/rt-1) > 0  <=>  r > a.

Also reports the expert-like signature we WANT: burn hard near apoapsis (to raise
periapsis / circularize), coast near periapsis.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "scripts")
import numpy as np
import jax.numpy as jnp
from jax import random
import jaxsim as J

ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/sapo_r9d.npz"
if len(sys.argv) > 2:
    J.DV_BUDGET = float(sys.argv[2])
d = np.load(ckpt)
mlp = [(jnp.asarray(d["w0"]), jnp.asarray(d["b0"])),
       (jnp.asarray(d["w1"]), jnp.asarray(d["b1"])),
       (jnp.asarray(d["w2"]), jnp.asarray(d["b2"]))]

B = 4096
state, rt = J.sample_orbits(random.PRNGKey(3), B)
fuel = jnp.full((B,), J.DV_BUDGET)
obs = J.observe(state, rt, fuel)
act = np.asarray(J.policy(mlp, obs))
throttle = np.clip(act[:, 3], 0.0, 1.0)
tang = act[:, 0]                       # tangential (prograde) direction coeff

a, e = (np.asarray(x) for x in J.elements(state))
r = np.asarray(J.snorm(state[:, 0:3], axis=1))
apo_half = r > a                       # cos E < 0: nearer apoapsis
rv = (np.asarray(state[:, 0:3]) * np.asarray(state[:, 3:6])).sum(1)  # r.v sign: ascending?

def stat(mask, name):
    print(f"  {name:24s} n={mask.sum():5d}  throttle={throttle[mask].mean():.3f}"
          f" (sd {throttle[mask].std():.3f})  tang_coeff={tang[mask].mean():+.3f}")

print(f"ckpt={ckpt}  budget={J.DV_BUDGET}")
print(f"overall: throttle mean={throttle.mean():.3f}  frac(throttle>0.5)={(throttle>0.5).mean():.1%}")
stat(apo_half, "apoapsis half (r>a)")
stat(~apo_half, "periapsis half (r<a)")
stat(apo_half & (rv > 0), "apo half, ascending")
stat(apo_half & (rv < 0), "apo half, descending")
# quartiles of r/a as a finer phase proxy
q = np.quantile(r / a, [0.25, 0.5, 0.75])
labels = ["r/a Q1 (lowest)", "r/a Q2", "r/a Q3", "r/a Q4 (highest)"]
bins = np.digitize(r / a, q)
for i, lab in enumerate(labels):
    stat(bins == i, lab)
# phase-awareness score: |throttle(apo) - throttle(peri)| relative to pooled sd
delta = abs(throttle[apo_half].mean() - throttle[~apo_half].mean())
score = delta / max(throttle.std(), 1e-9)
print(f"phase-awareness score |d_throttle|/sd = {score:.2f}  "
      f"({'PHASE-AWARE' if score > 0.5 else 'PHASE-BLIND'} at 0.5 threshold)")
