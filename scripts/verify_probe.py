#!/usr/bin/env python3
"""High-fidelity verification of dvr<1 ("beats the analytic transfer") claims.

Training/eval fidelity is float32, RK4 at dt=10 s. Before believing any
sub-analytic fuel number, re-fly the policy in float64 at dt=1 s (decision
cadence preserved: 200 substeps x 1 s per decision instead of 20 x 10 s) and
compare spent dv on latched episodes against the CLOSED-FORM impulsive
optimum for this task (ellipse -> circular at its own apoapsis radius =
single apoapsis circularization burn: dv = v_circ(ra) - v_apo), not just the
smooth two-burn estimate the reward uses. The DAgger expert is the control:
it flies the analytic maneuver, so its ratio should be ~1.0; a policy ratio
meaningfully below 1.0 that survives this probe is a real finding (finite
burns carry gravity losses, so beating impulsive is a high bar). Episodes
with any crash accrual are excluded (r-floor clamp region = fake dynamics).

Usage: verify_probe.py ckpt1.npz [ckpt2.npz ...]
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random
import jaxsim as J

J.DT = 1.0            # 10x finer integration,
J.REPEAT = 200        # same 200 s decision period -> same MDP for the policy
J.DV_BUDGET = 2.0
J.ABSORB = True
J.ABSORB_CRASH = False
J.PHI_DV = True
J.D_EPS = 1e-4

BATCH = 1024
diag = jax.jit(J.make_diag(120))

state, rt = J.sample_orbits(random.PRNGKey(555_557), BATCH)
state = state.astype(jnp.float64)
rt = rt.astype(jnp.float64)

# closed-form impulsive optimum from the initial osculating ellipse to a
# circular orbit at rt = r_a(0): one tangential burn at apoapsis
a0, e0 = J.elements(state)
ra0 = a0 * (1.0 + e0)
v_apo = jnp.sqrt(jnp.clip(J.MU * (2.0 / ra0 - 1.0 / a0), 1e-12, None))
v_circ = jnp.sqrt(J.MU / rt)
dv_imp = jnp.abs(v_circ - v_apo)

print(f"f64, dt={J.DT}s, {BATCH} fresh episodes; impulsive baseline "
      f"median={float(jnp.median(dv_imp)):.4f} km/s", flush=True)


def rollout(mlp):
    B = state.shape[0]
    carry = (state, jnp.full((B,), J.DV_BUDGET, dtype=state.dtype),
             jnp.zeros((B,), dtype=state.dtype), jnp.zeros((B,), dtype=state.dtype),
             jnp.zeros((B,), bool))
    step = jax.jit(lambda c: J._decision_step(mlp, c, rt))
    for _ in range(120):
        carry, _ = step(carry)
    st, fuel, dv, crash, latch = carry
    return st, dv, crash, latch


for ckpt in sys.argv[1:]:
    d = np.load(ckpt)
    mlp = [(jnp.asarray(d[f"w{i}"], dtype=jnp.float64),
            jnp.asarray(d[f"b{i}"], dtype=jnp.float64)) for i in range(3)]
    st, dv, crash, latch = rollout(mlp)
    ae, e = J.a_err_e(st, rt)
    clean = latch & (crash == 0.0)
    n_latch = int(latch.sum()); n_clean = int(clean.sum())
    ratio = dv / jnp.maximum(dv_imp, 1e-6)
    r_clean = ratio[clean]
    below = float((r_clean < 1.0).mean()) if n_clean else float("nan")
    print(f"{ckpt:30s} success={float(latch.mean()):6.2%}  crashed-any="
          f"{float((crash > 0).mean()):.2%}  clean-latched={n_clean}", flush=True)
    if n_clean:
        print(f"{'':30s} dv/dv_impulsive on clean latches: "
              f"median={float(jnp.median(r_clean)):.4f}  mean={float(r_clean.mean()):.4f}  "
              f"p10={float(jnp.percentile(r_clean, 10)):.4f}  frac<1: {below:.1%}",
              flush=True)
