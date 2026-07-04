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
CHUNK = 128   # f64 x 200-substep decision scans OOM a 12GB card at batch 1024

state, rt = J.sample_orbits(random.PRNGKey(555_557), BATCH)
state = state.astype(jnp.float64)
rt = rt.astype(jnp.float64)

# Two impulsive baselines from the initial osculating ellipse (one tangential
# burn at apoapsis, coast there free):
#   dv_imp: EXACT circularization at rt = r_a(0)  (what "Hohmann" prices)
#   dv_box: cheapest SUCCESS-ADMISSIBLE terminal orbit — the tolerance-box
#           corner (|a'/rt-1|<=5%, e'<=0.05 -> ra/a' = 1.05, v' = sqrt(0.95
#           MU/ra)). Success tolerances make the box corner up to ~10-24%
#           cheaper than exact circularization (biggest for low-e0 starts), so
#           dvr<1 vs dv_imp is expected tolerance exploitation, NOT a beaten
#           baseline. Only dv below dv_box is anomalous in a two-body sim.
a0, e0 = J.elements(state)
ra0 = a0 * (1.0 + e0)
v_apo = jnp.sqrt(jnp.clip(J.MU * (2.0 / ra0 - 1.0 / a0), 1e-12, None))
v_circ = jnp.sqrt(J.MU / rt)
dv_imp = jnp.abs(v_circ - v_apo)
v_box = jnp.sqrt(0.95 * J.MU / ra0)
dv_box = jnp.abs(v_box - v_apo)

print(f"f64, dt={J.DT}s, {BATCH} fresh episodes; impulsive baselines "
      f"exact median={float(jnp.median(dv_imp)):.4f} km/s, "
      f"box-corner median={float(jnp.median(dv_box)):.4f} km/s "
      f"(box/exact median={float(jnp.median(dv_box / jnp.maximum(dv_imp, 1e-9))):.3f})",
      flush=True)


def rollout(mlp):
    step = jax.jit(lambda c, r: J._decision_step(mlp, c, r)[0])
    sts, dvs, crs, lts = [], [], [], []
    for i in range(0, BATCH, CHUNK):
        s_c, r_c = state[i:i + CHUNK], rt[i:i + CHUNK]
        B = s_c.shape[0]
        carry = (s_c, jnp.full((B,), J.DV_BUDGET, dtype=s_c.dtype),
                 jnp.zeros((B,), dtype=s_c.dtype), jnp.zeros((B,), dtype=s_c.dtype),
                 jnp.zeros((B,), bool))
        for _ in range(120):
            carry = step(carry, r_c)
        st, fuel, dv, crash, latch = carry
        sts.append(st); dvs.append(dv); crs.append(crash); lts.append(latch)
    return (jnp.concatenate(sts), jnp.concatenate(dvs),
            jnp.concatenate(crs), jnp.concatenate(lts))


for ckpt in sys.argv[1:]:
    d = np.load(ckpt)
    mlp = [(jnp.asarray(d[f"w{i}"], dtype=jnp.float64),
            jnp.asarray(d[f"b{i}"], dtype=jnp.float64)) for i in range(3)]
    st, dv, crash, latch = rollout(mlp)
    ae, e = J.a_err_e(st, rt)
    clean = latch & (crash == 0.0)
    n_latch = int(latch.sum()); n_clean = int(clean.sum())
    r_exact = (dv / jnp.maximum(dv_imp, 1e-6))[clean]
    r_box = (dv / jnp.maximum(dv_box, 1e-6))[clean]
    print(f"{ckpt:30s} success={float(latch.mean()):6.2%}  crashed-any="
          f"{float((crash > 0).mean()):.2%}  clean-latched={n_clean}", flush=True)
    if n_clean:
        print(f"{'':30s} vs EXACT circ: median={float(jnp.median(r_exact)):.4f} "
              f"mean={float(r_exact.mean()):.4f}  frac<1: {float((r_exact < 1).mean()):.1%}",
              flush=True)
        print(f"{'':30s} vs BOX corner: median={float(jnp.median(r_box)):.4f} "
              f"mean={float(r_box.mean()):.4f}  frac<1: {float((r_box < 1).mean()):.1%}"
              f"   (<1 here = anomalous, investigate)", flush=True)
