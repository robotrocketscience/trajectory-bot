#!/usr/bin/env python3
"""Aim-scaling sweep: command a scaled target radius, score against the true box.

The f64 verification put the best-fuel checkpoint at 1.03x the exact-
circularization optimum, with the admissible box-corner bound at 0.849x —
and the policies aim at the box CENTER because that is what the objective
asked for. But the policy conditions on rt through its observations (all
obs are rt-normalized), so corner-finishing may need no training at all:
command rt_cmd = aim * rt, let the policy latch on its commanded box, then
score the final state against the TRUE rt box and price the fuel against
the TRUE baselines. Sweep aim over [1.0 .. 0.92] for each checkpoint; the
operational optimum of this scheme is wherever true-success x fuel-savings
is acceptable. (Latching stays tied to the commanded box — states latched
below 0.95*rt_true score as failures, so the trade shows up honestly.)
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

BATCH, CHUNK = 2048, 512
state, rt = J.sample_orbits(random.PRNGKey(424_251), BATCH)

a0, e0 = J.elements(state)
ra0 = a0 * (1.0 + e0)
v_apo = jnp.sqrt(jnp.clip(J.MU * (2.0 / ra0 - 1.0 / a0), 1e-12, None))
dv_exact = jnp.abs(jnp.sqrt(J.MU / rt) - v_apo)
dv_box = jnp.abs(jnp.sqrt(0.95 * J.MU / ra0) - v_apo)


def rollout(mlp, rt_cmd):
    step = jax.jit(lambda c, r: J._decision_step(mlp, c, r)[0])
    outs = []
    for i in range(0, BATCH, CHUNK):
        s_c, r_c = state[i:i + CHUNK], rt_cmd[i:i + CHUNK]
        B = s_c.shape[0]
        carry = (s_c, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        for _ in range(120):
            carry = step(carry, r_c)
        outs.append(carry)
    st = jnp.concatenate([o[0] for o in outs])
    dv = jnp.concatenate([o[2] for o in outs])
    crash = jnp.concatenate([o[3] for o in outs])
    latch = jnp.concatenate([o[4] for o in outs])
    return st, dv, crash, latch


for ckpt in sys.argv[1:]:
    d = np.load(ckpt)
    mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]
    print(f"--- {ckpt}", flush=True)
    for aim in (1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92):
        st, dv, crash, latch = rollout(mlp, rt * aim)
        ae_true, e_true = J.a_err_e(st, rt)
        ok = (ae_true < 0.05) & (e_true < 0.05) & (crash == 0.0)   # TRUE box
        n = int(ok.sum())
        if n:
            rex = (dv / jnp.maximum(dv_exact, 1e-6))[ok]
            rbx = (dv / jnp.maximum(dv_box, 1e-6))[ok]
            print(f"  aim={aim:.2f}  true-success={float(ok.mean()):6.2%}  "
                  f"(cmd-latch {float(latch.mean()):5.1%})  "
                  f"dv/exact med={float(jnp.median(rex)):.4f}  "
                  f"dv/box med={float(jnp.median(rbx)):.4f}", flush=True)
        else:
            print(f"  aim={aim:.2f}  true-success= 0.00%", flush=True)
