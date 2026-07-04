#!/usr/bin/env python3
"""Aim-low, terminate-at-true-box-entry: the operational split scheme.

aim_probe.py latches on the COMMANDED box, so low aims slide latches below
the true 0.95*rt floor and success craters with scatter. Deployment would
instead terminate the burn the moment the TRUE requirement is met. Same
policy, same physics: run with ABSORB off (latch is not observable, so
pre-entry behavior is identical), command rt_cmd = aim*rt, and record for
each episode the FIRST decision at which the state satisfies the true box
(|a/rt-1|<5%, e<0.05) plus fuel spent up to that entry. That is exactly the
split scheme's success/fuel accounting: aim at the cheap edge, stop when the
requirement is met.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import jaxsim as J

J.DV_BUDGET = 2.0
J.ABSORB = False          # latch must not freeze: we account first-entry ourselves
J.ABSORB_CRASH = False
J.PHI_DV = True
J.D_EPS = 1e-4

BATCH, CHUNK = 2048, 512
state, rt = J.sample_orbits(random.PRNGKey(424_251), BATCH)   # same batch as aim_probe

a0, e0 = J.elements(state)
ra0 = a0 * (1.0 + e0)
v_apo = jnp.sqrt(jnp.clip(J.MU * (2.0 / ra0 - 1.0 / a0), 1e-12, None))
dv_exact = jnp.abs(jnp.sqrt(J.MU / rt) - v_apo)
dv_box = jnp.abs(jnp.sqrt(0.95 * J.MU / ra0) - v_apo)


def first_entry(mlp, rt_cmd):
    step = jax.jit(lambda c, r: J._decision_step(mlp, c, r)[0])
    ent_all, dve_all, crash_all = [], [], []
    for i in range(0, BATCH, CHUNK):
        s_c, rc, rtrue = state[i:i + CHUNK], rt_cmd[i:i + CHUNK], rt[i:i + CHUNK]
        B = s_c.shape[0]
        carry = (s_c, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        entered = jnp.zeros((B,), bool)
        dv_entry = jnp.full((B,), jnp.nan)
        for _ in range(120):
            carry = step(carry, rc)
            st, fuel, dv, crash, _ = carry
            ae_t, e_t = J.a_err_e(st, rtrue)
            inside = (ae_t < 0.05) & (e_t < 0.05) & (crash == 0.0)
            newly = inside & (~entered)
            dv_entry = jnp.where(newly, dv, dv_entry)
            entered = entered | inside
        ent_all.append(entered); dve_all.append(dv_entry); crash_all.append(carry[3])
    return (jnp.concatenate(ent_all), jnp.concatenate(dve_all),
            jnp.concatenate(crash_all))


for ckpt in sys.argv[1:]:
    d = np.load(ckpt)
    mlp = [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]
    print(f"--- {ckpt}  (terminate at true-box entry)", flush=True)
    for aim in (1.0, 0.98, 0.97, 0.96, 0.95, 0.94):
        entered, dv_entry, crash = first_entry(mlp, rt * aim)
        ok = entered & jnp.isfinite(dv_entry)
        n = int(ok.sum())
        if n:
            rex = (dv_entry / jnp.maximum(dv_exact, 1e-6))[ok]
            rbx = (dv_entry / jnp.maximum(dv_box, 1e-6))[ok]
            print(f"  aim={aim:.2f}  entered-true-box={float(ok.mean()):6.2%}  "
                  f"dv/exact med={float(jnp.median(rex)):.4f} mean={float(rex.mean()):.4f}  "
                  f"dv/box med={float(jnp.median(rbx)):.4f}  frac<1(exact): "
                  f"{float((rex < 1).mean()):.1%}", flush=True)
        else:
            print(f"  aim={aim:.2f}  entered-true-box= 0.00%", flush=True)
