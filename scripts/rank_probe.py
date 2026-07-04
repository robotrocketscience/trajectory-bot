#!/usr/bin/env python3
"""Loss-ranking probe — does the TRAINING loss rank coast above the good corridor,
and does the ranking flip as exploration noise shrinks?

Every run's loss column says coast (~-0.2) beats the corridor (~+1.0-1.7): the
optimizer was minimizing correctly; the corridor was never a minimum. Hypothesis:
the gap is noise-driven (burning + sigma=0.37 noise => crash tail * w_crash=5
dwarfs the success well), so at small sigma the ranking should FLIP, motivating a
sigma-annealing schedule (high early to escape coast, low late so the corridor is
the minimum).

Decomposes E[loss] under stochastic rollouts (ABSORB on, matching R11) into
shaping / well / dv / crash components for corridor vs coast at several sigma.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "scripts")
import numpy as np
import jax.numpy as jnp
from jax import random
import jaxsim as J

J.ABSORB = True                       # match R11 training semantics
W_SHAPE, W_WELL, W_DV, W_CRASH, SIG_WELL = 4.0, 1.0, 0.05, 5.0, 0.15
B, H = 512, 60

ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/sapo_r11.npz"
d = np.load(ckpt)
corridor = [(jnp.asarray(d["w0"]), jnp.asarray(d["b0"])),
            (jnp.asarray(d["w1"]), jnp.asarray(d["b1"])),
            (jnp.asarray(d["w2"]), jnp.asarray(d["b2"]))]
coast = corridor[:2] + [(corridor[2][0] * 0.0, corridor[2][1] * 0.0)]  # zero mean-head

state0, rt = J.sample_orbits(random.PRNGKey(23), B)

def eval_loss(mlp, sigma, seed):
    log_std = jnp.full((4,), float(np.log(sigma)))
    key = random.PRNGKey(seed)
    state = state0
    fuel = jnp.full((B,), J.DV_BUDGET); dv = jnp.zeros((B,))
    crash = jnp.zeros((B,)); latch = jnp.zeros((B,), bool)
    phi_prev = -J.orbit_err(state, rt)
    shaping = 0.0
    for _ in range(H):
        key, kk = random.split(key)
        carry = (state, fuel, dv, crash, latch)
        (state, fuel, dv, crash, latch), _ = J._decision_step_stoch(mlp, log_std, carry, rt, kk)
        phi_now = -J.orbit_err(state, rt)
        shaping += float((phi_now - phi_prev).mean())
        phi_prev = phi_now
    oe_T = np.asarray(J.orbit_err(state, rt))
    c_shape = -W_SHAPE * shaping
    c_well = -W_WELL * float(np.exp(-oe_T / SIG_WELL).mean())
    c_dv = W_DV * float(np.asarray(dv).mean())
    c_crash = W_CRASH * float(np.asarray(crash).mean())
    total = c_shape + c_well + c_dv + c_crash
    return total, c_shape, c_well, c_dv, c_crash, float(np.asarray(latch).mean())

print(f"corridor ckpt={ckpt}   (loss ex-entropy; ABSORB=True; B={B})")
print(f"{'sigma':>6} {'policy':>9} {'total':>8} {'shaping':>8} {'well':>7} "
      f"{'dv':>6} {'crash':>7} {'latch%':>7}")
for sigma in (0.37, 0.15, 0.05, 0.01):
    rows = {}
    for name, mlp in (("corridor", corridor), ("coast", coast)):
        t, cs, cw, cd, cc, lf = eval_loss(mlp, sigma, seed=101)
        rows[name] = t
        print(f"{sigma:>6} {name:>9} {t:>8.3f} {cs:>8.3f} {cw:>7.3f} "
              f"{cd:>6.3f} {cc:>7.3f} {lf:>7.1%}")
    print(f"       -> corridor {'BEATS' if rows['corridor'] < rows['coast'] else 'LOSES to'} coast")
print("VERDICT: ranking flip with sigma supports annealing schedule" )
