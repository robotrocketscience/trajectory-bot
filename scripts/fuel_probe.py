#!/usr/bin/env python3
"""R9e-probe — discriminate H-fuel-starvation vs K-myopia for the e-wall.

Runs STOCHASTIC rollouts (the training distribution) at a saved checkpoint and
measures when the fuel gate kills the episode's gradient signal.
Predict (if H-fuel-starvation): median exhaustion <= ~decision 15 of 60, i.e.
most of the episode is action-dead in training rollouts.
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
    J.DV_BUDGET = float(sys.argv[2])   # must match the ckpt's TRAINING budget
d = np.load(ckpt)
mlp = [(jnp.asarray(d["w0"]), jnp.asarray(d["b0"])),
       (jnp.asarray(d["w1"]), jnp.asarray(d["b1"])),
       (jnp.asarray(d["w2"]), jnp.asarray(d["b2"]))]
log_std = jnp.asarray(d["log_std"]) if "log_std" in d else jnp.full((4,), -1.0)
print(f"ckpt={ckpt}  sigma={np.exp(np.asarray(log_std))}")

B, H = 256, 60
key = random.PRNGKey(7)
state, rt = J.sample_orbits(random.PRNGKey(11), B)
fuel = jnp.full((B,), J.DV_BUDGET)
dv = jnp.zeros((B,)); crash = jnp.zeros((B,)); latch = jnp.zeros((B,), bool)

exhaust_at = np.full(B, H, dtype=int)     # decision index when fuel first <= 0
live = 0
thr_sum = 0.0
for t in range(H):
    key, kk = random.split(key)
    carry = (state, fuel, dv, crash, latch)
    (state, fuel, dv, crash, latch), _ = J._decision_step_stoch(mlp, log_std, carry, rt, kk)
    f = np.asarray(fuel)
    newly = (f <= 0) & (exhaust_at == H)
    exhaust_at[newly] = t
    live += int((f > 0).sum())

frac_live = live / (B * H)
print(f"median fuel-exhaustion decision: {np.median(exhaust_at):.0f} / {H}")
print(f"fraction of (episode x decision) with fuel remaining: {frac_live:.1%}")
print(f"episodes exhausted before decision 20: {(exhaust_at < 20).mean():.1%}")
print(f"episodes never exhausted: {(exhaust_at == H).mean():.1%}")
print(f"mean dv spent: {float(dv.mean()):.2f} / budget {J.DV_BUDGET}")
verdict = "FUEL-STARVATION CONFIRMED" if np.median(exhaust_at) <= 20 else "fuel lasts — K-myopia stands"
print("VERDICT:", verdict)
