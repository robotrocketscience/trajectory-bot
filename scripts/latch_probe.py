#!/usr/bin/env python3
"""Latch-corruption probe — is success absorbing in the TRAINING loss?

The env terminates at success; the training rollout does not. With exploration
noise still firing post-latch, a policy that reaches tolerance mid-episode gets
knocked off target and the terminal well grades the drifted final state.

Measures, over stochastic (training-distribution) rollouts at a checkpoint:
  latched%      — episodes hitting (ae<tol & e<tol) at ANY decision
  final-in-tol% — episodes whose FINAL state is within tol (what the loss rewards)
  post-latch oe drift + fuel spend on latched episodes
A large latched% vs final-in-tol% gap = post-success corruption is real.
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
log_std = jnp.asarray(d["log_std"]) if "log_std" in d else jnp.full((4,), -1.0)
print(f"ckpt={ckpt}  budget={J.DV_BUDGET}  sigma={np.exp(np.asarray(log_std))}")

B, H = 1024, 60
key = random.PRNGKey(5)
state, rt = J.sample_orbits(random.PRNGKey(17), B)
fuel = jnp.full((B,), J.DV_BUDGET)
dv = jnp.zeros((B,)); crash = jnp.zeros((B,)); latch = jnp.zeros((B,), bool)

latch_at = np.full(B, -1)          # decision index of first latch
oe_at_latch = np.zeros(B)
fuel_at_latch = np.zeros(B)
for t in range(H):
    key, kk = random.split(key)
    carry = (state, fuel, dv, crash, latch)
    (state, fuel, dv, crash, latch), _ = J._decision_step_stoch(mlp, log_std, carry, rt, kk)
    l = np.asarray(latch); newly = l & (latch_at < 0)
    if newly.any():
        oe = np.asarray(J.orbit_err(state, rt))
        f = np.asarray(fuel)
        latch_at[newly] = t
        oe_at_latch[newly] = oe[newly]
        fuel_at_latch[newly] = f[newly]

ae_f, e_f = (np.asarray(x) for x in J.a_err_e(state, rt))
oe_f = np.asarray(J.orbit_err(state, rt))
fuel_f = np.asarray(fuel)
L = latch_at >= 0
final_ok = (ae_f < J.A_TOL) & (e_f < J.E_TOL)
print(f"latched at any point: {L.mean():.1%}   final state in tol: {final_ok.mean():.1%}")
print(f"  -> corruption gap: {(L & ~final_ok).sum()} of {L.sum()} latched episodes "
      f"({(L & ~final_ok).mean() / max(L.mean(), 1e-9):.0%}) end OUT of tolerance")
if L.any():
    print(f"latched episodes: median latch decision {np.median(latch_at[L]):.0f}/{H}, "
          f"oe at latch {oe_at_latch[L].mean():.3f} -> final oe {oe_f[L].mean():.3f}")
    print(f"  post-latch fuel spend: {(fuel_at_latch[L] - fuel_f[L]).mean():.2f} km/s "
          f"(fuel at latch {fuel_at_latch[L].mean():.2f})")
print("VERDICT:", "POST-LATCH CORRUPTION MATERIAL" if L.any()
      and (L & ~final_ok).sum() / max(L.sum(), 1) > 0.5 else "corruption minor")
