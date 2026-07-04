#!/usr/bin/env python3
"""Numerical parity: JAX port vs torch reference on a FIXED state (one rk4 step + elements)."""
import sys, numpy as np, torch
sys.path.insert(0, "scripts")
import train_diffsim3d as t
import jaxsim as j
import jax.numpy as jnp

rng = np.random.default_rng(0)
st = np.zeros((4, 13), np.float64)
st[:, 0:3] = np.array([7000., 500., -300.])
st[:, 3:6] = np.array([0.3, 7.4, 0.8])
q = rng.standard_normal((4, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
st[:, 6:10] = q
st[:, 10:13] = rng.standard_normal((4, 3)) * 0.01
omega = rng.standard_normal((4, 3)) * 0.02
thr = np.array([0.5, 0.0, 1.0, 0.2])

st_t = torch.tensor(st, dtype=torch.float32)
rk_t = t.rk4(st_t, torch.tensor(omega, dtype=torch.float32),
             torch.tensor(thr, dtype=torch.float32)).detach().numpy()
a_t, e_t, _ = t.elements(st_t)

st_j = jnp.asarray(st, dtype=jnp.float32)
rk_j = np.asarray(j.rk4(st_j, jnp.asarray(omega, jnp.float32), jnp.asarray(thr, jnp.float32)))
a_j, e_j = j.elements(st_j)

print(f"rk4  max|Δ| = {np.abs(rk_t - rk_j).max():.3e}  (rel {np.abs(rk_t-rk_j).max()/ (np.abs(rk_t).max()+1e-9):.2e})")
print(f"a    max|Δ| = {np.abs(a_t.detach().numpy() - np.asarray(a_j)).max():.3e}")
print(f"e    max|Δ| = {np.abs(e_t.detach().numpy() - np.asarray(e_j)).max():.3e}")
ok = (np.abs(rk_t - rk_j).max() < 1.0) and (np.abs(e_t.detach().numpy() - np.asarray(e_j)).max() < 1e-3)
print("PARITY:", "OK" if ok else "MISMATCH")
