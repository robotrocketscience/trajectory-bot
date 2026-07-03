#!/usr/bin/env python3
"""JAX/XLA port of the 3D decision-layer diff-sim (Circularize-3D hot path).

The spike measured ~50x over fused-torch (0.265 vs 13.3 s/iter, H=60/B=256, 3060)
because XLA fuses the 900-substep lax.scan rollout into a few kernels, killing the
launch-bound overhead diagnosed in R1. This module is the validated port:

  * physics/quaternion/controller identical to scripts/train_diffsim3d.py
  * rollout as nested lax.scan, training via jit(value_and_grad) + manual Adam
  * on-device latched success eval (mirrors the env's early-termination success)
  * FIXED objective: orbit-dominant, Δv tiebreaker, Ng-1999 potential shaping,
    smooth success well, and a FRACTIONAL crash penalty relu((R-r)/R)^2 (O(1)) so
    the loss dynamic range is sane (torch B2 exploded to ~1e6 with a km^2 penalty).

Experiment code (excluded from the strict-typed library).

  uv run --with "jax[cuda12]" python scripts/jaxsim.py --iters 300 --eval-every 25
"""
from __future__ import annotations
import argparse, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax, jit, value_and_grad, random

# --- constants (match tbot.orbital / Circularize3DConfig exactly) ---
MU = 398600.4418; R_BODY = 6378.137
DT = 10.0; REPEAT = 20
A_THRUST = 5e-3; RATE_GAIN = 0.1; K_P = 0.5; MAX_RATE = 0.05; DV_BUDGET = 2.0
# --absorb: success is absorbing in the rollout (env-style termination). Without it
# the loss grades the FINAL state of a full 60-decision rollout, so an episode that
# reaches tolerance mid-episode is dragged off target by exploration noise (latch
# probe at R9d ckpt: 78% of latched episodes end OUT of tolerance, +0.73 km/s
# wasted post-latch) — there is no loss basin around success.
ABSORB = False
# --e-weight: lambda on e in the loss potential. Rank probe at the R11 ckpt showed
# the average episode WORSENS oe deterministically (a improves, e pumps up — the
# policy accepts the a-for-e trade at lambda=1); lambda>1 reprices that trade.
E_WEIGHT = 1.0
ALT_PERI = (400.0, 800.0); RA_RP = (1.3, 2.5); INC_MAX = np.radians(40.0)
E_TOL = 0.05; A_TOL = 0.05
A_MAX = 50.0 * R_BODY          # semimajor-axis ceiling: keeps `a` finite & differentiable
HID = 128                      # (escape/parabolic -> energy~0 -> a=inf -> NaN grad through clip)


def cross(a, b): return jnp.cross(a, b, axis=-1)


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(q): return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def qrotate(q, v):                       # fused cross-product identity
    w = q[..., 0:1]; u = q[..., 1:4]
    t = 2.0 * cross(u, v)
    return v + w * t + cross(u, t)


def snorm(x, axis=-1, keepdims=False, eps=1e-12):
    """Safe Euclidean norm: sqrt(Σx²+ε). Unlike jnp.linalg.norm, its gradient is
    finite at x=0 (norm's is x/|x| = 0/0 = NaN there — and the pointing controller
    and circular-orbit target drive exactly to those zeros)."""
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + eps)


def qnorm(q):
    return q / snorm(q, axis=-1, keepdims=True)


# --- functional MLP policy (13 -> 128 -> 128 -> 4, tanh) ---
def init_params(key, final_scale=1.0):
    """final_scale=0.01 puts the untrained MEAN policy near coast (throttle≈0).
    R9b diagnostics showed final_scale=1 random-inits into a saturated random-burn
    regime (dv=budget, crash≈50% at iter 0) — a different failure mode than torch's
    near-coast default init, and one the gradient never organizes out of."""
    ks = random.split(key, 3)
    def layer(k, nin, nout, scale=1.0):
        return (random.normal(k, (nin, nout)) * (scale / np.sqrt(nin)),
                jnp.zeros((nout,)))
    return [layer(ks[0], 13, HID), layer(ks[1], HID, HID),
            layer(ks[2], HID, 4, scale=final_scale)]


def policy(params, obs):
    x = obs
    for (w, b) in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return jnp.tanh(x @ w + b)


# --- dynamics (mirror train_diffsim3d) ---
def deriv(state, omega_cmd, throttle):
    r = state[:, 0:3]; v = state[:, 3:6]; q = state[:, 6:10]; w = state[:, 10:13]
    # floor at the surface: a crashed sat's radius can collapse toward 0, and a 1 km
    # floor gave grav ~4e5 km/s^2 -> huge intermediates -> NaN gradient over long
    # rollouts. Valid orbits (r > R_BODY+400) are unaffected (parity preserved).
    rmag = jnp.maximum(snorm(r, axis=1, keepdims=True), R_BODY)
    grav = -MU * r / rmag ** 3
    b_hat = jnp.zeros_like(v).at[:, 0].set(1.0)
    tdir = qrotate(q, b_hat)
    acc = grav + throttle[:, None] * A_THRUST * tdir
    z = jnp.zeros((w.shape[0], 1))
    qdot = 0.5 * qmul(q, jnp.concatenate([z, w], axis=1))
    wdot = RATE_GAIN * (omega_cmd - w)
    return jnp.concatenate([v, acc, qdot, wdot], axis=1)


def rk4(state, omega_cmd, throttle):
    k1 = deriv(state, omega_cmd, throttle)
    k2 = deriv(state + 0.5 * DT * k1, omega_cmd, throttle)
    k3 = deriv(state + 0.5 * DT * k2, omega_cmd, throttle)
    k4 = deriv(state + DT * k3, omega_cmd, throttle)
    s = state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    q = qnorm(s[:, 6:10])
    return jnp.concatenate([s[:, 0:6], q, s[:, 10:13]], axis=1)


def elements(state):
    r_vec = state[:, 0:3]; v_vec = state[:, 3:6]
    r = jnp.maximum(snorm(r_vec, axis=1), R_BODY)      # surface floor (see deriv)
    v = snorm(v_vec, axis=1)
    energy = 0.5 * v ** 2 - MU / r
    # clamp strictly negative so a stays finite/differentiable even for escape
    # trajectories (energy>=0). An escaping orbit saturates a -> A_MAX (large penalty).
    energy_safe = jnp.minimum(energy, -MU / (2.0 * A_MAX))
    a = -MU / (2.0 * energy_safe)
    rv = (r_vec * v_vec).sum(axis=1)
    factor = (v ** 2 - MU / r)[:, None]
    e_vec = (factor * r_vec - rv[:, None] * v_vec) / MU
    e = snorm(e_vec, axis=1)                    # -> 0 at circular target (safe grad)
    return a, e


def a_err_e(state, rt):
    a, e = elements(state)
    return jnp.abs(a - rt) / rt, e


def orbit_err(state, rt):
    ae, e = a_err_e(state, rt)
    # Escape term: the energy clamp in elements() makes a (and thus ae) gradient-dead
    # once a would exceed A_MAX, and the ae clip below kills it past 5.0 — R10 showed
    # the optimizer parks on that plateau (mean e>1.4 for 1100 iters, no way back).
    # relu(energy/(MU/2A_MAX) + 1) is 0 for all orbits below the clamp and grows
    # linearly in energy beyond it — smooth in (r, v), so the plateau has a slope home.
    r = jnp.maximum(snorm(state[:, 0:3], axis=1), R_BODY)
    v = snorm(state[:, 3:6], axis=1)
    energy = 0.5 * v ** 2 - MU / r
    esc = jnp.clip(energy * (2.0 * A_MAX / MU) + 1.0, 0.0, None)
    return jnp.clip(ae, None, 5.0) + E_WEIGHT * jnp.clip(e, None, 2.0) + esc


def orbit_frame(r, v):
    t = v / snorm(v, axis=1, keepdims=True)
    h = cross(r, v)
    w = h / snorm(h, axis=1, keepdims=True)
    return t, w, cross(t, w)


def observe(state, rt, fuel):
    a, e = elements(state)
    r = snorm(state[:, 0:3], axis=1)
    L = rt[:, None]; V = jnp.sqrt(MU / rt)[:, None]
    b_in = qrotate(state[:, 6:10], jnp.tile(jnp.array([1.0, 0, 0]), (state.shape[0], 1)))
    t, w, s = orbit_frame(state[:, 0:3], state[:, 3:6])
    o = jnp.concatenate([
        state[:, 0:3] / L, state[:, 3:6] / V,
        (a / rt - 1.0)[:, None], e[:, None], (r / rt - 1.0)[:, None],
        (b_in * t).sum(1, keepdims=True), (b_in * w).sum(1, keepdims=True),
        (b_in * s).sum(1, keepdims=True), (fuel / DV_BUDGET)[:, None]], axis=1)
    return jnp.clip(o, -10.0, 10.0)


def point_rate(q, d):
    b_in = qrotate(q, jnp.tile(jnp.array([1.0, 0, 0]), (q.shape[0], 1)))
    err_body = qrotate(qconj(q), cross(b_in, d))
    omega = K_P * err_body
    n = snorm(omega, axis=1, keepdims=True)     # -> 0 when pointed on target (safe grad)
    return omega * jnp.clip(MAX_RATE / n, None, 1.0)


def sample_orbits(key, batch):
    k = random.split(key, 6)
    def u(kk, lo, hi): return lo + (hi - lo) * random.uniform(kk, (batch,))
    r_p = R_BODY + u(k[0], *ALT_PERI)
    r_a = r_p * u(k[1], *RA_RP)
    a = 0.5 * (r_p + r_a); e = (r_a - r_p) / (r_a + r_p)
    p = a * (1 - e ** 2); h = jnp.sqrt(MU * p)
    nu = u(k[2], 0.0, 2 * np.pi); r = p / (1 + e * jnp.cos(nu))
    pf = jnp.stack([r * jnp.cos(nu), r * jnp.sin(nu), jnp.zeros_like(r)], 1)
    pfv = jnp.stack([(MU / h) * (-jnp.sin(nu)),
                     (MU / h) * (e + jnp.cos(nu)), jnp.zeros_like(r)], 1)
    inc = u(k[3], 0.0, INC_MAX); raan = u(k[4], 0.0, 2 * np.pi)
    ci, si = jnp.cos(inc), jnp.sin(inc); cr, sr = jnp.cos(raan), jnp.sin(raan)
    def rot(vec):
        y = vec[:, 1] * ci - vec[:, 2] * si
        zc = vec[:, 1] * si + vec[:, 2] * ci
        x = vec[:, 0]
        return jnp.stack([x * cr - y * sr, x * sr + y * cr, zc], axis=1)
    r_vec = rot(pf); v_vec = rot(pfv)
    q0 = qnorm(random.normal(k[5], (batch, 4)))
    w0 = jnp.zeros((batch, 3))
    return jnp.concatenate([r_vec, v_vec, q0, w0], axis=1), r_a


def _decision_step(params, carry, rt):
    """One decision: policy picks orbit-frame dir + throttle; REPEAT substeps."""
    state, fuel, dv, crash, latch = carry
    obs = observe(state, rt, jnp.clip(fuel, 0.0, None))
    act = policy(params, obs)
    coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

    def substep(c2, _):
        state, fuel, dv, crash = c2
        t, w, s = orbit_frame(state[:, 0:3], state[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / snorm(d, axis=1, keepdims=True)
        omega_cmd = point_rate(state[:, 6:10], d)
        gate = (fuel > 0).astype(jnp.float32)
        thr = throttle * gate
        dv_sub = thr * A_THRUST * DT
        fuel = fuel - dv_sub; dv = dv + dv_sub
        state = rk4(state, omega_cmd, thr)
        rnow = snorm(state[:, 0:3], axis=1)
        crash = crash + jnp.clip((R_BODY - rnow) / R_BODY, 0.0, None) ** 2  # fractional
        return (state, fuel, dv, crash), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash), None, length=REPEAT)
    ae, e = a_err_e(state, rt)
    latch = latch | ((ae < A_TOL) & (e < E_TOL))            # env-style success latch
    if ABSORB:
        state0, fuel0, dv0, crash0, latch0 = carry
        state = jnp.where(latch0[:, None], state0, state)
        fuel = jnp.where(latch0, fuel0, fuel)
        dv = jnp.where(latch0, dv0, dv)
        crash = jnp.where(latch0, crash0, crash)
    return (state, fuel, dv, crash, latch), orbit_err(state, rt)


def make_loss(H, w_orbit=4.0, w_dv=0.05, w_crash=5.0, w_shape=1.0, w_well=1.0, sigma=0.15):
    def loss(params, state, rt):
        B = state.shape[0]
        carry = (state, jnp.full((B,), DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        phi0 = -orbit_err(state, rt)
        def scanfn(carry, _):
            carry, oe = _decision_step(params, carry, rt)
            return carry, oe
        (state, fuel, dv, crash, latch), oes = lax.scan(scanfn, carry, None, length=H)
        oe_T = orbit_err(state, rt)
        shape = (-oe_T) - phi0                             # Ng-1999 potential (telescoped)
        well = -jnp.exp(-oe_T / sigma)
        loss = (w_orbit * oe_T.mean() + w_dv * dv.mean() + w_crash * crash.mean()
                - w_shape * shape.mean() + w_well * well.mean())
        return loss
    return loss


def make_loss_tbptt(H, K=10, w_dv=0.05, w_crash=5.0, w_shape=4.0, w_well=1.0, sigma=0.15):
    """Truncated-BPTT loss: full H-decision forward rollout, but stop_gradient the
    physical state every K decisions so the backward chain is capped at K (avoids the
    long-chain gradient explosion that makes full-H BPTT non-finite). The Ng-1999
    potential shaping Φ=-orbit_err telescopes to the terminal orbit error, so w_shape
    acts as the orbit-error weight but with DENSE, short-chain gradients per chunk
    (an exact-potential analogue of SHAC's learned-critic bootstrap)."""
    nchunks = (H + K - 1) // K

    def loss(params, state, rt):
        B = state.shape[0]
        fuel = jnp.full((B,), DV_BUDGET); dv = jnp.zeros((B,)); crash = jnp.zeros((B,))
        latch = jnp.zeros((B,), bool)
        phi_prev = -orbit_err(state, rt)
        total = 0.0
        for c in range(nchunks):
            k = min(K, H - c * K)
            carry = (state, fuel, dv, crash, latch)
            def scanfn(carry, _):
                carry, _ = _decision_step(params, carry, rt)
                return carry, None
            (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=k)
            phi_now = -orbit_err(state, rt)
            total = total - w_shape * (phi_now - phi_prev).mean()   # = w_shape*Δorbit_err
            # truncate BPTT: cut the physics chain + shaping baseline across the boundary
            state = jax.lax.stop_gradient(state)
            fuel = jax.lax.stop_gradient(fuel)
            phi_prev = jax.lax.stop_gradient(phi_now)
        oe_T = orbit_err(state, rt)
        well = -jnp.exp(-oe_T / sigma)
        total = total + w_dv * dv.mean() + w_crash * crash.mean() + w_well * well.mean()
        return total
    return loss


def make_success(H):
    def success(params, state, rt):
        B = state.shape[0]
        carry = (state, jnp.full((B,), DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        def scanfn(carry, _):
            carry, _ = _decision_step(params, carry, rt)
            return carry, None
        (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
        return latch.mean()
    return success


def make_diag(H):
    """Diagnostic eval: what does the (deterministic-mean) policy actually DO?
    Returns success, mean dv used, mean final a_err, mean final e — discriminates
    'still coasting' (dv≈0) vs 'burning wrong' (dv high, e high) vs 'crashing'."""
    def diag(params, state, rt):
        B = state.shape[0]
        carry = (state, jnp.full((B,), DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        def scanfn(carry, _):
            carry, _ = _decision_step(params, carry, rt)
            return carry, None
        (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
        ae, e = a_err_e(state, rt)
        return latch.mean(), dv.mean(), ae.mean(), e.mean(), (crash > 0).mean()
    return diag


def save_params(path, params, stochastic):
    mlp, log_std = params if stochastic else (params, None)
    d = {}
    for i, (w, b) in enumerate(mlp):
        d[f"w{i}"] = np.asarray(w); d[f"b{i}"] = np.asarray(b)
    if log_std is not None:
        d["log_std"] = np.asarray(log_std)
    np.savez(path, **d)


# --- stochastic exploration (reparameterized / SVG-SAPO) -------------------
# Deterministic APG can't escape the coast basin (no sampling to probe a burn).
# A stochastic policy samples actions a = tanh(mean + std·ε) with reparameterized ε
# so the pathwise dynamics gradient still flows to BOTH mean and std, and an entropy
# bonus keeps std from collapsing to coast before exploration finds the burn.
def mlp_raw(params, obs):
    x = obs
    for (w, b) in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b                     # pre-squash mean (policy = tanh(mlp_raw))


def init_params_stoch(key, init_log_std=-1.0):
    return (init_params(key), jnp.full((4,), init_log_std))   # (mlp, log_std)


def _decision_step_stoch(mlp, log_std, carry, rt, key):
    state, fuel, dv, crash, latch = carry
    obs = observe(state, rt, jnp.clip(fuel, 0.0, None))
    mean = mlp_raw(mlp, obs)
    std = jnp.exp(jnp.clip(log_std, -5.0, 1.0))
    a = jnp.tanh(mean + std * random.normal(key, mean.shape))
    coeffs = a[:, 0:3]; throttle = jnp.clip(a[:, 3], 0.0, 1.0)

    def substep(c2, _):
        state, fuel, dv, crash = c2
        t, w, s = orbit_frame(state[:, 0:3], state[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / snorm(d, axis=1, keepdims=True)
        omega_cmd = point_rate(state[:, 6:10], d)
        gate = (fuel > 0).astype(jnp.float32); thr = throttle * gate
        dv_sub = thr * A_THRUST * DT
        fuel = fuel - dv_sub; dv = dv + dv_sub
        state = rk4(state, omega_cmd, thr)
        rnow = snorm(state[:, 0:3], axis=1)
        crash = crash + jnp.clip((R_BODY - rnow) / R_BODY, 0.0, None) ** 2
        return (state, fuel, dv, crash), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash), None, length=REPEAT)
    ae, e = a_err_e(state, rt)
    latch = latch | ((ae < A_TOL) & (e < E_TOL))
    if ABSORB:
        state0, fuel0, dv0, crash0, latch0 = carry
        state = jnp.where(latch0[:, None], state0, state)
        fuel = jnp.where(latch0, fuel0, fuel)
        dv = jnp.where(latch0, dv0, dv)
        crash = jnp.where(latch0, crash0, crash)
    return (state, fuel, dv, crash, latch), orbit_err(state, rt)


def make_loss_stoch(H, K=10, beta=1e-3, w_dv=0.05, w_crash=5.0, w_shape=4.0,
                    w_well=1.0, sigma=0.15):
    """Stochastic (reparameterized) truncated-BPTT loss with entropy bonus."""
    nchunks = (H + K - 1) // K

    def loss(params, state, rt, key):
        mlp, log_std = params
        B = state.shape[0]
        fuel = jnp.full((B,), DV_BUDGET); dv = jnp.zeros((B,)); crash = jnp.zeros((B,))
        latch = jnp.zeros((B,), bool)
        phi_prev = -orbit_err(state, rt)
        total = 0.0
        for c in range(nchunks):
            k = min(K, H - c * K)
            key, ck = random.split(key)
            keys = random.split(ck, k)
            carry = (state, fuel, dv, crash, latch)
            def scanfn(carry, kk):
                carry, _ = _decision_step_stoch(mlp, log_std, carry, rt, kk)
                return carry, None
            (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, keys)
            phi_now = -orbit_err(state, rt)
            total = total - w_shape * (phi_now - phi_prev).mean()
            state = jax.lax.stop_gradient(state)
            fuel = jax.lax.stop_gradient(fuel)
            phi_prev = jax.lax.stop_gradient(phi_now)
        oe_T = orbit_err(state, rt)
        well = -jnp.exp(-oe_T / sigma)
        entropy = H * jnp.sum(jnp.clip(log_std, -5.0, 1.0))   # ∝ Gaussian entropy, H injections
        total = total + w_dv * dv.mean() + w_crash * crash.mean() + w_well * well.mean()
        return total - beta * entropy
    return loss


# --- manual Adam on a params pytree ---
def adam_init(params):
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (z, jax.tree_util.tree_map(jnp.zeros_like, params), 0)


def adam_step(params, grads, st, lr=3e-4, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
    m, v, t = st
    gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, clip / jnp.clip(gnorm, 1e-9, None))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
    t = t + 1
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
    params = jax.tree_util.tree_map(lambda p, m_, v_: p - lr * m_ / (jnp.sqrt(v_) + eps),
                                    params, mh, vh)
    return params, (m, v, t)


def main():
    global DV_BUDGET, ABSORB, E_WEIGHT
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--eval-horizon", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--w-orbit", type=float, default=4.0)
    ap.add_argument("--w-dv", type=float, default=0.05)
    ap.add_argument("--w-crash", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=10,
                    help="truncated-BPTT chunk length K (0 = full-horizon BPTT)")
    ap.add_argument("--init", type=str, default="",
                    help="warm-start from a converted params .npz (else random)")
    ap.add_argument("--explore", action="store_true",
                    help="stochastic reparameterized policy + entropy (SAPO) to escape coast basin")
    ap.add_argument("--entropy", type=float, default=1e-3, help="entropy bonus beta")
    ap.add_argument("--init-log-std", type=float, default=-1.0)
    ap.add_argument("--throttle-log-std", type=float, default=None,
                    help="separate init log-std for the throttle dim (R9e: throttle noise "
                         "at sigma=0.37 spends the whole dv budget by ~decision 15/60, "
                         "killing late-episode gradient — keep direction exploration rich, "
                         "start throttle sigma small)")
    ap.add_argument("--final-init-scale", type=float, default=0.01,
                    help="final-layer init scale; small => untrained mean ≈ coast")
    ap.add_argument("--save", type=str, default="models/jaxsim_best.npz")
    ap.add_argument("--dv-budget", type=float, default=DV_BUDGET,
                    help="dv budget (km/s). R10 diagnostic: a large budget keeps the fuel "
                         "gate from zeroing the pathwise gradient late in the episode — "
                         "tests whether fuel starvation is what pins eccentricity")
    ap.add_argument("--absorb", action="store_true",
                    help="success is absorbing in the rollout (env-style termination)")
    ap.add_argument("--e-weight", type=float, default=1.0,
                    help="lambda on e in the loss potential (reprices the a-for-e trade)")
    ap.add_argument("--w-well", type=float, default=1.0,
                    help="terminal success-well weight (reprices reaching tolerance)")
    args = ap.parse_args()
    DV_BUDGET = args.dv_budget
    ABSORB = args.absorb
    E_WEIGHT = args.e_weight
    print(f"jax devices: {jax.devices()}  H={args.horizon} K={args.chunk} "
          f"init={args.init or 'random'} budget={DV_BUDGET} absorb={ABSORB} "
          f"e_w={E_WEIGHT} w_well={args.w_well}", flush=True)

    key = random.PRNGKey(args.seed)
    key, kp = random.split(key)
    mlp_init = None
    if args.init:
        d = np.load(args.init)
        mlp_init = [(jnp.asarray(d["w0"]), jnp.asarray(d["b0"])),
                    (jnp.asarray(d["w1"]), jnp.asarray(d["b1"])),
                    (jnp.asarray(d["w2"]), jnp.asarray(d["b2"]))]

    base_diag = make_diag(args.eval_horizon)
    if args.explore:
        mlp = mlp_init if mlp_init is not None else init_params(kp, args.final_init_scale)
        ls0 = np.full(4, args.init_log_std, dtype=np.float32)
        if args.throttle_log_std is not None:
            ls0[3] = args.throttle_log_std
        params = (mlp, jnp.asarray(ls0))
        loss_fn = make_loss_stoch(args.horizon, K=max(1, args.chunk), beta=args.entropy,
                                  w_dv=args.w_dv, w_crash=args.w_crash, w_shape=args.w_orbit,
                                  w_well=args.w_well)
        diag_fn = jit(lambda p, s, rt: base_diag(p[0], s, rt))   # deterministic mean
    else:
        params = mlp_init if mlp_init is not None else init_params(kp, args.final_init_scale)
        if args.chunk and args.chunk < args.horizon:
            det = make_loss_tbptt(args.horizon, K=args.chunk, w_dv=args.w_dv,
                                  w_crash=args.w_crash, w_shape=args.w_orbit)
        else:
            det = make_loss(args.horizon, w_orbit=args.w_orbit, w_dv=args.w_dv,
                            w_crash=args.w_crash)
        loss_fn = lambda p, s, rt, _key: det(p, s, rt)          # ignore key
        diag_fn = jit(base_diag)
    opt = adam_init(params)
    vg = jit(value_and_grad(loss_fn))

    # fixed held-out eval batch
    eval_state, eval_rt = sample_orbits(random.PRNGKey(999_983), 512)

    @jit
    def train_step(params, opt, state, rt, key, lr):
        # lr arrives as a jnp scalar (traced) — a bare Python float would bake into the
        # graph and force a recompile every iter under cosine decay.
        loss, grads = vg(params, state, rt, key)
        # Guard on GRADIENT finiteness, not loss (grad can blow while clipped loss is finite).
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        new_p, new_o = adam_step(params, grads, opt, lr=lr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        params = jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), params, new_p)
        (mo, vo, to), (mn, vn, tn) = opt, new_o
        opt = (jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), mo, mn),
               jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), vo, vn),
               jnp.where(ok, tn, to))
        return params, opt, loss, ok

    t0 = time.time()
    skipped = 0
    best = -1.0
    eta_min = args.lr * 0.1
    for it in range(args.iters):
        key, ks, kt = random.split(key, 3)
        state, rt = sample_orbits(ks, args.batch)
        lr_t = eta_min + 0.5 * (args.lr - eta_min) * (1.0 + np.cos(np.pi * it / args.iters))
        params, opt, loss, ok = train_step(params, opt, state, rt, kt, jnp.float32(lr_t))
        skipped += int(not bool(ok))
        if it % args.eval_every == 0 or it == args.iters - 1:
            s, dvu, ae, e, cr = (float(x) for x in diag_fn(params, eval_state, eval_rt))
            extra = ""
            if args.explore:
                ls = params[1]
                extra = f"  std~{float(jnp.exp(jnp.mean(ls))):.3f}"
            star = ""
            if s > best and args.save:
                best = s
                save_params(args.save, params, args.explore)
                star = "  <-best"
            print(f"iter {it:4d}  loss={float(loss):.4f}  success={s:.2%}  "
                  f"dv={dvu:.2f} a_err={ae:.2f} e={e:.2f} crash={cr:.1%}"
                  f"  skipped={skipped}{extra}{star}  [{time.time()-t0:.0f}s]", flush=True)
    if args.save:
        save_params(args.save.replace(".npz", "_final.npz"), params, args.explore)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
