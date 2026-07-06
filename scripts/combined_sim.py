#!/usr/bin/env python3
"""Combined circularize + plane-change task (JAX) — the single-burn KSC step.

An inclined ellipse (apoapsis on the equatorial line of nodes, as the shared
sampler builds it) -> circular AND equatorial at its own apoapsis, in ONE
combined apoapsis burn. The naive textbook route circularizes, then pays a
separate plane change at circular speed; the combined burn folds both into one
vector Δv (~25% cheaper at 28.5°, see tbot.orbital3d.combined_circularize_plane_dv).
The thesis test: a diff-sim policy optimizing raw Δv should DISCOVER the combined
burn rather than the naive decomposition.

Reuses jaxsim's dynamics/constants verbatim (deriv/rk4/orbit_frame/point_rate/
elements); adds only a plane-aware observation (13->16), an inclination success
term, and a privileged combined-burn expert for DAgger. This module is the env +
expert + a forward-only validation of the expert (no training).

    uv run --with jax python scripts/combined_sim.py --validate --episodes 512

Experiment code (excluded from the strict-typed library).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random

sys.path.insert(0, "scripts")
import jaxsim as J  # noqa: E402

MU = J.MU
Z_AXIS = jnp.array([0.0, 0.0, 1.0])

# task knobs
INC_TOL = np.radians(3.0)          # plane tolerance for the success latch
COS_INC_TOL = float(np.cos(INC_TOL))
W_PLANE = 4.0                      # weight on plane error in the shaping potential

# expert knobs (tuned by --validate)
K_EXPERT = 1.5                     # throttle gain on |Δv| / v_circ
GATE_RDOT = 0.35                   # [km/s] radial-speed half-width of the apo burn gate


# --- plane-aware observation (jaxsim's 13 + 3 equatorial-normal projections) ---
def observe16(state, rt, fuel):
    base = J.observe(state, rt, fuel)                  # 13, rt-normalized + clipped
    t, w, s = J.orbit_frame(state[:, 0:3], state[:, 3:6])
    # +z (the target equatorial normal) resolved in the orbit frame: z·w = cos(inc),
    # and z·t / z·s tell the policy which way to tilt the plane. Everything it needs
    # to null inclination, in the same frame it already acts in.
    plane = jnp.stack([(Z_AXIS * t).sum(1), (Z_AXIS * w).sum(1),
                       (Z_AXIS * s).sum(1)], axis=1)
    return jnp.concatenate([base, plane], axis=1)      # 16


def plane_err(state):
    """1 - cos(inc): ->0 at a prograde equatorial orbit, smooth (safe gradient)."""
    h = J.cross(state[:, 0:3], state[:, 3:6])
    cos_i = h[:, 2] / J.snorm(h, axis=1)
    return 1.0 - cos_i


def is_success(state, rt):
    ae, e = J.a_err_e(state, rt)
    h = J.cross(state[:, 0:3], state[:, 3:6])
    cos_i = h[:, 2] / J.snorm(h, axis=1)
    return (ae < J.A_TOL) & (e < J.E_TOL) & (cos_i > COS_INC_TOL)


def orbit_err_plane(state, rt):
    return J.orbit_err(state, rt) + W_PLANE * plane_err(state)


# --- 16-input MLP (identical forward to J.policy; only the init shape changes) ---
def init_params(key, final_scale=1.0):
    ks = random.split(key, 3)

    def layer(k, nin, nout, scale=1.0):
        return (random.normal(k, (nin, nout)) * (scale / np.sqrt(nin)),
                jnp.zeros((nout,)))
    return [layer(ks[0], 16, J.HID), layer(ks[1], J.HID, J.HID),
            layer(ks[2], J.HID, 4, scale=final_scale)]


# --- privileged combined-burn expert (full state; generates DAgger labels) ------
def expert_action(state):
    """One combined apoapsis burn: rotate+scale v toward the equatorial circular
    velocity at the current position. Gated to the apoapsis half near the apsis so
    the burn straddles the node; throttle scales with the remaining Δv so it stops
    when the orbit is circular+equatorial."""
    r = state[:, 0:3]; v = state[:, 3:6]
    rmag = J.snorm(r, axis=1, keepdims=True)
    a, e = J.elements(state)
    v_circ = jnp.sqrt(MU / rmag)                       # circular speed at current r
    rhat = r / rmag
    zc = jnp.cross(Z_AXIS, rhat)                        # ⊥r, in the equatorial plane
    zc = zc / J.snorm(zc, axis=1, keepdims=True)
    prog = jnp.sign((zc * v).sum(1, keepdims=True) + 1e-12)   # match motion sense
    v_tgt = v_circ * prog * zc                          # equatorial circular target v
    dv = v_tgt - v

    t, w, s = J.orbit_frame(r, v)
    coeffs = jnp.stack([(dv * t).sum(1), (dv * w).sum(1), (dv * s).sum(1)], axis=1)
    cdir = coeffs / J.snorm(coeffs, axis=1, keepdims=True, eps=1e-6)

    rdot = (r * v).sum(1) / rmag[:, 0]
    outer = (rmag[:, 0] > a).astype(jnp.float32)        # apoapsis half only
    gate = jnp.clip(1.0 - jnp.abs(rdot) / GATE_RDOT, 0.0, 1.0)
    dv_mag = J.snorm(dv, axis=1)
    throttle = jnp.clip(K_EXPERT * dv_mag / v_circ[:, 0], 0.0, 1.0) * outer * gate
    return jnp.concatenate([cdir, throttle[:, None]], axis=1)


# --- rollout: drive the shared substep dynamics with a chosen action fn ----------
def _decision(action_fn, params, carry, rt):
    """One decision with plane-aware obs/success. Mirrors J._decision_step's
    substep integration and Δv accounting exactly; action_fn(params, state)->act."""
    state, fuel, dv, crash, latch = carry
    act = action_fn(params, state)
    coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

    def substep(c2, _):
        st, fu, dvv, cr = c2
        t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
        d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
        d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
        omega_cmd = J.point_rate(st[:, 6:10], d)
        thr = throttle * (fu > 0).astype(jnp.float32)
        dv_sub = thr * J.A_THRUST * J.DT
        fu = fu - dv_sub; dvv = dvv + dv_sub
        st = J.rk4(st, omega_cmd, thr)
        rnow = J.snorm(st[:, 0:3], axis=1)
        cr = cr + jnp.clip((J.R_BODY - rnow) / J.R_BODY, 0.0, None) ** 2
        return (st, fu, dvv, cr), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash),
                                           None, length=J.REPEAT)
    latch = latch | is_success(state, rt)
    # absorbing success: episodes already latched BEFORE this step freeze, so
    # post-success exploration drift can't unlatch them or spend more Δv.
    dead = carry[4]
    state = jnp.where(dead[:, None], carry[0], state)
    fuel = jnp.where(dead, carry[1], fuel)
    dv = jnp.where(dead, carry[2], dv)
    crash = jnp.where(dead, carry[3], crash)
    latch = jnp.where(dead, carry[4], latch)
    return (state, fuel, dv, crash, latch), None


def rollout(action_fn, params, s0, rt, H=120, collect_obs=False):
    B = s0.shape[0]
    carry = (s0, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),
             jnp.zeros((B,)), jnp.zeros((B,), bool))
    if collect_obs:
        def scanfn(c, _):
            obs = observe16(c[0], rt, jnp.clip(c[1], 0.0, None))
            c2, _ = _decision(action_fn, params, c, rt)
            return c2, obs
        (state, fuel, dv, crash, latch), obs_seq = lax.scan(scanfn, carry, None, length=H)
        return (state, fuel, dv, crash, latch), obs_seq.reshape(-1, 16)

    def scanfn(c, _):
        c2, _ = _decision(action_fn, params, c, rt)
        return c2, None
    (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
    return (state, fuel, dv, crash, latch), None


# --- analytic per-episode baseline (vectorized combined_circularize_plane_dv) ---
def baseline_dv(s0):
    r = np.asarray(s0[:, 0:3]); v = np.asarray(s0[:, 3:6])
    rmag = np.linalg.norm(r, axis=1); vmag = np.linalg.norm(v, axis=1)
    energy = 0.5 * vmag ** 2 - MU / rmag
    a = -MU / (2.0 * energy)
    h = np.cross(r, v); hmag = np.linalg.norm(h, axis=1)
    e = np.sqrt(np.clip(1.0 - hmag ** 2 / (MU * a), 0.0, None))
    r_a = a * (1.0 + e); r_p = a * (1.0 - e)
    inc = np.arccos(np.clip(h[:, 2] / hmag, -1.0, 1.0))
    a_ell = 0.5 * (r_p + r_a)
    v_apo = np.sqrt(MU * (2.0 / r_a - 1.0 / a_ell))
    v_circ = np.sqrt(MU / r_a)
    naive = np.abs(v_circ - v_apo) + 2.0 * v_circ * np.abs(np.sin(inc / 2.0))
    combined = np.sqrt(v_apo ** 2 + v_circ ** 2 - 2.0 * v_apo * v_circ * np.cos(inc))
    return naive, combined, inc, r_a


def validate(episodes, H, seed):
    key = random.PRNGKey(seed)
    s0, rt = J.sample_orbits(key, episodes)     # inc∈[0,40°], rt=r_a
    (state, fuel, dv, crash, latch), _ = rollout(expert_action_wrap, None, s0, rt, H=H)
    naive, combined, inc, r_a = baseline_dv(s0)
    dv = np.asarray(dv); latch = np.asarray(latch); crash = np.asarray(crash)
    clean = (crash == 0.0)
    ok = latch & clean
    print(f"expert over {episodes} episodes (H={H}):")
    print(f"  success (a,e,inc<{np.degrees(INC_TOL):.0f}°) = {100*latch.mean():.1f}%"
          f"   crash-free = {100*clean.mean():.1f}%")
    if ok.sum() > 0:
        print(f"  measured Δv on clean successes: median={np.median(dv[ok]):.3f} km/s"
              f"  (n={ok.sum()})")
        print(f"  analytic combined (same eps):   median={np.median(combined[ok]):.3f} km/s")
        print(f"  analytic naive    (same eps):   median={np.median(naive[ok]):.3f} km/s")
        print(f"  measured/combined ratio median = {np.median(dv[ok]/combined[ok]):.3f}")
        print(f"  measured/naive    ratio median = {np.median(dv[ok]/naive[ok]):.3f}")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 41)]:
        m = (np.degrees(inc) >= lo) & (np.degrees(inc) < hi)
        if m.sum():
            print(f"  inc [{lo:2d},{hi:2d})°: succ={100*latch[m].mean():5.1f}%"
                  f"  n={m.sum():4d}  median Δv={np.median(dv[m & clean]) if (m&clean).any() else float('nan'):.3f}")


def expert_action_wrap(_params, state):
    return expert_action(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=4.0)
    args = ap.parse_args()
    J.DV_BUDGET = args.budget
    if args.validate:
        validate(args.episodes, args.horizon, args.seed)


if __name__ == "__main__":
    main()
