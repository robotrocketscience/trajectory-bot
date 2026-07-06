#!/usr/bin/env python3
"""Eclipse-avoidance probe — does the policy command thrust INTO the shadow?

Rolls a policy through the eclipse-gated dynamics on the fresh-4096 set and splits
its *commanded* Δv (throttle × A_THRUST × DT, before the shadow gate zeroes it)
into two buckets: thrust commanded in sunlight vs thrust commanded in shadow. The
shadow bucket is wasted intent — the craft asked to fire where it has no power.

A shadow-BLIND policy (trained without eclipse) has no reason to avoid the umbra,
so a chunk of its commanded thrust falls in shadow and is silently cancelled. A
shadow-AWARE policy (retrained with --eclipse) should push that fraction down —
concentrating burns in sunlight / on the sunlit apoapsis passes. That reduction is
the concrete manoeuvre a fixed continuous-yaw Edelbaum law cannot make, and the
mechanism behind any success recovery E2 shows.

    uv run --with "jax[cuda12]" python scripts/eclipse_probe.py \
        --a-thrust 5e-4 --horizon 300 CKPT [CKPT ...]
"""
import argparse
import sys

sys.path.insert(0, "scripts")
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, random
import jaxsim as J

J.DV_BUDGET = 2.0
J.ABSORB = True
J.PHI_DV = True
J.D_EPS = 1e-4


def make_probe(H):
    """Roll out and accumulate (sunlit commanded Δv, shadow commanded Δv, latch)."""
    def probe(params, state, rt):
        B = state.shape[0]
        carry = (state, jnp.full((B,), J.DV_BUDGET), jnp.zeros((B,)),   # sun_dv
                 jnp.zeros((B,)), jnp.zeros((B,), bool))                # shadow_dv, latch

        def decision(carry, _):
            st, fuel, sdv, shdv, latch = carry
            act = J.policy(params, J.observe(st, rt, jnp.clip(fuel, 0.0, None)))
            coeffs = act[:, 0:3]; throttle = jnp.clip(act[:, 3], 0.0, 1.0)

            def sub(c2, _):
                st, fuel, sdv, shdv = c2
                t, w, s = J.orbit_frame(st[:, 0:3], st[:, 3:6])
                d = coeffs[:, 0:1] * t + coeffs[:, 1:2] * w + coeffs[:, 2:3] * s
                d = d / J.snorm(d, axis=1, keepdims=True, eps=J.D_EPS)
                omega = J.point_rate(st[:, 6:10], d)
                sun = J.thrust_gate(st[:, 0:3])                 # 1 sunlit, 0 shadow
                fuelgate = (fuel > 0).astype(jnp.float32)
                cmd = throttle * fuelgate                       # commanded (pre-shadow)
                cmd_dv = cmd * J.A_THRUST * J.DT
                sdv = sdv + cmd_dv * sun                        # actually delivered
                shdv = shdv + cmd_dv * (1.0 - sun)              # cancelled in shadow
                thr = cmd * sun                                 # only sunlit thrust flies
                fuel = fuel - cmd_dv * sun
                st = J.rk4(st, omega, thr)
                return (st, fuel, sdv, shdv), None

            (st, fuel, sdv, shdv), _ = lax.scan(sub, (st, fuel, sdv, shdv),
                                                None, length=J.REPEAT)
            ae, e = J.a_err_e(st, rt)
            latch = latch | ((ae < J.A_TOL) & (e < J.E_TOL))
            # absorbing: frozen episodes stop accumulating
            dead = carry[4]
            st = jnp.where(dead[:, None], carry[0], st)
            fuel = jnp.where(dead, carry[1], fuel)
            sdv = jnp.where(dead, carry[2], sdv)
            shdv = jnp.where(dead, carry[3], shdv)
            latch = jnp.where(dead, carry[4], latch)
            return (st, fuel, sdv, shdv, latch), None

        (st, fuel, sdv, shdv, latch), _ = lax.scan(decision, carry, None, length=H)
        return sdv, shdv, latch
    return probe


def load(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--a-thrust", type=float, required=True)
    ap.add_argument("--horizon", type=int, required=True)
    ap.add_argument("--sun-dir", type=float, nargs=3, default=None)
    args = ap.parse_args()
    J.ECLIPSE = True
    J.A_THRUST = args.a_thrust
    if args.sun_dir is not None:
        J.SUN_DIR = np.asarray(args.sun_dir, dtype=np.float64)

    state, rt = J.sample_orbits(random.PRNGKey(31_337), 4096)
    probe = jax.jit(make_probe(args.horizon))
    print(f"eclipse-avoidance probe  a_thrust={args.a_thrust:.1e} H={args.horizon} "
          f"(fresh4096)")
    print(f"{'ckpt':>28} {'sun Δv':>8} {'shadow Δv':>10} {'shadow%':>8} {'succ':>7}")
    for path in args.ckpts:
        sdv, shdv, latch = probe(load(path), state, rt)
        sdv = np.asarray(sdv); shdv = np.asarray(shdv); latch = np.asarray(latch)
        tot = sdv + shdv
        frac = float(np.sum(shdv) / max(np.sum(tot), 1e-9))    # fleet-wide intent split
        print(f"{path.split('/')[-1]:>28} {np.mean(sdv):>8.3f} {np.mean(shdv):>10.3f} "
              f"{100*frac:>7.1f}% {100*latch.mean():>6.1f}%", flush=True)


if __name__ == "__main__":
    main()
