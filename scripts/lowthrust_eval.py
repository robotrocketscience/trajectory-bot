#!/usr/bin/env python3
"""Thrust-swept fresh-set eval of a circularize policy — the E1 gravity-loss probe.

Scores one or more checkpoints on the standard fresh-4096 set (seed 31_337, the
citable set from eval_probe) at each (A_THRUST, horizon) in a grid. The horizon
grows as thrust falls because a fixed-Δv burn spans more time / more revolutions;
the grid pairs each thrust with a horizon long enough for a ~1 km/s circularize to
have a chance to complete.

Metric: J.make_diag's dvr = spent Δv on latched successes / analytic two-impulse
cost. For circularize-at-own-apoapsis that denominator is the single apoapsis
burn (the impulsive lower bound), so **dvr - 1 is the finite-burn gravity loss.**
Success collapsing at some thrust is the RK4-horizon wall (a result, not a bug).

Config mirrors eval_probe (ABSORB, PHI_DV, D_EPS=1e-4, budget 2.0) so numbers are
comparable to the high-thrust scoreboard.

    uv run --with "jax[cuda12]" python scripts/lowthrust_eval.py CKPT [CKPT ...]
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

# (A_THRUST [km/s^2], horizon [decisions]). 5e-3 = chemical baseline; step down
# toward SEP, lengthening the horizon so the burn has time to complete.
GRID = [
    (5e-3, 120),
    (2e-3, 120),
    (1e-3, 180),
    (5e-4, 300),
    (2e-4, 480),
]

EVAL_SET = J.sample_orbits(random.PRNGKey(31_337), 4096)   # the citable fresh set


def load(path):
    d = np.load(path)
    return [(jnp.asarray(d[f"w{i}"]), jnp.asarray(d[f"b{i}"])) for i in range(3)]


def main():
    ckpts = sys.argv[1:]
    if not ckpts:
        print("usage: lowthrust_eval.py CKPT [CKPT ...]")
        return
    state, rt = EVAL_SET
    for path in ckpts:
        params = load(path)
        print(f"\n=== {path}  (fresh4096) ===")
        print(f"{'a_thrust':>10} {'H':>4} {'success':>8} {'dv':>7} "
              f"{'a_err':>6} {'e':>6} {'crash':>6} {'dvr':>6}")
        for a_thrust, h in GRID:
            J.A_THRUST = a_thrust    # set BEFORE tracing: A_THRUST bakes into the graph,
            diag = jax.jit(J.make_diag(h))   # so jit fresh per cell (no stale-thrust reuse)
            s, dvu, ae, e, cr, dvr = (float(x) for x in diag(params, state, rt))
            print(f"{a_thrust:>10.1e} {h:>4d} {100*s:>7.2f}% {dvu:>7.3f} "
                  f"{ae:>6.3f} {e:>6.3f} {100*cr:>5.1f}% {dvr:>6.3f}", flush=True)


if __name__ == "__main__":
    main()
