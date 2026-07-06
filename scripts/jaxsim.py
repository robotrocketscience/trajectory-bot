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
import argparse, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax import lax, jit, value_and_grad, random

# --- constants (match tbot.orbital / Circularize3DConfig exactly) ---
MU = 398600.4418; R_BODY = 6378.137
DT = 10.0; REPEAT = 20
A_THRUST = 5e-3; RATE_GAIN = 0.1; K_P = 0.5; MAX_RATE = 0.05; DV_BUDGET = 2.0
# J2 oblateness perturbation. 0.0 = pure two-body (bit-exact legacy path); set to
# J2_EARTH to enable. z is the equatorial normal (= the sim's spin axis; sample_orbits
# builds inclination about it), so the standard ECI J2 acceleration applies directly.
J2_COEF = 0.0
J2_EARTH = 1.08262668e-3
# Eclipse thrust-gating (low-thrust fidelity). ECLIPSE=False => thrust_gate returns
# all-ones and every rollout is bit-exact legacy. When True, a solar-powered
# low-thrust craft cannot fire in Earth's shadow, so throttle is forced to 0 inside
# a cylindrical umbra behind the planet w.r.t. SUN_DIR (a fixed inertial unit vector:
# the Sun moves ~1°/day, negligible over a maneuver measured in hours-to-days; wiring
# the real Sun ephemeris is a later fidelity step). This only matters at low thrust,
# where a maneuver spans many revolutions and forced coast arcs reshape the burn plan.
ECLIPSE = False
SUN_DIR = np.array([1.0, 0.0, 0.0])
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
# --phi-dv: use Φ = -dv_to_go (physics-informed control-distance potential) instead
# of Φ = -orbit_err for the shaping term. See dv_to_go() for the trap-state math.
PHI_DV = False
# --absorb-crash: episodes freeze once crash accrues (env-style termination at the
# OTHER terminal condition). Without it the rollout integrates 100+ substeps INSIDE
# the planet (gravity floored, garbage dynamics) and backprop through that segment
# is the prime suspect for the routine 1e17-1e19 gradient norms (R14-pre). The
# crashing decision itself still backprops its full 20 substeps (the meaningful,
# bounded deterrent); only post-crash decisions freeze.
ABSORB_CRASH = False
# --d-eps: eps in the thrust-direction normalization d/|d|. The gradient of
# x/sqrt(|x|^2+eps) at x~0 is 1/sqrt(eps) per component — at the default 1e-12
# that is 1e6, and a COASTING policy (near-zero direction coeffs, e.g. the DAgger
# expert most decisions) seeds monster gradients every step (R15: gmax 1e12 with
# crash=0.0%). 1e-4 caps the seed at 100; direction error is negligible once the
# policy actually burns (|d| >~ 0.1).
D_EPS = 1e-12
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
    if J2_COEF != 0.0:
        # Vallado J2 acceleration (ECI): smooth, differentiable, ->0 with distance.
        # z-term differs (3 - 5(z/r)²) from the x,y-term (1 - 5(z/r)²).
        zr2 = (r[:, 2:3] / rmag) ** 2
        pre = -1.5 * J2_COEF * MU / rmag ** 2 * (R_BODY / rmag) ** 2
        axy = pre * (1.0 - 5.0 * zr2)
        az = pre * (3.0 - 5.0 * zr2)
        a_j2 = jnp.concatenate([axy * r[:, 0:1] / rmag, axy * r[:, 1:2] / rmag,
                                az * r[:, 2:3] / rmag], axis=1)
        grav = grav + a_j2
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


def dv_to_go(state, rt):
    """Smooth two-impulse Δv estimate from the current orbit to circular at rt.

    Physics-informed potential (R13). The oe-potential's level sets ignore control
    distance: from the initial ellipse (apo=rt) circularization is ONE apoapsis burn
    (~1.04 km/s at ra/rp=1.75), while the 'improved' state every run reaches
    (a=rt, e=0.30) still needs ~1.04 km/s (two burns) — oe fell 0.49->0.30 while
    true progress was ZERO, so the policy spends the whole budget standing still.
    Φ = -dv_to_go makes shaping reward = actual progress toward the maneuver.
    Both burn orders evaluated, min taken (subgradient fine); rp/ra clamped so the
    estimate stays finite for degenerate/near-escape orbits (esc term still covers
    the escape direction; crash penalty covers sub-surface periapses).
    """
    a, e = elements(state)
    e_c = jnp.clip(e, 0.0, 0.95)
    lo, hi = 0.3 * R_BODY, 3.0 * A_MAX
    rp = jnp.clip(a * (1.0 - e_c), lo, hi)
    ra = jnp.clip(a * (1.0 + e_c), lo, hi)
    a_cur = 0.5 * (rp + ra)
    v_t = jnp.sqrt(MU / rt)

    def vis_viva(r, aa):
        return jnp.sqrt(jnp.clip(MU * (2.0 / r - 1.0 / aa), 1e-8, None))

    def two_burn(r_burn):
        a_tr = 0.5 * (r_burn + rt)
        dv1 = jnp.abs(vis_viva(r_burn, a_tr) - vis_viva(r_burn, a_cur))
        dv2 = jnp.abs(v_t - vis_viva(rt, a_tr))
        return dv1 + dv2

    return jnp.minimum(two_burn(ra), two_burn(rp))


def potential(state, rt):
    """Shaping potential Φ. --phi-dv switches from -orbit_err to -dv_to_go."""
    if PHI_DV:
        return -dv_to_go(state, rt)
    return -orbit_err(state, rt)


def orbit_frame(r, v):
    t = v / snorm(v, axis=1, keepdims=True)
    h = cross(r, v)
    w = h / snorm(h, axis=1, keepdims=True)
    return t, w, cross(t, w)


# penumbra softening widths [km] for the smooth shadow gate (see thrust_gate). Sharp
# enough to approximate the cylindrical umbra, smooth enough to give the diff-sim a
# gradient across the shadow boundary (a hard boolean gate is flat a.e. → no learning
# signal, and its scf.if select also trips an XLA GPU lowering bug in the scan graph).
SHADOW_W_ALONG = 200.0     # terminator (r·sun = 0) softening
SHADOW_W_RADIAL = 100.0    # umbra edge (perp = R_BODY) softening


def thrust_gate(r):
    """Multiplier on throttle for solar-powered low thrust: ≈1.0 in sunlight, ≈0.0 in
    Earth's shadow (a craft with no sun cannot fire). ECLIPSE=False => exact all-ones,
    so every substep is bit-exact legacy. Smooth cylindrical umbra: shadow grows as the
    craft moves onto the anti-sun side (r·sun < 0) AND within the planet radius of the
    Earth–Sun line. Product of two sigmoids (penumbra-softened) so the boundary is
    differentiable — the diff-sim can learn to steer burns out of shadow."""
    if not ECLIPSE:
        return jnp.ones((r.shape[0],))
    sun = jnp.asarray(SUN_DIR, dtype=r.dtype)
    sun = sun / snorm(sun[None, :], axis=1)[0]
    proj = (r * sun).sum(axis=1)                       # signed distance along the sun line
    perp = snorm(r - proj[:, None] * sun, axis=1)      # distance off the sun line
    behind = jax.nn.sigmoid(-proj / SHADOW_W_ALONG)        # →1 on the anti-sun side
    inside = jax.nn.sigmoid((R_BODY - perp) / SHADOW_W_RADIAL)  # →1 within the umbra
    return 1.0 - behind * inside                       # sunlit fraction ∈ (0, 1]


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


def sample_orbits(key, batch, frac=1.0, full_mix=0.25, rt_jitter=0.0,
                  jitter_frac=1.0):
    """Start-orbit sampler. frac<1 = reverse curriculum (Florensa 1707.05300):
    shrink perigee toward the apoapsis target, r_p' = r_a - (r_a - r_p)*frac, so
    starts sit near/inside the success well where the objective has live gradient
    (frac->0 = nearly circular at target radius; frac=1 = full task, bit-exact
    legacy path). full_mix keeps that fraction of the batch at full difficulty
    so the frontier stays connected to the real task (anti-forgetting).
    jitter_frac<1 (R33) jitters rt on only that share of episodes; the rest
    keep rt == r_a. 1.0 = legacy full-jitter, bit-exact."""
    k = random.split(key, 7)
    def u(kk, lo, hi): return lo + (hi - lo) * random.uniform(kk, (batch,))
    r_p = R_BODY + u(k[0], *ALT_PERI)
    r_a = r_p * u(k[1], *RA_RP)
    if frac < 1.0:
        eff = jnp.where(random.uniform(k[6], (batch,)) < full_mix, 1.0, frac)
        r_p = r_a - (r_a - r_p) * eff
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
    rt = r_a
    if rt_jitter > 0.0:
        # R29 exposed a degenerate conditioning: rt == r_a in EVERY training
        # episode, so the policy cannot fly any other commanded target
        # (aim-scaling collapsed OOD instead of tracing the fuel trade).
        # Decouple target from apoapsis. fold_in keeps all existing streams
        # (eval sets, training draws) bit-identical when jitter is off.
        kj = random.fold_in(key, 424_242)
        rt = r_a * (1.0 + rt_jitter * (2.0 * random.uniform(kj, (batch,)) - 1.0))
        if jitter_frac < 1.0:
            # R33: mixed-fraction jitter — only this share of episodes gets the
            # decoupled target; the rest keep rt == r_a so the gradient stays
            # dominated by the mastered task. Separate fold_in constant leaves
            # the jitter-value stream untouched: episodes that ARE jittered
            # draw the same rt they would at jitter_frac=1.0.
            km = random.fold_in(key, 424_243)
            rt = jnp.where(random.uniform(km, (batch,)) < jitter_frac, rt, r_a)
    return jnp.concatenate([r_vec, v_vec, q0, w0], axis=1), rt


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
        d = d / snorm(d, axis=1, keepdims=True, eps=D_EPS)
        omega_cmd = point_rate(state[:, 6:10], d)
        gate = (fuel > 0).astype(jnp.float32) * thrust_gate(state[:, 0:3])
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
    if ABSORB or ABSORB_CRASH:
        state0, fuel0, dv0, crash0, latch0 = carry
        dead = latch0 if ABSORB else jnp.zeros_like(latch0)
        if ABSORB_CRASH:
            dead = dead | (crash0 > 0.0)
        state = jnp.where(dead[:, None], state0, state)
        fuel = jnp.where(dead, fuel0, fuel)
        dv = jnp.where(dead, dv0, dv)
        crash = jnp.where(dead, crash0, crash)
        latch = jnp.where(dead, latch0, latch)
    return (state, fuel, dv, crash, latch), orbit_err(state, rt)


def make_loss(H, w_orbit=4.0, w_dv=0.05, w_crash=5.0, w_shape=1.0, w_well=1.0, sigma=0.15):
    def loss(params, state, rt):
        B = state.shape[0]
        carry = (state, jnp.full((B,), DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        phi0 = potential(state, rt)
        def scanfn(carry, _):
            carry, oe = _decision_step(params, carry, rt)
            return carry, oe
        (state, fuel, dv, crash, latch), oes = lax.scan(scanfn, carry, None, length=H)
        oe_T = orbit_err(state, rt)
        shape = potential(state, rt) - phi0                # Ng-1999 potential (telescoped)
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
        phi_prev = potential(state, rt)
        total = 0.0
        for c in range(nchunks):
            k = min(K, H - c * K)
            carry = (state, fuel, dv, crash, latch)
            def scanfn(carry, _):
                carry, _ = _decision_step(params, carry, rt)
                return carry, None
            (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=k)
            phi_now = potential(state, rt)
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
        dvgo0 = dv_to_go(state, rt)          # analytic two-impulse cost from the start
        carry = (state, jnp.full((B,), DV_BUDGET), jnp.zeros((B,)),
                 jnp.zeros((B,)), jnp.zeros((B,), bool))
        def scanfn(carry, _):
            carry, _ = _decision_step(params, carry, rt)
            return carry, None
        (state, fuel, dv, crash, latch), _ = lax.scan(scanfn, carry, None, length=H)
        ae, e = a_err_e(state, rt)
        # dv-ratio on successful episodes: spent dv (frozen at latch under ABSORB)
        # over the analytic benchmark — THE beats-the-analytic-transfer number.
        lf = latch.astype(jnp.float32)
        dvr = jnp.sum(lf * dv / jnp.maximum(dvgo0, 1e-3)) / jnp.maximum(lf.sum(), 1.0)
        return latch.mean(), dv.mean(), ae.mean(), e.mean(), (crash > 0).mean(), dvr
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
        d = d / snorm(d, axis=1, keepdims=True, eps=D_EPS)
        omega_cmd = point_rate(state[:, 6:10], d)
        gate = (fuel > 0).astype(jnp.float32) * thrust_gate(state[:, 0:3])
        thr = throttle * gate
        dv_sub = thr * A_THRUST * DT
        fuel = fuel - dv_sub; dv = dv + dv_sub
        state = rk4(state, omega_cmd, thr)
        rnow = snorm(state[:, 0:3], axis=1)
        crash = crash + jnp.clip((R_BODY - rnow) / R_BODY, 0.0, None) ** 2
        return (state, fuel, dv, crash), None

    (state, fuel, dv, crash), _ = lax.scan(substep, (state, fuel, dv, crash), None, length=REPEAT)
    ae, e = a_err_e(state, rt)
    latch = latch | ((ae < A_TOL) & (e < E_TOL))
    if ABSORB or ABSORB_CRASH:
        state0, fuel0, dv0, crash0, latch0 = carry
        dead = latch0 if ABSORB else jnp.zeros_like(latch0)
        if ABSORB_CRASH:
            dead = dead | (crash0 > 0.0)
        state = jnp.where(dead[:, None], state0, state)
        fuel = jnp.where(dead, fuel0, fuel)
        dv = jnp.where(dead, dv0, dv)
        crash = jnp.where(dead, crash0, crash)
        latch = jnp.where(dead, latch0, latch)
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
        phi_prev = potential(state, rt)
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
            phi_now = potential(state, rt)
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
    global DV_BUDGET, ABSORB, E_WEIGHT, PHI_DV, ABSORB_CRASH, D_EPS
    global A_THRUST, ECLIPSE, SUN_DIR
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
    ap.add_argument("--phi-dv", action="store_true",
                    help="physics-informed potential: Φ = -dv_to_go (control distance)")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="global grad-norm clip (0 = off); raw pre-clip norm logged as gmax")
    ap.add_argument("--absorb-crash", action="store_true",
                    help="episodes freeze once crash accrues (env-style termination; "
                         "kills backprop through sub-surface garbage dynamics)")
    ap.add_argument("--d-eps", type=float, default=1e-12,
                    help="eps in thrust-direction normalization (grad seed cap = "
                         "1/sqrt(eps); raise to 1e-4 to defuse coast-decision bombs)")
    ap.add_argument("--clip-ep", type=float, default=0.0,
                    help="per-episode grad-norm clip before the batch mean (0 = off); "
                         "monsters can no longer own the update direction")
    ap.add_argument("--ref", type=str, default="",
                    help="reference policy .npz; logs mean action drift on the eval "
                         "batch (RL's-Razor-style collapse early warning)")
    ap.add_argument("--trim-ep", type=int, default=0,
                    help="trimmed mean: DROP the top-k episodes by grad norm before "
                         "averaging (0 = off); composes with --clip-ep on survivors")
    ap.add_argument("--rt-jitter", type=float, default=0.0,
                    help="decouple target from apoapsis in TRAINING batches: "
                         "rt = r_a * (1 +- U(0,j)). Breaks the rt==r_a degeneracy "
                         "R29 exposed (policy OOD for any commanded target != its "
                         "own apoapsis). Eval sets stay rt=r_a")
    ap.add_argument("--jitter-warmup-iters", type=int, default=0,
                    help="R37: linearly ramp the TRAINING rt-jitter width 0 -> "
                         "--rt-jitter over the first N iters, then hold (0 = off, "
                         "full width from iter 0 = R31). Keeps the distribution shift "
                         "off the mastered specialist incremental, to avoid the "
                         "catastrophic interference R30-R36 showed at full width. The "
                         "eval_j (jit) gauge stays FULL width — it measures skill on "
                         "the whole target region regardless of the training schedule.")
    ap.add_argument("--jitter-frac", type=float, default=1.0,
                    help="fraction of TRAINING episodes whose rt is jittered when "
                         "--rt-jitter>0; the rest keep rt==r_a (R33: gradient stays "
                         "dominated by the mastered task while the new region "
                         "trains at the margin). 1.0 = legacy full-jitter")
    ap.add_argument("--curriculum", action="store_true",
                    help="reverse curriculum on start states (Florensa 1707.05300): "
                         "training batches sampled at difficulty frac (start 0.05), "
                         "25%% of each batch at full difficulty; every eval, if "
                         "success on a current-frac probe set >= 0.75, frac += 0.05. "
                         "Headline eval set stays full-difficulty")
    ap.add_argument("--anchor", type=float, default=0.0,
                    help="BC-anchor weight (TD3+BC 2106.06860 / RL's Razor): adds "
                         "w * MSE(policy, init-policy actions) on a frozen 8192-obs "
                         "set collected from the INIT policy's own rollouts, as a "
                         "separate smooth gradient added to the rollout gradient. "
                         "Guards against systematic drift off a warm-start peak "
                         "(R22: collapse into the burn-in-place trap with drift "
                         "0.155->0.45). Det path only")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="Polyak/EMA decay for an averaged policy (0 = off; e.g. 0.995 "
                         "~ 200-iter horizon). Passive w.r.t. training: raw params keep "
                         "training; eval/best-checkpointing switch to the EMA policy, "
                         "with the raw policy's success logged as raw= for a paired "
                         "within-run control (one-step probe: success is knife-edge "
                         "sensitive to single Adam steps, so in-run evals ride jitter)")
    ap.add_argument("--a-thrust", type=float, default=A_THRUST,
                    help="thrust acceleration [km/s^2] (default 5e-3 = chemical, "
                         "near-impulsive over the 200s decision). Lower it toward SEP "
                         "levels (~1e-4) to enter the low-thrust regime where the burn "
                         "spans many revs and gravity/turn losses appear vs the "
                         "impulsive lower bound. Longer maneuvers need a longer --horizon.")
    ap.add_argument("--eclipse", action="store_true",
                    help="solar-powered low thrust: force throttle to 0 in Earth's "
                         "cylindrical shadow (see thrust_gate). Only bites at low thrust.")
    ap.add_argument("--sun-dir", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="fixed inertial Sun direction for --eclipse (default +x)")
    ap.add_argument("--eval-cpu", action="store_true",
                    help="compile the in-run diag eval on the CPU backend (training "
                         "stays on GPU). Workaround for an intermittent XLA GPU "
                         "lowering bug (scf.if shape mismatch) that the --eclipse eval "
                         "graph trips at batch 512; same fix combined_sim uses for "
                         "its DAgger eval.")
    args = ap.parse_args()
    DV_BUDGET = args.dv_budget
    A_THRUST = args.a_thrust
    ECLIPSE = args.eclipse
    if args.sun_dir is not None:
        SUN_DIR = np.asarray(args.sun_dir, dtype=np.float64)
    ABSORB = args.absorb
    E_WEIGHT = args.e_weight
    PHI_DV = args.phi_dv
    ABSORB_CRASH = args.absorb_crash
    D_EPS = args.d_eps
    print(f"jax devices: {jax.devices()}  H={args.horizon} K={args.chunk} "
          f"init={args.init or 'random'} budget={DV_BUDGET} absorb={ABSORB} "
          f"e_w={E_WEIGHT} w_well={args.w_well} phi_dv={PHI_DV} "
          f"absorb_crash={ABSORB_CRASH} d_eps={D_EPS} ema={args.ema} "
          f"a_thrust={A_THRUST} eclipse={ECLIPSE} sun_dir={np.asarray(SUN_DIR).tolist()} "
          f"rt_jitter={args.rt_jitter} jitter_warmup={args.jitter_warmup_iters} jitter_frac={args.jitter_frac}", flush=True)
    # full argv in the log: the header above echoes only some flags, and the
    # calibrated stack lives in non-defaults — R33 launch 1 was voided by a
    # silently-missing flag set. Every run log must be self-describing.
    print("argv: " + " ".join(sys.argv[1:]), flush=True)

    key = random.PRNGKey(args.seed)
    key, kp = random.split(key)
    mlp_init = None
    if args.init:
        d = np.load(args.init)
        mlp_init = [(jnp.asarray(d["w0"]), jnp.asarray(d["b0"])),
                    (jnp.asarray(d["w1"]), jnp.asarray(d["b1"])),
                    (jnp.asarray(d["w2"]), jnp.asarray(d["b2"]))]

    base_diag = make_diag(args.eval_horizon)
    eval_backend = "cpu" if args.eval_cpu else None   # None = default (GPU when present)
    if args.explore:
        mlp = mlp_init if mlp_init is not None else init_params(kp, args.final_init_scale)
        ls0 = np.full(4, args.init_log_std, dtype=np.float32)
        if args.throttle_log_std is not None:
            ls0[3] = args.throttle_log_std
        params = (mlp, jnp.asarray(ls0))
        loss_fn = make_loss_stoch(args.horizon, K=max(1, args.chunk), beta=args.entropy,
                                  w_dv=args.w_dv, w_crash=args.w_crash, w_shape=args.w_orbit,
                                  w_well=args.w_well)
        diag_fn = jit(lambda p, s, rt: base_diag(p[0], s, rt), backend=eval_backend)   # deterministic mean
    else:
        params = mlp_init if mlp_init is not None else init_params(kp, args.final_init_scale)
        if args.chunk and args.chunk < args.horizon:
            det = make_loss_tbptt(args.horizon, K=args.chunk, w_dv=args.w_dv,
                                  w_crash=args.w_crash, w_shape=args.w_orbit)
        else:
            det = make_loss(args.horizon, w_orbit=args.w_orbit, w_dv=args.w_dv,
                            w_crash=args.w_crash)
        loss_fn = lambda p, s, rt, _key: det(p, s, rt)          # ignore key
        diag_fn = jit(base_diag, backend=eval_backend)
    opt = adam_init(params)
    vg = jit(value_and_grad(loss_fn))

    # fixed held-out eval batch
    eval_state, eval_rt = sample_orbits(random.PRNGKey(999_983), 512)
    eval_j = None
    if args.rt_jitter > 0.0:
        # deliberately full-jitter regardless of --jitter-frac: the jit gauge
        # measures skill on the decoupled-target region itself
        eval_j = sample_orbits(random.PRNGKey(999_991), 512, rt_jitter=args.rt_jitter)

    ref_mlp = act_ref = obs0 = None
    if args.ref:
        dref = np.load(args.ref)
        ref_mlp = [(jnp.asarray(dref[f"w{i}"]), jnp.asarray(dref[f"b{i}"])) for i in range(3)]
        obs0 = observe(eval_state, eval_rt, jnp.full((eval_state.shape[0],), DV_BUDGET))
        act_ref = policy(ref_mlp, obs0)

    # BC anchor: freeze the INIT policy's own visited-state distribution once,
    # then every update adds w_a * dMSE/dparams toward the init actions on it.
    # The anchor gradient is smooth and bounded (tanh MSE) — no monsters.
    anchor_vg = None
    w_a = float(args.anchor)
    if w_a > 0.0:
        assert mlp_init is not None and not args.explore, "--anchor needs --init, det path"
        a_state, a_rt = sample_orbits(random.PRNGKey(424_243), 512)
        a_carry = (a_state, jnp.full((512,), DV_BUDGET), jnp.zeros((512,)),
                   jnp.zeros((512,)), jnp.zeros((512,), bool))
        a_step = jit(lambda c: _decision_step(mlp_init, c, a_rt))
        obs_list = []
        for _ in range(args.horizon):
            obs_list.append(observe(a_carry[0], a_rt, jnp.clip(a_carry[1], 0.0, None)))
            a_carry, _ = a_step(a_carry)
        all_obs = jnp.concatenate(obs_list, axis=0)
        idx = random.permutation(random.PRNGKey(31_415), all_obs.shape[0])[:8192]
        anchor_obs = all_obs[idx]
        anchor_act = policy(mlp_init, anchor_obs)
        anchor_vg = jit(value_and_grad(
            lambda p: ((policy(p, anchor_obs) - anchor_act) ** 2).mean()))

    clip_g = float(args.clip_grad)

    clip_ep = float(args.clip_ep)
    trim_ep = int(args.trim_ep)
    per_ep = clip_ep > 0.0 or trim_ep > 0
    if per_ep:
        # Per-episode gradient handling: rollout gradients are heavy-tailed (R14-pre:
        # routine 1e12-1e19 monsters), and with only a GLOBAL clip the update becomes a
        # unit step in the monster episode's direction — 1 pathological episode owns
        # the whole batch (R14/R15). --clip-ep rescales each episode's grad to
        # norm<=clip_ep before the mean; --trim-ep DELETES the top-k episodes by norm
        # (trimmed mean, Yin et al. 2018) — removes the monster without distorting the
        # healthy magnitude structure, which R19 showed clipping still erodes slowly.
        base_vg = value_and_grad(lambda p, s1, r1, k1: loss_fn(p, s1[None], r1[None], k1))
        vg_ep = jax.vmap(base_vg, in_axes=(None, 0, 0, 0))

        def vg(p, s, rt, key):  # noqa: F811 — replaces the batch-grad path
            B = s.shape[0]
            keys = random.split(key, B)
            losses, grads = vg_ep(p, s, rt, keys)
            # sanitize per-episode: a non-finite episode must not poison the mean
            # (inf * scale-0 would be NaN) — zero it out instead of skipping the batch
            grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
            losses = jnp.where(jnp.isfinite(losses), losses, 0.0)
            norms = jnp.sqrt(sum(jnp.sum(g.reshape(g.shape[0], -1) ** 2, axis=1)
                                 for g in jax.tree_util.tree_leaves(grads)))
            scale = jnp.ones_like(norms)
            if trim_ep > 0:
                cutoff = jnp.sort(norms)[B - trim_ep - 1]
                scale = scale * (norms <= cutoff).astype(jnp.float32)
            if clip_ep > 0.0:
                scale = scale * jnp.minimum(1.0, clip_ep / jnp.maximum(norms, 1e-12))
            kept = jnp.maximum(jnp.sum((scale > 0).astype(jnp.float32)), 1.0)
            grads = jax.tree_util.tree_map(
                lambda g: jnp.sum(g * scale.reshape((-1,) + (1,) * (g.ndim - 1)), axis=0) / kept,
                grads)
            return (losses.mean(), norms.max()), grads

    @jit
    def train_step(params, opt, state, rt, key, lr):
        # lr arrives as a jnp scalar (traced) — a bare Python float would bake into the
        # graph and force a recompile every iter under cosine decay.
        out, grads = vg(params, state, rt, key)
        loss, epmax = out if per_ep else (out, jnp.float32(0.0))
        if anchor_vg is not None:
            _, ag = anchor_vg(params)
            grads = jax.tree_util.tree_map(lambda g, a: g + w_a * a, grads, ag)
        # Guard on GRADIENT finiteness, not loss (grad can blow while clipped loss is finite).
        gnorm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
        g_report = jnp.maximum(gnorm, epmax)   # telemetry keeps showing raw monsters
        if clip_g > 0.0:
            # Global-norm clip. Crash-zone gradients through rk4 (gravity ~ 1/r^2) are
            # unbounded; R10/R13 regime jumps + late loss RISES look like catapult
            # steps hurling the policy into tanh saturation it never escapes.
            scale = jnp.minimum(1.0, clip_g / jnp.maximum(gnorm, 1e-12))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        new_p, new_o = adam_step(params, grads, opt, lr=lr)
        ok = jnp.isfinite(loss) & jnp.isfinite(gnorm)
        params = jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), params, new_p)
        (mo, vo, to), (mn, vn, tn) = opt, new_o
        opt = (jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), mo, mn),
               jax.tree_util.tree_map(lambda o, n: jnp.where(ok, n, o), vo, vn),
               jnp.where(ok, tn, to))
        return params, opt, loss, ok, g_report

    ema_p = None
    if args.ema > 0.0:
        ema_p = params
        tau = float(args.ema)
        ema_fn = jit(lambda e, p: jax.tree_util.tree_map(
            lambda a, b: tau * a + (1.0 - tau) * b, e, p))

    t0 = time.time()
    skipped = 0
    best = -1.0
    eta_min = args.lr * 0.1
    gmax = 0.0                       # max raw (pre-clip) grad norm since last eval line
    frac = 0.05 if args.curriculum else 1.0
    for it in range(args.iters):
        key, ks, kt = random.split(key, 3)
        # R37 curriculum: ramp jitter width 0 -> args.rt_jitter over the warmup,
        # then hold. Off (warmup=0) -> constant full width (R31 behaviour).
        jit_w = args.rt_jitter
        if args.jitter_warmup_iters > 0:
            jit_w = args.rt_jitter * min(1.0, it / args.jitter_warmup_iters)
        state, rt = sample_orbits(ks, args.batch, frac=frac, rt_jitter=jit_w,
                                  jitter_frac=args.jitter_frac)
        lr_t = eta_min + 0.5 * (args.lr - eta_min) * (1.0 + np.cos(np.pi * it / args.iters))
        params, opt, loss, ok, gn = train_step(params, opt, state, rt, kt, jnp.float32(lr_t))
        skipped += int(not bool(ok))
        if ema_p is not None:
            ema_p = ema_fn(ema_p, params)
        gf = float(gn)
        if np.isfinite(gf):
            gmax = max(gmax, gf)
        if it % args.eval_every == 0 or it == args.iters - 1:
            eval_p = ema_p if ema_p is not None else params
            s, dvu, ae, e, cr, dvr = (float(x) for x in diag_fn(eval_p, eval_state, eval_rt))
            extra = f"  dvr={dvr:.2f}" if s > 0 else ""
            if ema_p is not None:
                sr = float(diag_fn(params, eval_state, eval_rt)[0])
                extra += f"  raw={sr:.2%}"
            if anchor_vg is not None:
                extra += f"  aloss={float(anchor_vg(eval_p)[0]):.4f}"
            if eval_j is not None:
                extra += f"  jit={float(diag_fn(eval_p, *eval_j)[0]):.2%}"
            if args.curriculum:
                cs, crt = sample_orbits(random.PRNGKey(777_000 + int(frac * 100)),
                                        eval_state.shape[0], frac=frac, full_mix=0.0)
                s_curr = float(diag_fn(eval_p, cs, crt)[0])
                if s_curr >= 0.75:
                    frac = min(1.0, round(frac + 0.05, 2))
                extra += f"  frac={frac:.2f} s_cur={s_curr:.0%}"
            if ref_mlp is not None:
                mlp_now = eval_p[0] if args.explore else eval_p
                drift = float(jnp.abs(policy(mlp_now, obs0) - act_ref).mean())
                extra += f"  drift={drift:.4f}"
            if args.explore:
                ls = params[1]
                extra += f"  std~{float(jnp.exp(jnp.mean(ls))):.3f}"
            star = ""
            if s > best and args.save:
                best = s
                save_params(args.save, eval_p, args.explore)
                star = "  <-best"
            print(f"iter {it:4d}  loss={float(loss):.4f}  success={s:.2%}  "
                  f"dv={dvu:.2f} a_err={ae:.2f} e={e:.2f} crash={cr:.1%}"
                  f"  gmax={gmax:.1f}  skipped={skipped}{extra}{star}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            gmax = 0.0
    if args.save:
        save_params(args.save.replace(".npz", "_final.npz"), params, args.explore)
        if ema_p is not None:
            save_params(args.save.replace(".npz", "_ema_final.npz"), ema_p, args.explore)
    print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
