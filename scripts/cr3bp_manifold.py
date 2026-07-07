#!/usr/bin/env python3
"""Stable-manifold ballistic capture in the CR3BP (Build H).

G's naive departure-burn grid could not find a ballistic capture because the capture
set is the STABLE MANIFOLD of an L1/L2 periodic (Lyapunov) orbit — a measure-zero
tube you must seed onto deliberately (Conley-McGehee; Koon-Lo-Marsden-Ross 2011). H
builds that structure:

  R-H1  Lyapunov orbit about L1/L2 via differential correction   (--orbit)
  R-H2  monodromy eigenstructure + stable manifold               (--manifold)
  R-H3  sweep manifold trajectories for VERIFIED ballistic capture (--capture)
  R-H4  Earth-connection + honest Δv vs Hohmann+capture           (--transfer)

Dynamics here are an INDEPENDENT float64 numpy reimplementation of cr3bp_sim's
rotating-frame CR3BP (differential correction needs float64, and a second
implementation is a bug-catch); cross-checked against cr3bp_sim.accel in --check.

    uv run --with jax python scripts/cr3bp_manifold.py --check --orbit
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402  (for MU, geometry, cross-check, physical units)

MU = C.MU
OM = 1.0 - MU
R_MOON = np.array([1.0 - MU, 0.0, 0.0])
R_HILL = (MU / 3.0) ** (1.0 / 3.0)


# ----- independent float64 dynamics -------------------------------------------------
def accel_np(s):
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    ax = 2.0 * vy + x - OM * (x + MU) / r1 ** 3 - MU * (x - 1.0 + MU) / r2 ** 3
    ay = -2.0 * vx + y - OM * y / r1 ** 3 - MU * y / r2 ** 3
    az = -OM * z / r1 ** 3 - MU * z / r2 ** 3
    return np.array([vx, vy, vz, ax, ay, az])


def rk4_np(s, dt):
    k1 = accel_np(s)
    k2 = accel_np(s + 0.5 * dt * k1)
    k3 = accel_np(s + 0.5 * dt * k2)
    k4 = accel_np(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobi_np(s):
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    Om = 0.5 * (x ** 2 + y ** 2) + OM / r1 + MU / r2
    return 2.0 * Om - (vx * vx + vy * vy + vz * vz)


def omega_second(xL):
    """Ω_xx, Ω_yy at a collinear point (y=z=0). c = (1-μ)/|x+μ|³ + μ/|x-1+μ|³."""
    c = OM / abs(xL + MU) ** 3 + MU / abs(xL - 1.0 + MU) ** 3
    return 1.0 + 2.0 * c, 1.0 - c, c


def linear_seed(xL, Ax):
    """In-plane center-mode seed: x = xL - Ax cos(ω t), y = κ Ax sin(ω t)."""
    Uxx, Uyy, _ = omega_second(xL)
    # Λ⁴ + (4 - Uxx - Uyy)Λ² + Uxx·Uyy = 0; center root β = Λ² < 0 → ω = √(-β)
    b = 4.0 - Uxx - Uyy
    disc = np.sqrt(b * b - 4.0 * Uxx * Uyy)
    beta_center = 0.5 * (-b - disc)           # the negative root
    wp = np.sqrt(-beta_center)
    kappa = (wp * wp + Uxx) / (2.0 * wp)
    return np.array([xL - Ax, 0.0, 0.0, 0.0, kappa * Ax * wp, 0.0]), wp


# ----- differential correction (R-H1) -----------------------------------------------
def half_period_crossing(s0, dt):
    """Integrate until y returns to 0 (from +); linear-interp the crossing state & time."""
    s = s0.copy()
    t = 0.0
    y_prev = s[1]
    s_prev = s.copy()
    # step off the y=0 axis first
    for _ in range(20_000_000):
        s_new = rk4_np(s, dt)
        t_new = t + dt
        if s[1] >= 0.0 and s_new[1] < 0.0 and t_new > dt * 5:
            frac = s[1] / (s[1] - s_new[1])       # fraction to y=0
            s_cross = s + frac * (s_new - s)
            return s_cross, t + frac * dt
        s_prev, y_prev = s, s[1]
        s, t = s_new, t_new
    raise RuntimeError("no y=0 crossing found")


def integrate_fixed_time(s0, dt, T):
    """Integrate for exactly time T (floor steps + one remainder step)."""
    n = int(np.floor(T / dt))
    s = s0.copy()
    for _ in range(n):
        s = rk4_np(s, dt)
    rem = T - n * dt
    if rem > 1e-15:
        s = rk4_np(s, rem)
    return s


def differential_correct(xL, Ax, dt=1e-4, tol=1e-11, maxit=40):
    """STM-based correction: hold x0 fixed, correct vy0 so the return y=0 crossing is
    perpendicular (vx=0). Accounts for crossing-time variation via the
    a_x/vy term (Koon-Lo-Marsden-Ross §4): δvy0 = -vx_c / (Φ34 − (a_xc/vy_c)Φ24)."""
    s0, wp = linear_seed(xL, Ax)
    for it in range(maxit):
        sc, thalf = half_period_crossing(s0, dt)
        vx_c = sc[3]
        if abs(vx_c) < tol:
            return s0.copy(), 2.0 * thalf, wp, it, abs(vx_c)
        a_c = accel_np(sc)                          # accel at crossing → a_xc = a_c[3]
        vy_c = sc[4]
        # STM columns for the vy0 perturbation, evaluated at the SAME crossing time
        h = 1e-7
        s0h = s0.copy(); s0h[4] += h
        sch = integrate_fixed_time(s0h, dt, thalf)
        phi34 = (sch[3] - vx_c) / h                 # ∂vx_c/∂vy0
        phi24 = (sch[1] - sc[1]) / h                # ∂y_c/∂vy0
        denom = phi34 - (a_c[3] / vy_c) * phi24
        s0[4] -= vx_c / denom
    sc, thalf = half_period_crossing(s0, dt)
    return s0.copy(), 2.0 * thalf, wp, maxit, abs(sc[3])


# ----- batched float64 dynamics (STM finite-difference + manifold propagation) -----
def accel_np_batch(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    vx, vy, vz = S[:, 3], S[:, 4], S[:, 5]
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    ax = 2.0 * vy + x - OM * (x + MU) / r1 ** 3 - MU * (x - 1.0 + MU) / r2 ** 3
    ay = -2.0 * vx + y - OM * y / r1 ** 3 - MU * y / r2 ** 3
    az = -OM * z / r1 ** 3 - MU * z / r2 ** 3
    return np.stack([vx, vy, vz, ax, ay, az], axis=1)


def rk4_np_batch(S, dt):
    k1 = accel_np_batch(S)
    k2 = accel_np_batch(S + 0.5 * dt * k1)
    k3 = accel_np_batch(S + 0.5 * dt * k2)
    k4 = accel_np_batch(S + dt * k3)
    return S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def jacobi_np_batch(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    r1 = np.sqrt((x + MU) ** 2 + y ** 2 + z ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2 + z ** 2)
    Om = 0.5 * (x ** 2 + y ** 2) + OM / r1 + MU / r2
    return 2.0 * Om - (S[:, 3] ** 2 + S[:, 4] ** 2 + S[:, 5] ** 2)


def jac_A(s):
    """State-derivative Jacobian A = d(deriv)/d(state) for the rotating-frame CR3BP.
    A = [[0, I], [G, Ω_v]]; G = Hessian of the pseudo-potential U, Ω_v = Coriolis."""
    x, y, z = s[0], s[1], s[2]
    dx1, dx2 = x + MU, x - 1.0 + MU
    r1 = np.sqrt(dx1 * dx1 + y * y + z * z)
    r2 = np.sqrt(dx2 * dx2 + y * y + z * z)
    m1, m2 = OM, MU
    a1, b1 = m1 / r1 ** 3, 3.0 * m1 / r1 ** 5
    a2, b2 = m2 / r2 ** 3, 3.0 * m2 / r2 ** 5
    Gxx = 1.0 - (a1 + a2) + b1 * dx1 * dx1 + b2 * dx2 * dx2
    Gyy = 1.0 - (a1 + a2) + b1 * y * y + b2 * y * y
    Gzz = 0.0 - (a1 + a2) + b1 * z * z + b2 * z * z
    Gxy = b1 * dx1 * y + b2 * dx2 * y
    Gxz = b1 * dx1 * z + b2 * dx2 * z
    Gyz = b1 * y * z + b2 * y * z
    G = np.array([[Gxx, Gxy, Gxz], [Gxy, Gyy, Gyz], [Gxz, Gyz, Gzz]])
    Ov = np.array([[0.0, 2.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    A = np.zeros((6, 6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = G
    A[3:6, 3:6] = Ov
    return A


def _var_deriv(yv):
    s = yv[:6]
    Phi = yv[6:].reshape(6, 6)
    ds = accel_np(s)                            # [v(3), a(3)]
    dPhi = jac_A(s) @ Phi
    return np.concatenate([ds, dPhi.reshape(-1)])


def _rk4_generic(f, yv, dt):
    k1 = f(yv); k2 = f(yv + 0.5 * dt * k1)
    k3 = f(yv + 0.5 * dt * k2); k4 = f(yv + dt * k3)
    return yv + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def stm_series(s0, dt, N, sample_idx):
    """Integrate the VARIATIONAL equations Φ'=A(t)Φ (Φ(0)=I) alongside the state.
    Accurate for highly-unstable orbits where finite-differencing the flow fails
    (the λ≈2000 unstable direction otherwise swamps the 1/λ stable one). Returns
    {i: (state, Φ(t_i))}."""
    yv = np.concatenate([s0, np.eye(6).reshape(-1)])
    sample_idx = set(sample_idx)
    out = {}
    if 0 in sample_idx:
        out[0] = (yv[:6].copy(), yv[6:].reshape(6, 6).copy())
    for i in range(1, N + 1):
        yv = _rk4_generic(_var_deriv, yv, dt)
        if i in sample_idx:
            out[i] = (yv[:6].copy(), yv[6:].reshape(6, 6).copy())
    return out


def stable_eigvec(M):
    """Real eigenvector of the monodromy matrix for the |λ|<1 (stable) real eigenvalue."""
    w, V = np.linalg.eig(M)
    real = np.abs(w.imag) < 1e-6
    cand = [(abs(w[k]), k) for k in range(len(w)) if real[k] and abs(w[k]) < 0.999]
    cand.sort()                                 # smallest |λ| = most stable
    k = cand[0][1]
    v = np.real(V[:, k])
    return v / np.linalg.norm(v), float(np.real(w[k])), w


def propagate_batch(S, dt, n, record_every=1):
    """Propagate a batch (K,6) for n steps; return array (n//record_every+1, K, 6)."""
    frames = [S.copy()]
    for i in range(1, n + 1):
        S = rk4_np_batch(S, dt)
        if i % record_every == 0:
            frames.append(S.copy())
    return np.array(frames)


def propagate_np(s0, dt, n):
    traj = np.empty((n + 1, 6))
    traj[0] = s0
    s = s0.copy()
    for i in range(n):
        s = rk4_np(s, dt)
        traj[i + 1] = s
    return traj


def check_dynamics():
    """Cross-check the numpy dynamics against the verified JAX cr3bp_sim.accel."""
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    maxerr = 0.0
    for _ in range(200):
        s = rng.normal(size=6) * np.array([1.5, 1.5, 0.3, 1.0, 1.0, 0.3]) + \
            np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
        a_np = accel_np(s)[3:]                 # accel_np returns [v(3), a(3)]
        a_jx = np.asarray(C.accel(jnp.asarray(s)))   # C.accel returns a(3) only
        maxerr = max(maxerr, float(np.max(np.abs(a_np - a_jx))))
    print(f"  numpy-vs-JAX accel max |Δ| over 200 random states: {maxerr:.2e} "
          f"(float32 JAX vs float64 numpy; want <1e-4)")


def orbit_and_monodromy(xL, Ax, dt):
    """Diff-correct the Lyapunov orbit; return (s0, T, N, orbit_pts, M, v_stable, λ)."""
    s0, T, wp, it, res = differential_correct(xL, Ax, dt=dt)
    N = int(round(T / dt))
    orbit = propagate_batch(s0[None, :], dt, N, record_every=1)[:, 0, :]   # (N+1,6)
    ser = stm_series(s0, dt, N, [N])
    M = ser[N][1]
    v_s, lam, w = stable_eigvec(M)
    return s0, T, N, orbit, M, v_s, lam, w, res


def nearest_dist(state, orbit):
    return float(np.min(np.linalg.norm(orbit[:, 0:3] - state[0:3], axis=1)))


def run_manifold(args):
    print("=== R-H2: monodromy eigenstructure + stable manifold ===")
    lp = C.lagrange_points()
    for name, Ax in (("L1", args.ax_l1), ("L2", args.ax_l2)):
        xL = lp[name]
        s0, T, N, orbit, M, v_s, lam, w, res = orbit_and_monodromy(xL, Ax, args.dt)
        mags = np.sort(np.abs(w))[::-1]
        lam_u = mags[0]
        nu = 0.5 * (lam_u + 1.0 / lam_u)
        detM = float(np.linalg.det(M))
        print(f"  {name} (Ax={Ax:.3f}, T={T:.4f}): monodromy |eigs| = "
              f"[{', '.join(f'{m:.3g}' for m in mags)}]")
        print(f"     det M={detM:.6f} (symplectic→1); unstable λ={lam_u:.4g}, "
              f"reciprocal 1/λ={1/lam_u:.3g} vs min|eig|={mags[-1]:.3g}; ν={nu:.4g}")
        print(f"     stable eigenvalue λ_s={lam:.4g}  (|λ_s|={abs(lam):.4g}<1)")
        # ballistic + asymptotic checks on the stable direction
        eps = args.eps
        s_plus = s0 + eps * v_s
        fwd = propagate_batch(s_plus[None, :], args.dt, N, record_every=1)[:, 0, :]
        d0 = nearest_dist(fwd[0], orbit)
        dT = nearest_dist(fwd[-1], orbit)
        Cj = jacobi_np(s_plus)
        Cj_end = jacobi_np(fwd[-1])
        print(f"     stable check: perturb +{eps:g}·v_s, integrate 1 period FORWARD → "
              f"dist-to-orbit {d0:.2e} → {dT:.2e} (shrinks ~1/λ={1/lam_u:.2e})")
        print(f"     Jacobi along manifold: {Cj:.6f} → {Cj_end:.6f} "
              f"(drift {abs(Cj_end-Cj):.1e}, ballistic)")


def moon_rel_np(S):
    """G's verified capture criterion in numpy: Moon-relative distance, inertial speed
    relative to the (stationary) Moon (v_rot + ω×r_rel), osculating Moon energy E."""
    rrelx = S[..., 0] - R_MOON[0]
    rrely = S[..., 1] - R_MOON[1]
    rrelz = S[..., 2] - R_MOON[2]
    vx = S[..., 3] - rrely
    vy = S[..., 4] + rrelx
    vz = S[..., 5]
    d = np.sqrt(rrelx ** 2 + rrely ** 2 + rrelz ** 2)
    speed = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    E = 0.5 * speed ** 2 - MU / np.maximum(d, 1e-9)
    return d, speed, E


def count_moon_revs(traj):
    """Revolutions around the Moon = total unwrapped angle of r_rel / 2π."""
    ang = np.unwrap(np.arctan2(traj[:, 1] - R_MOON[1], traj[:, 0] - R_MOON[0]))
    return abs(ang[-1] - ang[0]) / (2.0 * np.pi)


def manifold_ics(s0, N, v_s, n_seed, pos_disp, dt):
    """Seed the stable manifold: at n_seed points around the orbit, transport v_s by
    Φ(t_i) and offset by ±pos_disp (both branches). Returns (K,6) ICs + a label list."""
    seed_idx = [int(round(k)) for k in np.linspace(0, N, n_seed, endpoint=False)]
    ser = stm_series(s0, dt, N, seed_idx)
    ics = []
    labels = []
    for i in seed_idx:
        X, Phi = ser[i]
        d = Phi @ v_s
        scale = pos_disp / np.linalg.norm(d[0:3])
        for sgn in (+1.0, -1.0):
            ics.append(X + sgn * scale * d)
            labels.append((i, sgn))
    return np.array(ics), labels


def bounded_verify(s_ca, dt, max_steps):
    """From a closest-approach state, propagate ballistically forward; report time
    inside the Hill sphere and Moon revolutions accrued while bound."""
    traj = propagate_batch(s_ca[None, :], dt, max_steps, record_every=1)[:, 0, :]
    d = np.linalg.norm(traj[:, 0:3] - R_MOON, axis=1)
    outside = np.where(d > R_HILL)[0]
    left = outside[0] if len(outside) else len(d)
    bound_time = left * dt
    revs = count_moon_revs(traj[:max(left, 1)])
    return bound_time, revs, float(d.min())


def run_capture(args):
    print("=== R-H3: stable-manifold ballistic-capture sweep ===")
    lp = C.lagrange_points()
    targets = [("L1", args.ax_l1), ("L2", args.ax_l2)]
    for name, Ax in targets:
        xL = lp[name]
        s0, T, N, orbit, M, v_s, lam, w, res = orbit_and_monodromy(xL, Ax, args.dt)
        ics, labels = manifold_ics(s0, N, v_s, args.n_seed, args.pos_disp, args.dt)
        n_prop = int(round(args.t_prop / args.dt))
        # trace the stable manifold BACKWARD (dt<0): away from the orbit toward its origin
        traj = propagate_batch(ics, -args.dt, n_prop, record_every=args.rec_every)
        # traj shape (frames, K, 6)
        C_orbit = jacobi_np(s0)
        best = None
        n_cap = 0
        for k in range(ics.shape[0]):
            tk = traj[:, k, :]
            d, sp, E = moon_rel_np(tk)
            j = int(np.argmin(d))
            d_ca, E_ca = d[j], E[j]
            entered = d.min() < R_HILL
            if entered and E_ca < 0.0:
                n_cap += 1
                # capture window: contiguous E<0 around closest approach
                if best is None or E_ca < best["E_ca"]:
                    best = {"k": k, "label": labels[k], "d_ca": float(d_ca),
                            "E_ca": float(E_ca), "s_ca": tk[j].copy(),
                            "dmin": float(d.min())}
        print(f"  {name} (Ax={Ax:.3f}, C={C_orbit:.5f}, Hill={R_HILL:.3f}): "
              f"{ics.shape[0]} manifold ICs, {n_cap} reach Hill sphere with E_moon<0")
        if best is None:
            print("    no manifold trajectory achieved E_moon<0 inside the Hill "
                  "sphere in this window — honest null for this orbit/amplitude.")
            continue
        bt, revs, dmin_fwd = bounded_verify(best["s_ca"], args.dt, args.verify_steps)
        print(f"    BEST capture: closest {best['d_ca']*C.L_UNIT_KM:.0f} km, "
              f"E_moon={best['E_ca']:+.4f} (<0, BOUND); seed idx={best['label'][0]} "
              f"branch={best['label'][1]:+.0f}")
        print(f"    bounded-prop from closest approach: stays in Hill sphere "
              f"{bt:.3f} nondim (~{bt*C.T_UNIT_S/86400:.2f} d), {revs:.2f} Moon revs, "
              f"min dist {dmin_fwd*C.L_UNIT_KM:.0f} km")
        verdict = "VERIFIED temporary capture" if revs >= 2.0 else \
                  ("partial (K<2 revs)" if revs >= 0.5 else "grazing (not a real capture)")
        print(f"    → {verdict}")


R_EARTH = np.array([-MU, 0.0, 0.0])


def earth_dist(S):
    return np.sqrt((S[..., 0] - R_EARTH[0]) ** 2 + S[..., 1] ** 2 + S[..., 2] ** 2)


def earth_rel_inertial_speed(S):
    """Inertial speed relative to (moving) Earth: v_sc_inertial − v_earth_inertial,
    v_sc_inertial = (vx−y, vy+x, vz), v_earth_inertial = ω×r_earth = (0,−μ,0)."""
    vxi = S[..., 3] - S[..., 1]
    vyi = S[..., 4] + S[..., 0] + MU
    vzi = S[..., 5]
    return np.sqrt(vxi ** 2 + vyi ** 2 + vzi ** 2)


def run_transfer(args):
    print("=== R-H4: Earth-connection of the captured manifold + honest Δv ===")
    lp = C.lagrange_points()
    v_unit = C.L_UNIT_KM / C.T_UNIT_S
    reached_leo = False
    for name, Ax in (("L1", args.ax_l1), ("L2", args.ax_l2)):
        s0, T, N, orbit, M, v_s, lam, w, res = orbit_and_monodromy(lp[name], Ax, args.dt)
        ics, labels = manifold_ics(s0, N, v_s, args.n_seed, args.pos_disp, args.dt)
        n_far = int(round(args.t_far / args.dt))
        traj = propagate_batch(ics, -args.dt, n_far, record_every=args.rec_every)
        best = None
        n_cap = 0
        perigees = []
        for k in range(ics.shape[0]):
            tk = traj[:, k, :]
            dM, spM, EM = moon_rel_np(tk)
            jM = int(np.argmin(dM))
            if not (dM.min() < R_HILL and EM[jM] < 0.0):
                continue                          # only trajectories that get captured
            n_cap += 1
            re = earth_dist(tk)
            jp = int(np.argmin(re))
            rp = re[jp]
            perigees.append(rp)
            v_sc = earth_rel_inertial_speed(tk[jp])
            v_circ = np.sqrt((1.0 - MU) / max(rp, 1e-9))
            inj = abs(v_sc - v_circ)              # tangential injection from circular
            if best is None or rp < best["rp"]:
                best = {"rp": float(rp), "inj": float(inj), "v_sc": float(v_sc),
                        "v_circ": float(v_circ)}
        perigees = np.array(perigees)
        print(f"  {name} (Ax={Ax:.3f}): {n_cap} capturing trajectories traced back "
              f"{args.t_far:.0f} nondim (~{args.t_far*C.T_UNIT_S/86400:.0f} d)")
        if not len(perigees):
            print("    no capturing trajectory in this trace window.")
            continue
        print(f"    Earth-perigee of captured set: min {perigees.min()*C.L_UNIT_KM:.0f} km, "
              f"median {np.median(perigees)*C.L_UNIT_KM:.0f} km (LEO≈6800 km)")
        rp_km = best["rp"] * C.L_UNIT_KM
        print(f"    closest-to-Earth: perigee {rp_km:.0f} km; if injected there "
              f"v_manifold={best['v_sc']*v_unit:.3f} vs v_circ={best['v_circ']*v_unit:.3f} "
              f"→ injection {best['inj']*v_unit:.3f} km/s")
        if rp_km < 12000:
            reached_leo = True
    if not reached_leo:
        print("  → NEITHER the L1 nor L2 pure-CR3BP manifold descends to LEO (perigees "
              "≫ LEO). As pre-registered (assumption #3), bridging to LEO needs a lunar "
              "flyby or the Sun's 4-body term (exactly how flown WSB transfers, e.g. "
              "Hiten, work). HONEST NULL on the Δv beat; H's deliverable is the verified "
              "capture (R-H3) + the manifold engine (R-H1/H2), not a claimed transfer win.")


def run_orbit(args):
    print("=== R-H1: Lyapunov orbits via differential correction ===")
    lp = C.lagrange_points()
    for name, Ax in (("L1", args.ax_l1), ("L2", args.ax_l2)):
        xL = lp[name]
        s0, T, wp, it, res = differential_correct(xL, Ax, dt=args.dt)
        traj = propagate_np(s0, args.dt, int(round(T / args.dt)))
        C0 = jacobi_np(s0); Cend = jacobi_np(traj[-1])
        close = np.linalg.norm(traj[-1] - s0)
        xamp = 0.5 * (traj[:, 0].max() - traj[:, 0].min())
        yamp = traj[:, 1].max()
        print(f"  {name} (x_L={xL:+.5f}, seed Ax={Ax:.3f}): converged in {it} Newton "
              f"steps, |vx_cross|={res:.1e}")
        print(f"     period T={T:.5f} ({T*C.T_UNIT_S/86400:.3f} d)  linear 2π/ω={2*np.pi/wp:.4f}"
              f"  Jacobi C={C0:.6f} drift={abs(Cend-C0):.1e}")
        print(f"     closure |s(T)-s0|={close:.1e}  x-amp={xamp:.4f} y-amp={yamp:.4f} "
              f"({xamp*C.L_UNIT_KM:.0f} km)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="cross-check numpy vs JAX dynamics")
    ap.add_argument("--orbit", action="store_true", help="R-H1: Lyapunov orbits")
    ap.add_argument("--manifold", action="store_true", help="R-H2: monodromy + manifold")
    ap.add_argument("--capture", action="store_true", help="R-H3: capture sweep")
    ap.add_argument("--transfer", action="store_true", help="R-H4: Earth-connection + Δv")
    ap.add_argument("--t-far", type=float, default=20.0)     # long backward trace to Earth
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--ax-l1", type=float, default=0.02)
    ap.add_argument("--ax-l2", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--n-seed", type=int, default=40)
    ap.add_argument("--pos-disp", type=float, default=1e-4)   # ~38 km manifold offset
    ap.add_argument("--t-prop", type=float, default=6.0)      # backward manifold time
    ap.add_argument("--rec-every", type=int, default=5)
    ap.add_argument("--verify-steps", type=int, default=40000)
    args = ap.parse_args()
    if args.check:
        check_dynamics()
    if args.orbit:
        run_orbit(args)
    if args.manifold:
        run_manifold(args)
    if args.capture:
        run_capture(args)
    if args.transfer:
        run_transfer(args)


if __name__ == "__main__":
    main()
