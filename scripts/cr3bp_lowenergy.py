#!/usr/bin/env python3
"""Does the ballistic capture beat Hohmann in the PURE CR3BP? (Build I).

Build H verified a temporary ballistic capture on the L2 stable manifold but claimed
an "honest null" on the transfer beat with "the manifold does not reach LEO → needs
the Sun." That conflates GEOMETRY with COST: you don't need the manifold to reach
LEO, you PATCH onto it with a departure burn. This module tests the actual question —
is the total Δv of a patched LEO→manifold→ballistic-capture transfer below
Hohmann+capture? — instead of asserting it.

  total_manifold = Δv1(LEO→apogee r) + Δv2(match manifold at r);  capture is ballistic
  total_hohmann  = TLI + minimal capture burn to the SAME (d_ca, E_ca) captured state

Steel-man BOTH sides: best manifold insertion point, and the minimal (hardest-to-beat)
Hohmann capture burn.

    uv run --with jax python scripts/cr3bp_lowenergy.py --patched
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "scripts")
import cr3bp_sim as C  # noqa: E402
import cr3bp_manifold as M  # noqa: E402

MU = C.MU
MU_E = 1.0 - MU
MU_M = MU
R_EM = 1.0                                   # Earth-Moon distance (Earth-relative), nondim
V_UNIT = C.L_UNIT_KM / C.T_UNIT_S
R_LUNAR = 1737.4 / C.L_UNIT_KM               # Moon radius (nondim); d_ca below this = collision


def v_circ(mu, r):
    return np.sqrt(mu / r)


def earth_pos_vel(S):
    """Earth-relative position and inertial velocity vectors (…,3) each."""
    px = S[..., 0] - (-MU); py = S[..., 1]; pz = S[..., 2]
    vx = S[..., 3] - S[..., 1]
    vy = S[..., 4] + S[..., 0] + MU
    vz = S[..., 5]
    pos = np.stack([px, py, pz], axis=-1)
    vel = np.stack([vx, vy, vz], axis=-1)
    return pos, vel


def hohmann_capture(r_leo, d_ca, E_ca):
    """Fair steel-manned baseline: Hohmann TLI + MINIMAL burn to bind at the same
    captured periapsis d_ca and target energy E_ca the manifold reaches."""
    v_leo = v_circ(MU_E, r_leo)
    tli = v_leo * (np.sqrt(2.0 * R_EM / (r_leo + R_EM)) - 1.0)
    v_apo = v_circ(MU_E, R_EM) * np.sqrt(2.0 * r_leo / (r_leo + R_EM))   # arrival apogee speed
    v_moon = np.sqrt((MU_E + MU_M) / R_EM)          # Earth-relative Moon speed (two-body, =1)
    v_inf = abs(v_moon - v_apo)                                         # Moon-relative approach
    v_peri_arr = np.sqrt(v_inf ** 2 + 2.0 * MU_M / d_ca)               # hyperbolic periapsis speed
    v_peri_tgt = np.sqrt(max(2.0 * (E_ca + MU_M / d_ca), 0.0))         # speed for E_ca at d_ca
    dv_cap = max(0.0, v_peri_arr - v_peri_tgt)
    return tli + dv_cap, tli, dv_cap, v_inf


def best_insertion(seg, r_leo, r_hi):
    """Cheapest two-impulse LEO→manifold insertion over pre-capture points of one arc.
    Returns (total, dv1, dv2, r_ins) or None."""
    pos, vel = earth_pos_vel(seg)
    r = np.linalg.norm(pos, axis=1)
    mask = (r > r_leo) & (r < r_hi)
    if not mask.any():
        return None
    r = r[mask]; pos = pos[mask]; vel = vel[mask]
    rhat = pos / r[:, None]
    v_r = np.abs((vel * rhat).sum(axis=1))
    v_t = np.sqrt(np.maximum((vel ** 2).sum(axis=1) - v_r ** 2, 0.0))
    v_leo = v_circ(MU_E, r_leo)
    dv1 = v_leo * (np.sqrt(2.0 * r / (r_leo + r)) - 1.0)
    v_apo = v_circ(MU_E, r) * np.sqrt(2.0 * r_leo / (r_leo + r))
    dv2 = np.sqrt(v_r ** 2 + (v_t - v_apo) ** 2)
    total = dv1 + dv2
    k = int(np.argmin(total))
    return float(total[k]), float(dv1[k]), float(dv2[k]), float(r[k])


def batch_bounded_verify(s_ca, dt, steps):
    """Propagate every closest-approach state forward together; per trajectory return
    (Moon revolutions accrued while inside the Hill sphere, bound time)."""
    traj = M.propagate_batch(s_ca, dt, steps, record_every=1)   # (steps+1, K, 6)
    K = s_ca.shape[0]
    revs = np.zeros(K); bt = np.zeros(K)
    for k in range(K):
        tk = traj[:, k, :]
        d = np.linalg.norm(tk[:, 0:3] - M.R_MOON, axis=1)
        out = np.where(d > M.R_HILL)[0]
        left = out[0] if len(out) else len(d)
        bt[k] = left * dt
        revs[k] = M.count_moon_revs(tk[:max(left, 1)])
    return revs, bt


def run_patched(args):
    print(f"=== R-I1: patched LEO→manifold→capture vs Hohmann+capture (r_leo="
          f"{args.r_leo:.4f} = {args.r_leo*C.L_UNIT_KM:.0f} km) ===")
    lp = C.lagrange_points()
    best = None
    for name, Ax in (("L1", args.ax_l1), ("L2", args.ax_l2)):
        s0, T, N, orbit, mono, v_s, lam, w, res = M.orbit_and_monodromy(lp[name], Ax, args.dt)
        ics, labels = M.manifold_ics(s0, N, v_s, args.n_seed, args.pos_disp, args.dt)
        n_far = int(round(args.t_far / args.dt))
        traj = M.propagate_batch(ics, -args.dt, n_far, record_every=args.rec_every)
        n_cap = 0
        n_artifact = 0
        cands = []
        for k in range(ics.shape[0]):
            tk = traj[:, k, :]
            dM, spM, EM = M.moon_rel_np(tk)
            jcap = int(np.argmin(dM))
            if not (dM.min() < M.R_HILL and EM[jcap] < 0.0):
                continue
            d_ca, E_ca = float(dM[jcap]), float(EM[jcap])
            if d_ca < R_LUNAR:                       # sub-surface plunge = collision/artifact
                n_artifact += 1
                continue
            n_cap += 1
            seg = tk[jcap + 1:]                       # pre-capture inbound arc (forward time)
            ins = best_insertion(seg, args.r_leo, args.r_hi)
            if ins is None:
                continue
            tot_m, dv1, dv2, r_ins = ins
            base, tli, dvcap, vinf = hohmann_capture(args.r_leo, d_ca, E_ca)
            cands.append(dict(name=name, tot_m=tot_m, dv1=dv1, dv2=dv2, r_ins=r_ins,
                              base=base, tli=tli, dvcap=dvcap, d_ca=d_ca, E_ca=E_ca,
                              ratio=tot_m / base, s_ca=tk[jcap].copy()))
        # symmetric fairness: batch-verify EVERY sensible arc — a "free capture" only
        # counts if it is a genuine multi-rev bound orbit, the SAME bar Hohmann's paid
        # capture burn clears; otherwise it is a flyby and the comparison is rigged.
        if not cands:
            print(f"  {name}: 0 sensible capturing arcs ({n_artifact} sub-surface "
                  f"artifacts excluded).")
            continue
        s_all = np.array([c["s_ca"] for c in cands])
        revs_all, bt_all = batch_bounded_verify(s_all, args.dt, args.verify_steps)
        for c, rv, bt in zip(cands, revs_all, bt_all):
            c["revs"], c["bound_days"] = float(rv), float(bt) * C.T_UNIT_S / 86400.0
        verified = [c for c in cands if c["revs"] >= 2.0]
        cheapest = min(cands, key=lambda c: c["ratio"])
        n_ver = len(verified)
        print(f"  {name}: {n_cap} sensible capturing arcs ({n_artifact} sub-surface "
              f"artifacts excluded); {n_ver} are GENUINE captures (K>=2 revs), "
              f"{n_cap-n_ver} are flybys (momentary E<0).")
        print(f"     cheapest arc overall: ratio {cheapest['ratio']:.3f} but only "
              f"{cheapest['revs']:.1f} revs — {'a real capture' if cheapest['revs']>=2 else 'a FLYBY, not a capture'}.")
        if not verified:
            print(f"     among GENUINE captures: none — the manifold's cheap transfers "
                  f"are all flybys; no verified-capture beat exists.")
            continue
        loc_best = min(verified, key=lambda c: c["ratio"])
        print(f"     best GENUINE-capture patched transfer ({loc_best['revs']:.1f} revs, "
              f"~{loc_best['bound_days']:.1f} d bound):")
        b = loc_best
        print(f"     manifold total {b['tot_m']*V_UNIT:.3f} km/s = dep {b['dv1']*V_UNIT:.3f} "
              f"+ insert {b['dv2']*V_UNIT:.3f}  (insert at {b['r_ins']*C.L_UNIT_KM:.0f} km, "
              f"capture BALLISTIC)")
        print(f"     Hohmann+capture {b['base']*V_UNIT:.3f} km/s = TLI {b['tli']*V_UNIT:.3f} "
              f"+ min-capture {b['dvcap']*V_UNIT:.3f}  (to d_ca={b['d_ca']*C.L_UNIT_KM:.0f} km, "
              f"E_ca={b['E_ca']:+.3f})")
        print(f"     ratio {b['ratio']:.3f}  →  "
              f"{'manifold BEATS Hohmann' if b['ratio']<0.999 else ('~tie' if b['ratio']<1.001 else 'Hohmann wins')}")
        if best is None or b["ratio"] < best["ratio"]:
            best = b
    print("  ---")
    if best is not None:
        verdict = ("BEAT" if best["ratio"] < 0.98 else
                   "marginal/within-noise" if best["ratio"] < 1.02 else "NULL (Hohmann wins)")
        print(f"  OVERALL best ratio {best['ratio']:.3f} ({best['name']}) → {verdict}. "
              f"Δ = {(best['base']-best['tot_m'])*V_UNIT:+.3f} km/s vs Hohmann+capture.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", action="store_true")
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--ax-l1", type=float, default=0.02)
    ap.add_argument("--ax-l2", type=float, default=0.02)
    ap.add_argument("--n-seed", type=int, default=40)
    ap.add_argument("--pos-disp", type=float, default=1e-4)
    ap.add_argument("--t-far", type=float, default=20.0)
    ap.add_argument("--rec-every", type=int, default=5)
    ap.add_argument("--r-leo", type=float, default=6778.0 / C.L_UNIT_KM)   # ~400 km LEO
    ap.add_argument("--r-hi", type=float, default=1.05)                    # max insertion radius
    ap.add_argument("--verify-steps", type=int, default=80000)             # ~4 nondim bound check
    args = ap.parse_args()
    if args.patched:
        run_patched(args)


if __name__ == "__main__":
    main()
