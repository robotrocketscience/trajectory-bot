#!/usr/bin/env python3
"""Amortized mission-planner — learn the (mission → warm-start) map (Build N, R-N13).

The seven prior rounds built the gravity-assist TOOLKIT (R-N10 enumerate → R-N8 flyby-node optimize →
R-N9 primer certify → R-N12 phase), but every mission is still solved FROM SCRATCH. The north-star
generalization is amortization: learn the map (mission parameters) → (near-optimal departure decision
variables) so a NEW mission is solved by cheap inference + a few diff-sim refinement steps, not a full
re-search.

Problem family (offline, CI-safe, coplanar): aim an Earth-departure transfer at a Jupiter flyby point,
under the REAL perturbed Sun+Jupiter dynamics (the diff-sim N-body engine). A mission is θ=(Jupiter arrival
angle, TOF). Ground-truth labels come from a backtracking Gauss-Newton solve through the differentiable
rollout (Lambert seed → converged departure velocity). We train a tiny MLP θ→v_dep* and test:

  H-N13a  generalization: test error ≈ train error and ≪ mean / nearest-neighbour baselines.
  H-N13b  amortization pays: MLP warm-start needs fewer refinement steps than a cold Lambert seed.
  H-N13c  the honest boundary: the solution-map sensitivity ‖∂r_end/∂v‖ and conditioning EXPLODE once the
          target crosses past closest approach (R-N7's razor basin), so amortization is regime-bounded —
          learnable in the smooth pre-flyby regime, not in the chaotic post-flyby one.

    uv run --with jax python scripts/amortized_planner.py --verify        # offline, CI-safe
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import lambert as LAM                  # noqa: E402
import nbody_sim as NB                 # noqa: E402

MU_S = NB.GM["sun"]
MU_J = NB.GM["jupiter"]
AU = NB.AU
DAY = NB.DAY
R_J = 5.2028 * AU
W_J = np.sqrt(MU_S / R_J ** 3)
R_JUP = 71492.0
B_OFF = 300.0 * R_JUP                  # flyby aim offset (perp) — inside Jupiter's SOI, no plunge
SOFT = 3.0 * R_JUP                     # Plummer softening (bounds the gradient through the pass)
N = 1500                               # RK4 steps per transfer
TOL_KM = 1.0                           # convergence tolerance on the terminal miss (km)
BODY_GM = jnp.array([MU_S, MU_J])
R1 = np.array([AU, 0.0, 0.0])          # Earth at departure (frame-fixed)
V_EARTH = np.array([0.0, np.sqrt(MU_S / AU), 0.0])
THA = (150.0, 210.0)                   # mission family: Jupiter arrival angle (deg)
TOFY = (2.8, 3.8)                      #                 time of flight (yr)


def jup_seq(theta_arr, tof, horizon=None, n=N):
    """Sun+Jupiter body positions over the rollout, and the flyby aim target at arrival.
    horizon (≥tof) extends the propagation past arrival for the H-N13c sensitivity probe."""
    horizon = tof if horizon is None else horizon
    ts = np.linspace(0.0, horizon, n, endpoint=False)
    ang = theta_arr - W_J * (tof - ts)
    jup = np.stack([R_J * np.cos(ang), R_J * np.sin(ang), np.zeros_like(ang)], axis=1)
    body_seq = np.stack([np.zeros((n, 3)), jup], axis=1)          # Sun at origin, Jupiter moving
    rj_arr = R_J * np.array([np.cos(theta_arr), np.sin(theta_arr), 0.0])
    perp = np.array([-np.sin(theta_arr), np.cos(theta_arr), 0.0])
    target = (rj_arr + B_OFF * perp)[:2]
    return jnp.asarray(body_seq), jnp.asarray(target), horizon


@jax.jit
def _endpoint(vdep2, body_seq, dt):
    v = jnp.array([vdep2[0], vdep2[1], 0.0])
    rv0 = jnp.concatenate([jnp.asarray(R1), v])
    rvT, _ = NB.rollout(rv0, body_seq, BODY_GM, dt, soft=SOFT)
    return rvT[:2]


def lambert_seed(theta_arr, tof):
    _, target, _ = jup_seq(theta_arr, tof)
    tgt3 = jnp.array([target[0], target[1], 0.0])
    vd, _ = LAM.lambert(jnp.asarray(R1), tgt3, tof, mu=MU_S)
    return np.asarray(vd)[:2]


def gauss_newton(theta_arr, tof, v0, iters=25, tol=TOL_KM):
    """Backtracking Gauss-Newton on the terminal miss through the perturbed rollout."""
    body_seq, target, _ = jup_seq(theta_arr, tof)
    dt = tof / N
    v = jnp.asarray(v0)
    miss = float(jnp.linalg.norm(_endpoint(v, body_seq, dt) - target))
    hist = [miss]
    for _ in range(iters):
        if miss < tol:
            break
        r = _endpoint(v, body_seq, dt) - target
        J = jax.jacfwd(lambda vv: _endpoint(vv, body_seq, dt))(v)
        dv = jnp.linalg.solve(J, r)
        # backtracking line search: only accept a step that decreases the miss, and keep
        # (v, miss) consistent — the accepted pair always corresponds to the same step.
        step, improved = 1.0, False
        while step > 1e-4:
            vn = v - step * dv
            mn = float(jnp.linalg.norm(_endpoint(vn, body_seq, dt) - target))
            if mn < miss:
                v, miss, improved = vn, mn, True
                break
            step *= 0.5
        hist.append(miss)
        if not improved:                          # stalled — no descent step found
            break
    return np.asarray(v), hist


def make_dataset(n, rng):
    ths = rng.uniform(*THA, n)
    tfs = rng.uniform(*TOFY, n)
    X, Y, seed_miss, corr, final_miss = [], [], [], [], []
    for th, tf in zip(ths, tfs):
        thr, tof = np.radians(th), tf * 365.25 * DAY
        vs = lambert_seed(thr, tof)
        vstar, hist = gauss_newton(thr, tof, vs)
        if hist[-1] < TOL_KM and np.all(np.isfinite(vstar)):
            X.append([th, tf])
            Y.append(vstar)
            seed_miss.append(hist[0])
            corr.append(np.linalg.norm(vstar - vs))
            final_miss.append(hist[-1])
    return (np.array(X), np.array(Y), np.array(seed_miss),
            np.array(corr), np.array(final_miss))


# ---------- tiny MLP (pure JAX, hand-rolled Adam) ----------
def mlp_init(key, h=64):
    ks = jax.random.split(key, 3)
    def lyr(k, i, o):
        return (jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i), jnp.zeros(o))
    return [lyr(ks[0], 2, h), lyr(ks[1], h, h), lyr(ks[2], h, 2)]


def mlp_fwd(p, x):
    for (W, b) in p[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = p[-1]
    return x @ W + b


def mlp_train(p, X, Y, iters=3000, lr=3e-3):
    def loss(p, x, y):
        return jnp.mean((mlp_fwd(p, x) - y) ** 2)
    gl = jax.jit(jax.value_and_grad(loss))
    m = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in p]
    v = [(jnp.zeros_like(W), jnp.zeros_like(b)) for W, b in p]
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    for it in range(iters):
        _, g = gl(p, Xj, Yj)
        t = it + 1
        np_, nm, nv = [], [], []
        for (W, b), (gW, gb), (mW, mb), (vW, vb) in zip(p, g, m, v):
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb ** 2
            W = W - lr * (mW / (1 - b1 ** t)) / (jnp.sqrt(vW / (1 - b2 ** t)) + eps)
            b = b - lr * (mb / (1 - b1 ** t)) / (jnp.sqrt(vb / (1 - b2 ** t)) + eps)
            np_.append((W, b))
            nm.append((mW, mb))
            nv.append((vW, vb))
        p, m, v = np_, nm, nv
    return p


def verify(args):
    print("=== R-N13: amortized mission-planner — learn the (mission → warm-start) map (offline) ===")
    rng = np.random.default_rng(0)
    X, Y, seed_miss, corr, final_miss = make_dataset(args.ndata, rng)
    print(f"  dataset: {len(X)}/{args.ndata} missions solved to <{TOL_KM:.0f} km (median final miss "
          f"{np.median(final_miss):.2e} km) through the perturbed Sun+Jupiter rollout;\n"
          f"           Lambert-seed miss median {np.median(seed_miss):.2e} km, "
          f"Lambert→optimal correction median {np.median(corr):.3f} km/s (the amortization headroom)")

    # ---- H-N13a: generalization ----
    ntr = int(0.75 * len(X))
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    xm, xs, ym, ysd = Xtr.mean(0), Xtr.std(0), Ytr.mean(0), Ytr.std(0)
    p = mlp_train(mlp_init(jax.random.PRNGKey(0)),
                  (Xtr - xm) / xs, (Ytr - ym) / ysd, iters=args.iters)

    def predict(Xraw):
        return np.asarray(mlp_fwd(p, jnp.asarray((Xraw - xm) / xs))) * ysd + ym
    Ptr, Pte = predict(Xtr), predict(Xte)

    def rmse(P, Yt):
        return float(np.sqrt(np.mean(np.sum((P - Yt) ** 2, 1))))
    tr, te = rmse(Ptr, Ytr), rmse(Pte, Yte)
    mean_bl = rmse(np.repeat(Ytr.mean(0)[None], len(Yte), 0), Yte)
    nn = np.array([Ytr[np.argmin(np.sum((((Xtr - xm) / xs) - (xt - xm) / xs) ** 2, 1))] for xt in Xte])
    nn_bl = rmse(nn, Yte)
    a_ok = te < 3.0 * tr and te < 0.5 * nn_bl
    print(f"  H-N13a: MLP train RMSE {tr:.4f} km/s | test {te:.4f} km/s | "
          f"mean-baseline {mean_bl:.3f} | nearest-neighbour {nn_bl:.3f}")
    print(f"          → test ≈ {te/tr:.1f}× train and {nn_bl/te:.0f}× better than NN lookup "
          f"→ {'SUPPORTED' if a_ok else 'REFUTED'} (the map generalizes; not memorization)")

    # ---- H-N13b: amortization pays — warm-start vs cold Lambert seed ----
    warm0, cold0, warm_s, cold_s = [], [], [], []
    for (th, tf), yhat in zip(Xte, Pte):
        thr, tof = np.radians(th), tf * 365.25 * DAY
        vs = lambert_seed(thr, tof)
        bs, tg, _ = jup_seq(thr, tof)
        dt = tof / N
        cold0.append(float(jnp.linalg.norm(_endpoint(jnp.asarray(vs), bs, dt) - tg)))
        warm0.append(float(jnp.linalg.norm(_endpoint(jnp.asarray(yhat), bs, dt) - tg)))
        cold_s.append(len(gauss_newton(thr, tof, vs, tol=TOL_KM)[1]) - 1)
        warm_s.append(len(gauss_newton(thr, tof, yhat, tol=TOL_KM)[1]) - 1)
    b_ok = np.median(warm_s) < np.median(cold_s) and np.median(warm0) < np.median(cold0)
    print(f"  H-N13b: warm-start miss@0 median {np.median(warm0):.2e} km vs cold Lambert "
          f"{np.median(cold0):.2e} km ({np.median(cold0)/max(np.median(warm0),1):.0f}× lower)")
    print(f"          refinement steps to <{TOL_KM:.0f} km: warm median {np.median(warm_s):.1f} vs "
          f"cold {np.median(cold_s):.1f} → {'SUPPORTED' if b_ok else 'REFUTED'} "
          f"(inference + fewer diff-sim steps replaces the full search)")

    # ---- H-N13c: the amortizability boundary — sensitivity explodes past the flyby ----
    print("  H-N13c: solution-map sensitivity ‖∂r_end/∂v‖ and conditioning vs target horizon "
          "(pre→post flyby):")
    horizons = [0.0, 0.5, 1.0, 2.0, 3.0]
    Jn = {h: [] for h in horizons}
    Cn = {h: [] for h in horizons}
    for (th, tf), yhat in list(zip(Xte, Pte))[:args.nsens]:
        thr, tof = np.radians(th), tf * 365.25 * DAY
        for ex in horizons:
            H = tof + ex * 365.25 * DAY
            bs, _, _ = jup_seq(thr, tof, horizon=H)
            J = np.asarray(jax.jacfwd(lambda vv: _endpoint(vv, bs, H / N))(jnp.asarray(yhat)))
            s = np.linalg.svd(J, compute_uv=False)
            Jn[ex].append(s[0])
            Cn[ex].append(s[0] / s[1])
    j0, jf = np.median(Jn[0.0]), np.median(Jn[3.0])
    c0, cf = np.median(Cn[0.0]), np.median(Cn[3.0])
    for ex in horizons:
        tag = "pre-flyby (aim)" if ex == 0.0 else f"+{ex:.1f}yr post-flyby"
        print(f"     horizon {tag:>20}:  ‖J‖ median {np.median(Jn[ex]):.2e}  "
              f"cond median {np.median(Cn[ex]):.2e}")
    c_ok = jf > 20 * j0 and cf > 20 * c0
    print(f"          → past the flyby ‖J‖ grows {jf/j0:.0f}× and conditioning {cf/c0:.0f}× "
          f"→ {'SUPPORTED' if c_ok else 'REFUTED'}: the map's Lipschitz constant blows up, so")
    print("          amortization is REGIME-BOUNDED — learnable in the smooth pre-flyby regime (H-N13a/b),")
    print("          NOT in R-N7's post-flyby razor basin. The learned prior inherits R-N7's discontinuity.")
    print(f"  → verdicts: H-N13a {'SUPPORTED' if a_ok else 'REFUTED'}, "
          f"H-N13b {'SUPPORTED' if b_ok else 'REFUTED'}, H-N13c {'SUPPORTED' if c_ok else 'REFUTED'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--ndata", type=int, default=180)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--nsens", type=int, default=8)
    args = ap.parse_args()
    if args.verify:
        verify(args)


if __name__ == "__main__":
    main()
