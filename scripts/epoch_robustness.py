#!/usr/bin/env python3
"""Is the discovered pump+crank tour TYPICAL across launch epochs, or a lucky epoch? (Build N, R-N39).

The entire hard-constrained discovery arc (R-N36 chain, R-N37 depth to v inf 16.27, R-N38 crank to 27.09 deg)
ran at ONE launch epoch (start+400 d). R-N30 found only 40-53% of epochs have usable single handoffs, and
R-N32's greedy stalled at 0 legs at one epoch — per-epoch variability is KNOWN to be large. This round sweeps
8 launch epochs across ~2 Earth-Venus synodic cycles and reports the DISTRIBUTION of tour depth, final v inf,
and crank availability. ONE knob: the launch epoch (architecture + search bounds held at the R-N37/R-N38
configuration).

Per epoch: (1) the R-N37 GREEDY chain (greedy == beam there within bounds — a disclosed proxy) -> depth, final
v inf; (2) at the epoch's saturated node, ONE dense phi-aware crank enumeration (R-N38's fixed enumerator) ->
count of closed inclination-RAISING resonant returns (crank availability; the full 8-step walk per epoch is
compute-prohibitive).

  H-N39a  the deep pump is TYPICAL: >= 50% of epochs reach final v inf >= 12 (R-N33 level) with >= 3 flybys.
  H-N39b  the typical pump is at least R-N36-level: MEDIAN final v inf >= 10 km/s.
  H-N39c  crank availability is GENERIC: >= 50% of saturated nodes have >= 1 closed inclination-raising return.

Runtime note: the sweep is chunkable across processes (--epochs subsets append JSON rows via --rows FILE;
--verdict reads the accumulated rows) with IDENTICAL per-epoch configuration — disclosed in the pre-reg.
Exactly ballistic, Sun-only two-body legs, patched-conic flybys, real cached-JPL ephemeris. Mechanism/
DISCOVERY study, never a Delta-v beat (418e2e2).

    uv run --with jax --with astroquery --with astropy python scripts/epoch_robustness.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
jax.config.update("jax_enable_x64", True)

import beam_constrained_tour as B        # noqa: E402  (R-N37 greedy/beam machinery)
import constrained_tour_discovery as C   # noqa: E402  (R-N36 architecture)
import crank_walk as K                   # noqa: E402  (R-N38 dense phi-aware crank enumerator)

EPOCH_OFFSETS = [100.0, 250.0, 400.0, 550.0, 700.0, 850.0, 1000.0, 1150.0]   # ~2 synodic cycles; 400 = reference
LAUNCH_VMAX = 8.0    # usable launch v inf cap (km/s) = R-N30's pre-registered usable window [2, 8]. Without it
#                      the min-|v inf| launch search at unfavorable epochs returns ENORMOUS launches (17+ km/s,
#                      0 flybys) whose "final v inf" is BOUGHT at launch, not pumped — an instrument artifact
#                      caught on the first sweep rows and fixed BEFORE verdicts (disclosed in the campaign log).


LAUNCH_TOFS = (90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0)
# R-N30's porkchop tof span (90-340 d). The R-N36 grid (4 tofs, 120-240) misses epochs whose cheap window
# needs a longer/shorter transfer — a second instrument bound caught on the capped rows and widened BEFORE
# verdicts. Jitted launch GN keeps the 9x12 search fast.


def _make_launch():
    from jax import jacfwd, jit
    def res(u, jd0, tof):
        miss, _ = C.shoot("earth", "venus", jd0, u[2] * C.unit_dir(u[0], u[1]), tof)
        return miss / 1e6

    @jit
    def gn(u0, jd0, tof):
        sm = jnp.array([0.4, 0.2, 1.5])
        def body(_, u):
            r = res(u, jd0, tof)
            J = jacfwd(res)(u, jd0, tof)
            JTJ = J.T @ J
            JTJ = JTJ + 1e-3 * jnp.diag(jnp.diag(JTJ)) + 1e-12 * jnp.eye(3)
            du = jnp.linalg.solve(JTJ, J.T @ r)
            return u - jnp.clip(du, -sm, sm)
        u = jax.lax.fori_loop(0, 40, body, u0)
        return u, jnp.linalg.norm(res(u, jd0, tof)) * 1e6

    @jit
    def arr(u, jd0, tof):
        _, va = C.shoot("earth", "venus", jd0, u[2] * C.unit_dir(u[0], u[1]), tof)
        return va
    return gn, arr


_LAUNCH = None


def _capped_launch(t0):
    """Min-|v inf| closed launch with vmag <= LAUNCH_VMAX, over R-N30's tof span. None if no viable launch."""
    global _LAUNCH
    if _LAUNCH is None:
        _LAUNCH = _make_launch()
    gn, arr = _LAUNCH
    b = None
    for lt in LAUNCH_TOFS:
        for th in np.linspace(0, 2 * np.pi, 12, endpoint=False):
            u, miss = gn(jnp.array([th, 0.0, 4.0]), jnp.float64(t0), jnp.float64(lt))
            vmag = float(u[2])
            if float(miss) < C.SOI_KM["venus"] and 0.0 < vmag <= LAUNCH_VMAX and (b is None or vmag < b[2]):
                b = (u, float(miss), vmag, arr(u, jnp.float64(t0), jnp.float64(lt)), lt)
    return b


def sweep_epoch(t0):
    """One epoch: capped-launch greedy chain -> (depth, final v inf); dense crank enumeration at saturation."""
    b = _capped_launch(t0)
    if b is None:
        return {"depth": 0, "vfinal": 0.0, "seed": 0.0, "n_raise": 0, "i_in": 0.0, "viable": False}
    u_l, miss_l, seed_v, vinf_vec, lt = b
    root = {"at": "venus", "jd": t0 + lt, "vin": vinf_vec, "legs": [],
            "mag": float(jnp.linalg.norm(vinf_vec)), "seed_v": seed_v, "launch_tof": lt, "launch_miss": miss_l}
    beam = [root]
    best = root
    for _ in range(B.MAX_FLYBYS):
        children = B.expand(beam)
        if not children:
            break
        children.sort(key=lambda n: -n["mag"])
        beam = [children[0]]                                 # greedy (the R-N37 beam proxy)
        if beam[0]["mag"] > best["mag"]:
            best = beam[0]
    legs = best["legs"]
    depth = len(legs)
    vfinal = best["mag"]
    # saturated node = just after the last pumping leg
    jd = root["jd"]
    vin = root["vin"]
    at = "venus"
    prev = root["mag"]
    sat_idx = 0
    for i, lg in enumerate(legs):
        if lg["arr_mag"] > prev + 0.05:
            sat_idx = i
        prev = lg["arr_mag"]
    for lg in legs[:sat_idx + 1]:
        jd = jd + lg["tof"]
        vin = lg["vinf_arr"]
        at = lg["to"]
    rV, vV = C.rv_p(at, jd)
    hV = jnp.cross(rV, vV)
    i_in = K._ang_deg(jnp.cross(rV, vV + vin), hV)
    n_raise = 0
    if depth >= 1:
        for s in K.crank_continuations(at, jd, vin):
            vout = C.rodrigues(vin, s["dmax"] * jnp.tanh(s["u"][0]), s["u"][1])
            i_out = K._ang_deg(jnp.cross(rV, vV + vout), hV)
            if i_out > i_in + 0.5:
                n_raise += 1
    return {"depth": depth, "vfinal": round(vfinal, 2), "seed": round(root["seed_v"], 2),
            "n_raise": n_raise, "i_in": round(i_in, 2), "viable": True}


def verdicts(rows):
    vf = np.array([r["vfinal"] for r in rows])
    a_frac = float(np.mean([(r["vfinal"] >= 12.0 and r["depth"] >= 3) for r in rows]))
    med = float(np.median(vf))
    # H-N39c denominator = epochs WITH a saturated node (the crank enumeration ran: viable, depth >= 1) —
    # per the pre-registration's "epochs' saturated nodes" (CodeRabbit caught the all-epochs denominator).
    sat_rows = [r for r in rows if r.get("viable", True) and r["depth"] >= 1]
    c_frac = float(np.mean([(r["n_raise"] >= 1) for r in sat_rows])) if sat_rows else 0.0
    a_ok, b_ok, c_ok = a_frac >= 0.5, med >= 10.0, c_frac >= 0.5
    print(f"\n  distribution over {len(rows)} epochs: final v inf min/median/max = "
          f"{vf.min():.2f}/{med:.2f}/{vf.max():.2f} km/s; depth min/median/max = "
          f"{min(r['depth'] for r in rows)}/{int(np.median([r['depth'] for r in rows]))}/"
          f"{max(r['depth'] for r in rows)} flybys.")
    print(f"\n  → H-N39a {'SUPPORTED' if a_ok else 'REFUTED'}: {100 * a_frac:.0f}% of epochs reach final "
          f"v inf ≥ 12 with ≥ 3 flybys ({'≥' if a_ok else '<'} 50%) — the deep pump is "
          f"{'TYPICAL' if a_ok else 'EPOCH-FRAGILE (the 16.27 tour was a lucky epoch; consolidation claims must say so)'}.")
    print(f"  → H-N39b {'SUPPORTED' if b_ok else 'REFUTED'}: median final v inf {med:.2f} "
          f"{'≥' if b_ok else '<'} 10 km/s — the typical epoch pumps "
          f"{'at least to the R-N36 level' if b_ok else 'well below the demonstrated tours'}.")
    print(f"  → H-N39c {'SUPPORTED' if c_ok else 'REFUTED'}: {100 * c_frac:.0f}% of saturated nodes "
          f"({sum(r['n_raise'] >= 1 for r in sat_rows)}/{len(sat_rows)} epochs that reached one) have ≥ 1 "
          f"closed inclination-raising return ({'≥' if c_ok else '<'} 50%) — crank availability is "
          f"{'GENERIC' if c_ok else 'epoch-dependent'}.")
    print(f"\n  → verdicts: H-N39a {'SUPPORTED' if a_ok else 'REFUTED'}, H-N39b {'SUPPORTED' if b_ok else 'REFUTED'}, "
          f"H-N39c {'SUPPORTED' if c_ok else 'REFUTED'}")
    print("  NET: read with the full distribution above (the verdict bars are the pre-registered falsifiers, the")
    print("    distribution is the finding). Greedy is the beam proxy (R-N37: greedy ≡ beam at the reference —")
    print("    epochs where greedy stalls shallow are bounds, not proofs of absence). One crank STEP proxies")
    print("    availability, not the full walk. Exactly ballistic, Sun-only legs, patched-conic, cached JPL;")
    print("    identical compute bounds at every epoch. Mechanism/DISCOVERY study, never a Δv beat (418e2e2).")


def verify(args):
    print("=== R-N39: is the discovered pump+crank tour TYPICAL across launch epochs, or lucky? ===")
    if not C.F._require_cache():
        return
    sjd = C.F._start_jd()
    for p in ("earth", "venus"):
        C._tab(p)
    offsets = [float(x) for x in args.epochs.split(",")] if args.epochs else EPOCH_OFFSETS
    rows = []
    if args.rows and os.path.exists(args.rows):
        with open(args.rows) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        print(f"  loaded {len(rows)} prior rows from {args.rows}")
    if not args.verdict_only:
        print(f"  sweeping epochs (offsets d): {[int(o) for o in offsets]}")
        for off in offsets:
            r = sweep_epoch(sjd + off)
            r["offset"] = int(off)
            rows.append(r)
            if r.get("viable", True):
                print(f"    t0+{int(off):5d}d: depth {r['depth']} flybys, seed {r['seed']:5.2f} -> final v inf "
                      f"{r['vfinal']:6.2f}, saturated i_in {r['i_in']:5.2f} deg, "
                      f"{r['n_raise']} inclination-raising closed returns", flush=True)
            else:
                print(f"    t0+{int(off):5d}d: NO viable launch (no closed Earth->Venus with v inf <= "
                      f"{LAUNCH_VMAX:.0f} km/s in the R-N36 tof grid) — counts against all metrics", flush=True)
            if args.rows:
                with open(args.rows, "a") as f:
                    f.write(json.dumps(r) + "\n")
    rows = list({r["offset"]: r for r in rows}.values())     # dedupe by offset (chunked reruns)
    if len(rows) >= len(EPOCH_OFFSETS) or args.verdict_only:
        verdicts(sorted(rows, key=lambda r: r["offset"]))
    else:
        print(f"\n  partial sweep ({len(rows)} rows) — run remaining epochs, then --verdict-only with --rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--epochs", default="", help="comma-separated epoch offsets (d); default = full sweep")
    ap.add_argument("--rows", default="", help="JSONL file to append/load per-epoch rows (chunked runs)")
    ap.add_argument("--verdict-only", action="store_true", help="skip sweeping; verdicts from --rows")
    args = ap.parse_args()
    if args.verify or args.verdict_only:
        verify(args)


if __name__ == "__main__":
    main()
