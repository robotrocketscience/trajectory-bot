# JAX/XLA diff-sim port + 3D from-scratch learnability

Status: `feat/jax-rollout` (stacked on `feat/attitude-3d` / PR #4).
Experiment code: `scripts/jaxsim.py`, `scripts/jax_parity.py`, `scripts/jax_from_torch.py`.

## Why

The 3D differentiable-sim training rollout is **launch-bound**: ~900 sequential RK4
substeps per rollout (H=60 decisions × 20 substeps), each firing many tiny
quaternion kernels; the GPU sat at ~22% utilization. Fusing the quaternion rotate
bought 2× (18.8→9.6 s/iter) but the serial-launch overhead still dominated. The
right lever for a launch-bound sequential loop is XLA: `jax.lax.scan` + `jit`
compile the whole rollout into a handful of kernels.

## The port (and the number)

`scripts/jaxsim.py` is a faithful port of the Circularize-3D hot path — quaternion
math, coupled 13-state RK4, the decision-layer pointing controller, orbital
elements, observation, and the differentiable rollout — with the rollout expressed
as nested `lax.scan` and training via `jit(value_and_grad)`.

| | s/iter (H=60, B=256, RTX 3060) |
|---|---|
| torch (fused qrotate) | 13.3 |
| **JAX (lax.scan + jit)** | **0.265** |
| **speedup** | **~50×** |

One-time XLA compile ~9 s. Debug/iterate cycles dropped from ~90 min → ~90 s, which
is what made the diagnosis below tractable.

**Numerical parity** (`scripts/jax_parity.py`): one RK4 step + elements vs the torch
reference on a fixed state — rk4 `max|Δ| = 6e-8` (rel 8e-12), `a`/`e` bit-exact.

**Record correction:** JAX/XLA — not SHAC — is the wall-clock lever. SHAC shortens
the *backward* chain; the *forward* launch-bound rollout dominated, so SHAC gave ~0×.
XLA attacks the forward directly.

## From-scratch 3D learnability — the diagnosis chain

Question: can from-scratch diff-sim *learn* 3D circularization (prerequisite to
asking whether RL beats the analytic/expert maneuver)? Findings, in order:

1. **Objective was mis-scaled, not just hard.** On the env distribution the original
   terminal objective `orbit_err + 0.5·Δv` actually *prefers coasting*: the available
   orbit-error reduction (~0.27) ≈ the Δv penalty it costs (~0.57), so only a weak
   shaping term tipped the loss toward success. Fixed: orbit-dominant weight, Δv
   demoted to a tiebreaker, Ng-1999 potential shaping (Φ=−orbit_err), smooth success
   well, and a **fractional** crash penalty `relu((R−r)/R)²` (the km² version blew the
   loss to ~1e6).

2. **NaN wall = long-BPTT gradient explosion** (not an op bug). Unrolled H=4 has a
   finite gradient; H=20 *descends* (loss 2.13→2.00) then dies; H=40/H=60 produce a
   non-finite gradient on ~every batch. The Δv budget (2.0 km/s) rules out escape/
   overflow — the states stay bounded. This is the Metz-2021 differentiable-sim
   pathology. The forward loss is finite at the iter the *gradient* first blows, so
   the guard must check **gradient** finiteness, not loss.

3. **Fix = truncated BPTT.** Full H=60 forward rollout, but `stop_gradient` the state
   every K=10 decisions so the backward chain is capped. The potential shaping
   telescopes to the terminal orbit error, so it acts as an *exact-potential* analogue
   of SHAC's learned-critic bootstrap. Stable: `skipped=3/300`.

4. **From-scratch still 0% = the coast basin.** With numerics *and* objective correct,
   from-scratch sits at 0%: it burns randomly, crashes/worsens orbits, and the
   gradient pushes it back to coasting. Deterministic analytic policy gradient has no
   sampling, so nothing pushes it to *initiate* a burn when the first mis-pointed burn
   only hurts.

5. **Warm-start can't beat the expert.** Loading the DAgger policy (~78–80%) into JAX
   (`scripts/jax_from_torch.py`) and continuing with truncated BPTT: it holds ~80%
   (blip 81.45%) at gentle LR and craters-then-recovers to ~73% at higher LR, but
   never meaningfully exceeds the start. Loss drops while success doesn't — the
   objective is only loosely coupled to success. **Imitation (~80%) is the ceiling for
   this diff-sim approach.**

## Open question / next levers

Beating the analytic/expert maneuver (the "is Hohmann optimal?" research angle) is
**not** reachable by refinement here. The paths that could still get a "yes":
- **Stochastic exploration** (reparameterized action noise + entropy, SAPO-style) so
  from-scratch can discover a *qualitatively different* maneuver — the real prize.
- **Learned-critic SHAC** for genuine long-range burn-timing credit (vs the myopic
  K=10 exact-potential truncation).
- **Fix the loss↔success gap** so lower loss actually means more successes.

All are now fast to try in JAX (~90 s/run).
