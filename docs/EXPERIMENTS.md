# Experiments — Milestone 1: 2-D Circularization

This is the running results log for the revived project. Milestone 1 is the
minimal case — circularize a spacecraft from an elliptical orbit to a circular
orbit at the apoapsis radius, in a plane, around one body (Earth), using as
little Δv as possible. It exists to get the RL loop correct before extending to
3-D attitude control and multi-body transfers.

## The task

- **Dynamics:** planar two-body gravity (Earth, μ = 398600.4418 km³/s²),
  integrated with RK4 (`tbot/dynamics.py`).
- **Start:** an elliptical orbit — periapsis altitude 400–800 km, apoapsis/periapsis
  ratio 1.3–2.5 — entered at a **random true anomaly** and random orientation.
- **Goal:** reach a circular orbit at the initial apoapsis radius `r_a`
  (eccentricity `e < 0.05` and `|a − r_a|/r_a < 0.05`).
- **Baseline (the number to beat):** the analytic single-impulse apoapsis
  circularization Δv (`tbot/orbital.circularize_apoapsis_dv`). This is the
  theoretical optimum for this maneuver.
- **Action:** continuous, velocity-frame throttle `[tangential, radial]`, held
  over `decision_repeat = 20` physics substeps (the agent decides every 200 s).
  This decouples decision frequency from the fine integrator so the decisive
  burn is a few consequential decisions, not a needle in a ~2000-step horizon.
- **Reward:** potential-based shaping (policy-invariant) toward circular, minus a
  Δv (fuel) cost and a small per-step living cost, plus a large terminal bonus
  for finishing in-tolerance; escape/hyperbolic and impact terminate as failures.

The env is `tbot/envs/circularize2d.py` (Gymnasium). It is pyright-strict typed
and covered by tests (`tests/`), including a solvability proof: a hand-coded
"coast to apoapsis, burn prograde to circular speed" controller solves it for
every seed at Δv ≈ the analytic baseline.

## Scoreboard

| Method | Success rate | Δv vs optimal | Notes |
|---|---|---|---|
| **Scripted expert** (hand-coded) | **100%** | **1.00** | proves the task is solvable with the physics |
| Cold PPO (model-free, 8 runs) | 0–3% | — (2% successes ≈ 1.1) | never learns the precise burn |
| Behavior cloning (BC) | ~16% | ~1.37 | capped by covariate shift |
| DAgger (BC + on-policy relabel) | ~45% (peak 48.5%) | ~1.5 | plateaus below the expert |
| **Differentiable-sim policy gradient** | **81%** | **1.26** | exact gradients through the rollout — beats every learned method |

Success is over 100–200 held-out random orbits; Δv ratio is agent Δv / analytic
baseline (1.0 = optimal). Trained on a single RTX 3060 (Torch + CUDA).

## What each method showed

### Scripted expert — 100%
`scripts/dagger.py::Expert` (also the env's solvability test): coast, latch
"burning" on the apoapsis crossing, burn prograde with the throttle that reaches
the target circular speed. Solves every orbit at Δv ≈ optimal. This is the
reference the learned methods are measured against — and proof the environment,
dynamics, and reward are correct.

### Cold model-free PPO — 0–3% (8 runs)
`scripts/train_circularize2d.py`. Stable-Baselines3 PPO, no demonstrations.
Across eight runs with progressively better reward/MDP conditioning it never
exceeded ~3% success. The failure mode was consistent: the agent learns to
**coast/hover** near the target rather than commit to the precise burn, because
the reward gradient toward "attempt the maneuver and maybe miss the tolerance"
is locally worse than doing nothing until the (sparse) success signal is
reachable. This reproduces — and explains — the original 2021 project's failure.
Bugs found and fixed along the way are listed in the appendix.

### Behavior cloning — ~16%
`scripts/bc_then_ppo.py`. Supervised regression of the policy's action onto the
expert's, with burn steps up-weighted. Reaches ~16%, then **PPO fine-tuning
destroys it** (16% → 3%) because the RL gradient pulls back toward the coast
attractor. Pure BC is capped by **covariate shift**: the policy is only trained
on states the expert visited, so its own small errors compound into unseen
states where it has no guidance.

### DAgger — ~45% (the ceiling)
`scripts/dagger.py`. Iterate: fit → roll out the policy → relabel the states it
actually visits with the expert's action → aggregate → refit. This directly
attacks covariate shift and success jumped 11% → 45% in three iterations — but
then **plateaued and oscillated around 45%** for a dozen more iterations even as
the fit kept improving (weighted MSE 0.0115 → 0.0039).

The ceiling is structural: the expert is **history-dependent** (a near-discontinuous
"latch and burn at apoapsis" rule), while the policy is a **Markovian feedforward
MLP**. The MLP must infer "burn now?" from the current observation alone and
smooths the sharp decision boundary; the residual ~6% action error plus ±1-decision
timing slop misses the tight 5% tolerance on ~55% of orbits. Δv also runs ~1.5×
optimal (sloppy corrective burns).

### Differentiable-simulation policy gradient — 81% (the winner)
`scripts/train_diffsim.py`. The two-body RK4 dynamics, orbital elements, and
observation are reimplemented in PyTorch (batched, mirroring `tbot` exactly so the
policy transfers to the Gymnasium env for eval). The policy is rolled out through
these differentiable dynamics and the episode loss (final orbit error + Δv +
impact penalty) is **backpropagated through the entire rollout** — an analytic
policy gradient. No exploration noise, no demonstrations: it optimizes the true
objective directly via the known physics.

It reached **91% at peak in ~100 gradient steps** (a smoke test, ~2 min) and, with
cosine-decayed LR + best-model checkpointing, settled at a stable **81% success /
1.26× optimal Δv** over 200 held-out orbits. That is a decisive win over every
model-free/imitation method — and it was reached in a fraction of their training.
The huge iteration-0 loss (~2×10⁹, from random policies flinging orbits) is
handled by gradient clipping without NaNs.

## Conclusion

Model-free RL and imitation wall out here because they **discard the fact that the
dynamics are known and exactly differentiable**. The differentiable-simulation
policy gradient exploits exactly that, and unlike imitation it optimizes the *true*
objective — so it is the method positioned to eventually *beat* analytic baselines,
not merely match them (see `docs/ROADMAP.md` for where that matters:
combined maneuvers, low-thrust, and multi-body, where Hohmann is not optimal).

**Milestone 1 verdict:** the physics-informed differentiable-sim approach is the
approach going forward. Next milestones (`docs/ROADMAP.md`): 3-D env with full
quaternion attitude control → re-validate circularize in 3-D → plane change →
the KSC combined-maneuver mission → low-thrust (Edelbaum) → multi-body
(Earth–Moon/Mars, JPL-ephemeris N-body).

## Appendix — bugs found and fixed on this milestone

1. Reward-shaping trap: fuel penalty so high that attempting the maneuver was
   net-negative before success was reachable → agent learned to coast.
2. Raw-inertial action space (required emitting a rotating vector) → switched to
   velocity-frame `[tangential, radial]`.
3. Terminal proximity bonus scaled by the tolerance → ~0 gradient until nearly
   at success; made it gradual in the raw element error.
4. No incentive to *finish* → agent hovered near target for the proximity bonus;
   added a per-step living cost.
5. Per-decision Δv could reach ~5 km/s (thrust × dt × repeat), flinging orbits
   hyperbolic and exploding the shaping → rescaled thrust, added
   escape/hyperbolic termination, clamped the potential.
6. Demo controller `dv_per_step` omitted the `× decision_repeat` factor (20×
   overshoot) → demos were only 35% successful; BC was cloning a broken expert.
7. BC under-fit on PPO's optimizer → dedicated Adam at 1e-3.
