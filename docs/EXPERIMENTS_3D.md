# The 3-D campaign: optimizer forensics for backprop-through-physics

How the 3-D circularization agent went from "imitates the oracle at ~80%" to
"92% success at analytic-matching fuel, verified" — and what broke along the
way. Every round below was pre-registered (hypothesis, prediction, refute-by
criterion) before launch; verdicts were logged verbatim, including the wrong
ones. Rounds are numbered R… as in the internal log; ~30 ran in total.

## Setup

- Task: from a random ellipse (perigee 400–800 km alt, apo/peri 1.3–2.5,
  inclination ≤ 40°), reach a circular orbit at the initial apoapsis radius
  within 5% on semi-major axis and eccentricity, under a 2.0 km/s Δv budget.
- Policy: 13→128→128→4 MLP at the decision layer (orbit-frame thrust
  direction + throttle every 200 s); deterministic pointing controller and
  quaternion rigid-body RK4 underneath (`scripts/jaxsim.py`, JAX/XLA,
  ~50× the torch reference).
- Gradient: `jit(value_and_grad)` through the full 1200-substep rollout.
- Bootstrap: behavior-cloned scripted analytic expert (79.9% on the fresh
  evaluation set), then true-objective fine-tuning — the hybrid
  oracle-then-remove-the-oracle strategy.

## The five walls, in the order they fell

**1. Objective geometry.** Escaped orbits were gradient-dead (energy clamps),
success was terminal in the env but not absorbing in the loss (78% of
"latched" episodes ended out of tolerance), and the original orbit-error
potential paid full shaping reward for "progress" that left the true cost--
to-go unchanged — a closed-form trap state where policies burned the whole
budget standing still. Fixes: a smooth escape term, `--absorb` latches, and
a physics-informed potential Φ = −Δv-to-go (smooth two-impulse estimate).
Re-weighting knobs were tried first and refuted: geometry, not weights.

**2. Heavy-tailed gradients.** Per-episode gradient norms follow a fat power
law: median ~50–700 (loss-dependent), p99 up to 1e8+, with routine *finite*
1e12–1e19 monsters. One episode owns the batch; a global clip turns the
update into a unit step in the monster's direction. Source isolated by
two-arm experiment: the thrust-direction normalization `d/‖d‖` at ε=1e-12
seeds ~1e6 gradients per component at every coast decision (fixed with
ε=1e-4); crash-zone backprop was the suspected source and was refuted.

**3. Truncation bias.** Truncated BPTT (K=10 of 60 decisions) was stable but
*biased*: loss rose while following its own gradient, and no per-episode clip
threshold (1e0…1e5) made it an ascent direction near the expert. Full-horizon
K=60 with per-episode sanitization climbed immediately. The "warm-start
crater" mystery was truncation bias all along.

**4. Aggregation calibration.** The per-episode clip threshold was calibrated
on the K=10 norm distribution and never re-measured at K=60, where the median
is ~10× higher — so "clip the monsters" was actually full per-episode
normalization: equal-vote averaging with a persistent 3–6% monster fraction.
Two runs from the same checkpoint with the same seed: trim-only oscillates
and recovers; miscalibrated clip dies monotonically into the burn-in-place
trap *while its own loss rises*. Measured fix: delete the top ~6% by norm
(trimmed mean), clip survivors at the measured p90 (~2e4). The collapse
failure mode disappeared entirely.

**5. Step-noise jitter.** One Adam step at lr 5e-5 moves eval success by
±3pp typically and −12pp in the tail (measured over repeated single steps),
so in-run "best" numbers are winner's-curse samples. Fixes: an EMA policy
(τ=0.995) for evaluation/checkpointing — passive w.r.t. training, so its
`raw=` companion field is a same-batch control — and a low-lr polish phase
(at lr 5e-6 the post-peak erosion vanishes, identifying it as lr-scaled
noise-wandering rather than a systematic pull).

## Verified endpoints

Numbers quoted only from the fresh 4096-episode set and the float64/dt=1 s
re-flight harness (`scripts/verify_probe.py`, exact-circularization and
tolerance-box baselines, clean episodes only):

| Checkpoint | Success | Δv / impulsive optimum (median, f64) |
|---|---|---|
| Oracle imitation (control) | 79.9% | 0.989 |
| Best-success policy | 92.3% | ~1.17 |
| Best-fuel policy | 91.9% | **1.032** |

Zero of 4096 episodes beat the tolerance-box bound (0.849× exact) — the sim
is not being gamed. An RK4 energy audit bounds integrator error at
0.005–0.02 m/s Δv-equivalent per episode (float32 roundoff-dominated), four
orders below claim scale.

## Open lanes

- **Fuel headroom:** the tolerance box admits solutions up to ~15% cheaper
  than exact circularization; current best-fuel policy is 1.03× exact.
  Raising the fuel price 4× changed nothing (refuted — the drift toward
  fuel-sloppiness is not price-sensitive), so this likely needs credit
  assignment (a terminal critic) rather than reward pricing.
- **Target generalization:** training always had target ≡ initial apoapsis;
  policies cannot fly any other commanded target (aim-scaling collapsed
  out-of-distribution, twice-confirmed). Jittered-target training interferes
  destructively at full and narrow jitter so far; whether EMA-averaged
  checkpoints are fragile as training inits is under test.
- **From-scratch discovery:** blocked twice (flat coast basin; the
  inefficient-burn valley defeats even well-edge reverse-curriculum starts).
  Parked pending a critic or zeroth-order gradient mixing.

## Method notes

Pre-registration kept us honest: of the ~30 rounds, more than half refuted
their hypotheses, and several refutations (the same-seed aggregation
contrast, the one-step sensitivity probe, the eval-set winner's curse) were
worth more than the confirmations. Measurement before knobs — the two
biggest fixes (d-eps, trim/clip thresholds) came directly from measuring a
distribution someone had previously guessed at.
