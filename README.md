# TrajectoryBot — differentiable-simulation RL for orbital maneuvers

A from-scratch orbital-dynamics stack (no Basilisk, no poliastro) and a
policy-gradient agent that learns spacecraft maneuvers by **backpropagating
through the physics**, benchmarked against analytic transfers — with every
claim re-verified at high fidelity before it's stated.

<p align="center">
  <img src="docs/media/maneuver.gif" width="420" alt="policy flying a circularization maneuver"/>
</p>

The agent is a small MLP (multi-layer perceptron) at the **decision layer
only**: every 200 s it emits a thrust direction (in the orbit frame) and a
throttle; a deterministic pointing controller slews the spacecraft and a
quaternion rigid-body integrator does the physics. The policy never touches
attitude directly — burn *planning* is learned, burn *execution* is classical
GNC (guidance, navigation, and control).

```mermaid
flowchart LR
    O[observations<br/>r, v, elements, fuel] --> P[policy MLP<br/>13 → 128 → 128 → 4]
    P -->|"direction (orbit frame) + throttle"| C[pointing controller<br/>rate command]
    C --> D[quaternion rigid-body dynamics<br/>RK4, dt = 10 s]
    D -->|"1200+ substeps, fully differentiable (JAX)"| L[loss: terminal orbit error + Δv<br/>+ Δv-to-go potential shaping]
    L -.->|"∂loss/∂θ through the whole rollout"| P
```

The whole policy is a tiny **13 → 128 → 128 → 4** tanh MLP — three weight
matrices, trained end-to-end through the physics:

<p align="center">
  <img src="docs/media/weights.png" width="820" alt="the three trained policy weight matrices"/>
</p>

## Verified results

<p align="center">
  <img src="docs/media/results.png" width="820" alt="verified scoreboard"/>
</p>

| Claim | Number | How it's verified |
|---|---|---|
| Success (5% tolerance terminal set, 3D) | **92.3%** | fresh 4096-episode set, never in-run telemetry |
| Fuel vs the impulsive analytic optimum | **1.03× median** (best-fuel policy, ~92% success) | float64 re-flight at dt = 1 s, clean episodes only, closed-form baseline (`scripts/verify_probe.py`) |
| Oracle it improves on | scripted analytic expert imitated at 79.9% | same fresh set |
| Sim gaming ruled out | 0 of 4096 episodes beat the admissible tolerance-box bound | same probe, both baselines |

Honest framing: the agent **matches** the analytic transfer's fuel cost while
running the full closed-loop attitude + finite-burn pipeline; it does not yet
beat it. The tolerance box admits solutions up to ~15% cheaper than exact
circularization — that quantified headroom is the open research target, and
the verification harness exists precisely so any future "beats the baseline"
claim survives scrutiny (integrator-energy audit, float64 re-flight,
clamp-region exclusion, per-geometry closed-form baselines).

Scope: the headline policy above circularizes at its own apoapsis (target
radius = initial apoapsis). Generalizing across *commanded* target radii —
a degeneracy where that specialist collapses out-of-distribution (~18%
success once the target leaves its trained radius) — is solved separately.
A target-conditioned scripted expert (a two-apse tangential controller that
drives both apses to the commanded radius) DAgger'd into the same
13→128→128→4 network reaches **~99% success across commanded radii spanning
±15% of apoapsis** on fresh 4096-episode sets. The cause was not network
capacity or optimization but the *imitation source* — the specialist lineage
was cloned from a fixed-target expert and had never imitated tracking. See
[`scripts/dagger_target_jax.py`](scripts/dagger_target_jax.py).

<p align="center">
  <img src="docs/media/trajectory.png" width="820" alt="one episode: path and orbital elements"/>
</p>

The final policy solves the distribution, not one lucky episode. Forty
random start orbits, each flown to termination, coloured by whether the
finish landed inside the 5% tolerance box:

<p align="center">
  <img src="docs/media/generalization.png" width="480" alt="40 random start orbits flown by the final policy, solved vs missed"/>
</p>

## Why differentiable simulation

Model-free RL was given every chance and walled out; exploiting the known,
exactly-differentiable dynamics wins by an order of magnitude (2-D milestone,
identical env):

| Method | Success | Δv vs optimal |
|---|---|---|
| Scripted analytic expert | 100% | 1.00 |
| Cold PPO (model-free, 8 runs) | 0–3% | — |
| Behavior cloning | ~16% | ~1.37 |
| DAgger | ~45% | ~1.5 |
| **Diff-sim policy gradient** | **81%** | 1.26 |

In 3-D, differentiable-sim refinement then climbs *past* the imitation oracle
it started from — each point is the fresh 4096-episode success of a
checkpoint along the campaign:

<p align="center">
  <img src="docs/media/progression.png" width="720" alt="fresh-set success climbing from the imitation oracle across the campaign"/>
</p>

The 3-D stack ports the hot path to **JAX/XLA** (`scripts/jaxsim.py`):
`lax.scan` + `jit(value_and_grad)` over the full 1200-substep rollout,
numerically exact against the torch reference and **~50× faster** on a
consumer GPU — which is what made the research loop below possible.

## The part that was actually hard

Backprop-through-physics gradients are **heavy-tailed**: one episode in a
few hundred carries a gradient norm of 1e12–1e19 (finite, not NaN) and a
single Adam step can erase 12 points of success. Getting from "imitates the
oracle at 80%" to "92% and stable" was optimizer forensics, run as
pre-registered hypothesis-refutation rounds (30+, logged verbatim):

<p align="center">
  <img src="docs/media/training-curve.png" width="820" alt="same seed, four aggregation strategies"/>
</p>

Same start checkpoint, same seed — only the gradient aggregation differs.

And here is what that refinement buys on a single episode — the *same* start
orbit flown by four training stages, imitation → refined, cycling so you can
watch the path tighten onto the target ring:

<p align="center">
  <img src="docs/media/learning.gif" width="380" alt="the same start orbit flown at four training stages"/>
</p>

The playbook that survived: **measure the per-episode norm distribution,
delete the monster tail (trimmed mean), clip survivors at the measured p90,
bank progress with an EMA (exponential moving average) policy, polish at low
lr**. The failure modes en route (truncation bias in short-horizon
backpropagation through time, absorbing-state mismatches,
potential-shaping traps that pay for zero progress, a normalize-epsilon that
seeded 1e6 gradients at every coast decision) are documented in
[`docs/EXPERIMENTS_3D.md`](docs/EXPERIMENTS_3D.md) (3-D campaign) and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (2-D milestone).

## Beyond circularize: the research frontier

The same backprop-through-physics approach extends to multi-body dynamics and to
maneuvers where a fidelity term the textbook ignores changes the optimal strategy.
Two recent threads (full methodology and verdicts in
[`docs/ROADMAP.md`](docs/ROADMAP.md)):

**Ballistic capture at the Moon (CR3BP).** A hand-rolled differentiable
circular-restricted three-body engine (rotating Earth–Moon frame, Jacobi constant
conserved to ~1e-7), the stable manifold of an L2 Lyapunov orbit computed from its
monodromy matrix (symplectic, det = 1), and a manifold-seeded transfer that arrives
*captured* at the Moon: Moon-relative energy < 0, staying bound for **4.6 lunar
revolutions** (~26 days, verified at dt = 5e-5) with **no capture burn**, where a
naive departure-burn search finds none.

<p align="center">
  <img src="docs/media/cr3bp_capture.png" width="860" alt="L2 stable manifold funnelling onto the Moon and a ballistic capture arc into a bound lunar orbit"/>
  <br/>
  <img src="docs/media/cr3bp_capture.gif" width="360" alt="the capture arc animated into its bound lunar revolutions"/>
</p>

**Rediscovering J2 nodal-drift phasing.** For a RAAN (orbital-plane) change, the
textbook impulsive plane change is J2-*blind* — but J2 oblateness precesses the node
for free, faster at lower altitude. A diff-sim policy minimizing true Δv over J2-on
dynamics, with no maneuver structure baked in, **rediscovers the operational phasing
technique**: dive to a lower altitude, let the node drift faster, return. It reaches
the target node for ~0.88× the Δv of naive passive waiting at 30°, falling to a
quarter–half at 60–90° (the bigger the node change, the more the dive pays) — and
lands within ~10–15% of an *analytic* optimization of that same dive-drift strategy
(which is slightly cheaper still). So this is autonomous rediscovery of a known
technique from the raw objective, **not a novel beat**: the same genre as the agent
rediscovering the raise-to-plane-change trick elsewhere in the project.

<p align="center">
  <img src="docs/media/j2_beat.png" width="960" alt="the policy dives to speed J2 nodal drift, reaches the target node passive falls short of, and matches the analytic dive optimum well under passive"/>
</p>

Both results come with their deflations attached: the capture is in the idealized
CR3BP (not real JPL ephemerides yet); the J2 result is a rediscovery that an
analytic dive beats, and its low-thrust Δv is idealized (impulsive-per-step). The
regimes where the savings vanish (small node angles, budgets so long that passive
is free) are recorded per build.

## Quickstart

```bash
uv sync --extra dev            # Python 3.12+, from pyproject/uv.lock
uv run pytest                  # physics, baselines, envs, solvability
uv run pyright                 # strict typing on the shipped library

# 2-D milestone (torch)
uv run python scripts/train_circularize2d.py

# 3-D JAX training run (GPU; CPU works, slower)
uv run --with "jax[cuda12]" python scripts/jaxsim.py \
  --init models/dagger_jax.npz --iters 3000 --chunk 60 --lr 1e-4 \
  --absorb --phi-dv --d-eps 1e-4 --trim-ep 16 --clip-ep 20000 --ema 0.995

# verify a fuel claim at high fidelity (float64, dt=1 s, both baselines)
uv run --with jax python scripts/verify_probe.py models/<ckpt>.npz

# regenerate the README figures
uv run --with jax --with matplotlib --with pillow python scripts/viz_readme.py models/<ckpt>.npz <logdir>
uv run --with jax --with matplotlib --with pillow python scripts/viz_readme_extra.py  # progression / weights / generalization / learning gif
uv run --with jax --with matplotlib --with pillow python scripts/viz_tier3.py          # CR3BP capture + J2-beat frontier (no checkpoint needed)
uv run --with jax --with matplotlib --with pillow python scripts/viz_lowthrust.py # low-thrust spiral (flies the E1 lt_2e-4 specialist)
```

Checkpoints and run logs are not committed (they regenerate by training); the
circularize-lineage checkpoints behind the figures are published as a
[GitHub release](../../releases/tag/readme-checkpoints). `docs/media/` holds the
small rendered figures only.

## Repository map

| Path | What it is |
|---|---|
| `tbot/` | The shipped library: quaternions, 2-D/3-D dynamics, orbital elements, attitude controller, Gymnasium envs. pyright-strict. |
| `scripts/jaxsim.py` | JAX/XLA 3-D diff-sim trainer (the research workhorse) with the full knob set. |
| `scripts/verify_probe.py` | float64/dt=1 s claim-verification harness (exact + tolerance-box baselines). |
| `scripts/eval_probe.py`, `norm_probe.py`, `step_probe.py`, … | The measurement toolkit the optimizer forensics ran on. |
| `docs/EXPERIMENTS_3D.md`, `docs/EXPERIMENTS.md` | The optimizer-forensics campaign (3-D) and the 2-D milestone methodology. |
| `docs/ROADMAP.md`, `docs/AUDIT.md` | Target maneuvers and the audit of the 2021 code. |
| `v2/`, `archive/` | The 2021 course project, kept as the "before" picture (excluded from CI/typing). |

## Roadmap

Circularize (3-D, done) → plane change → combined LEO→GEO transfer →
low-thrust (Edelbaum spiral) → multi-body with real JPL Horizons ephemerides
(Earth–Moon, Earth–Mars, capture) → benchmark against historically flown
trajectories. The N-body tier reuses the 2021 project's Horizons stack. For
any N-body fuel comparison, results must additionally be tested across
solar-system phases (syzygy effects are real physics but epoch-specific).

## Provenance

Started in 2021 as a graduate astrodynamics course project that trained but
never converged — the original report and code are preserved in
[`docs/final-report.md`](docs/final-report.md), `v2/`, and `archive/`, and
the revival began by auditing why it failed
([`docs/AUDIT.md`](docs/AUDIT.md)). Everything physics is hand-rolled on
purpose: the point is to own the whole stack from the RK4 up.

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md) for how changes are
evaluated (pre-registered experiments, verification-fidelity claims).
