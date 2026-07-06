# Roadmap — maneuvers, missions, and the optimality question

## Current status (2026-07-06)

- **2D circularize:** solved — diff-sim policy gradient ~76–80%, Δv ~1.24× optimal
  (Milestone 1, merged).
- **3D circularize (full quaternion attitude, decision-layer control):** solved at
  **92.3%** on a fresh 4096-episode set, **Δv 1.03× the impulsive analytic optimum**
  (matches it; does not yet beat it). Diff-sim refinement climbs *past* the imitation
  oracle it started from (79.9% → 92.3%). Getting there was optimizer forensics, not
  a bigger model: backprop-through-physics gradients are heavy-tailed (routine finite
  1e12–1e19 norms), and the fix was measure-the-distribution → trimmed mean + measured
  p90 per-episode clip + EMA banking + full-horizon (not truncated) BPTT. See
  [EXPERIMENTS_3D.md](EXPERIMENTS_3D.md).
- **JAX/XLA port:** the diff-sim hot path is ported to JAX (`scripts/jaxsim.py`) —
  numerically exact vs torch, **~50× faster** (0.27 vs 13.3 s/iter); R&D iterates in
  ~90 s. See [JAX_PORT.md](JAX_PORT.md).
- **Target-conditioning (commanded target radius ≠ apoapsis):** **solved.** A
  specialist trained only at rt = apoapsis collapses out-of-distribution (~18% off its
  radius); a target-conditioned scripted expert DAgger'd into the same network reaches
  **~99% across ±15% of apoapsis**. The blocker was the imitation source, not capacity
  or optimization — a slope-based diagnosis and three refuted in-place fixes are
  recorded in the R&D log. `scripts/dagger_target_jax.py`.
- **f&g coast propagator:** a differentiable universal-variable Kepler propagator is
  built and verified on a research branch (4.6e-7 vs RK4 at the decision interval);
  reserved for Tier-3 long two-body coast arcs (mixed coast/burn batches block a clean
  training-loop integration).
- **Combined circularize + plane-change (beat-the-naive-textbook):** **first learned
  discovery result.** An inclined ellipse → circular *and equatorial* at apoapsis in one
  combined burn. A diff-sim policy optimizing raw Δv reaches **84%** on a fresh
  4096-episode set at **median Δv 0.75× the naive** circularize-then-separate-plane-change
  — matching the analytic combined optimum (Δv/combined ≈ 1.05), i.e. the agent *discovers*
  folding the plane change into the burn. Baselines `combined_circularize_plane_dv` /
  `combined_plane_altitude_dv` in `tbot.orbital3d`; env/expert/refinement in
  `scripts/combined_sim.py`.
- **Edelbaum low-thrust baseline:** the analytic low-thrust combined altitude+plane Δv
  is in (`orbital3d.edelbaum_dv`), verified to the LEO→GEO literature anchors — the
  yardstick for the low-thrust tier.
- **J2 oblateness:** first environmental-fidelity term, in `jaxsim.deriv` (flag-gated,
  two-body bit-exact when off), verified by its −4.78°/day nodal regression.
- **Lambert solver + guidance/control study:** a differentiable universal-variable
  Lambert solver (`scripts/lambert.py`), used to measure the finite-thrust execution
  loss of an impulsive plan — small at high thrust-to-weight, growing as ~1/thrust, so
  a direct per-step diff-sim policy earns its keep in the **low-thrust** regime, not the
  high-thrust transfer. Motivates the low-thrust-vs-Edelbaum experiment next.
- **Low-thrust regime (Tier-2 entry):** `A_THRUST` is now a runtime knob and the
  sim has a differentiable eclipse thrust-gate (`scripts/jaxsim.py`, flag-gated,
  two-body bit-exact when off). **E1 (gravity losses):** curriculum-retrained
  diff-sim policies circularize across a 25× thrust drop (5e-3→2e-4) with no RK4
  horizon wall in that band — success declines gracefully (92%→77% on fresh-4096)
  and the finite-burn gravity loss shows up monotonically in Δv (`dvr` 1.20→1.38
  vs the impulsive lower bound). The control law is thrust-specific (a 2e-4
  specialist craters at chemical thrust) → argues for a thrust-conditioned
  generalist. **E2 (eclipse, null):** retraining with the shadow gate did *not*
  teach the policy to avoid the umbra (thrust-in-shadow 41.8%→41.4%, success
  67.8%→69.2%) — because circularize-at-own-apoapsis has **no burn-location
  freedom** (the efficient burn is pinned at apoapsis, whose sun/shadow phase the
  policy can't change). The eclipse gift needs a multi-rev *transfer* with choice
  over which sunlit arcs to burn. Probes: `scripts/lowthrust_eval.py`,
  `scripts/eclipse_probe.py`.
- **Beat-the-textbook (the deeper optimality question below):** partially answered. On
  the *impulsive* single-body case the agent *matches* the analytic optimum (tolerance
  box ~15% headroom; `verify_probe.py` float64/dt=1 s guards any sub-baseline claim), and
  it now *beats the naive decomposition* on the combined maneuver. The genuinely open
  frontier is low-thrust (vs Edelbaum) and multi-body (Tier-3).

## Guiding principle: optimize the true objective, don't imitate the textbook

The diff-sim policy gradient minimizes **actual Δv** (backprop through the real
dynamics), using analytic results only as **yardsticks**, not training targets.
That distinction matters: imitation (BC/DAgger) can only ever match its expert,
so it inherits the expert's optimality assumptions. Diff-sim optimizes the real
objective, so **it can discover solutions better than the analytic "optimum"
wherever one exists.** Keeping the reward on true Δv (never "match Hohmann") is
what makes the discovery question below answerable.

## Is Hohmann actually optimal? (the assumption worth questioning)

Hohmann is optimal only under a stack of narrow assumptions. It is a **two-impulse,
coplanar, circular-to-circular, impulsive, two-body** transfer — and even then
only below a radius ratio of ~11.94. Relax any of these and better solutions
exist:

- **Radius ratio > 11.94:** the three-impulse **bi-elliptic** transfer beats
  Hohmann (always beats it above ~15.58). Already implemented as a baseline
  (`orbital.bielliptic_dv`). So Hohmann isn't even the impulsive optimum for
  large ratios. (Vallado, *Fundamentals of Astrodynamics and Applications*.)
- **More than 2/3 impulses:** **Lawden's primer-vector theory** gives the
  necessary conditions for an impulsive trajectory to be optimal, and tells you
  when adding an impulse lowers Δv. The optimal impulse count/timing is not
  always 2.
- **Plane changes:** a pure plane change costs `Δv = 2·v·sin(Δi/2)`, so it's far
  cheaper where `v` is small (high apoapsis). The optimal way to change both
  altitude and plane is a **combined maneuver** (e.g. plane change at GTO
  apogee), *not* Hohmann-then-separate-plane-change. See the KSC mission below.
- **Finite / low thrust:** Hohmann is the impulsive limit and a **lower bound**
  for the two-body Δv; finite burns cost more (gravity losses). But low thrust
  changes the objective (continuous transfer, mass efficiency via high Isp), and
  **Edelbaum** gives the analytic optimal low-thrust combined altitude+plane Δv.
- **Multi-body (the big one):** in the CR3BP / N-body regime, Hohmann is **not**
  optimal. **Belbruno's ballistic-capture / weak-stability-boundary** transfers
  use solar perturbation to capture at the Moon with *less* Δv than a
  Hohmann-plus-capture-burn (flown for real by Hiten). **Gravity assists** and
  **invariant-manifold / low-energy transfers** (the "interplanetary transport
  network") beat patched-conic Hohmann for Earth–Moon and Earth–Mars. This is
  precisely where an optimizer over the full dynamics can find things a two-body
  analysis can't — the discovery question is genuinely open here.

**Takeaway:** Hohmann is the right yardstick only for the simplest single-body
coplanar case, where we expect the agent to *match* it (a correctness check). In
the combined-maneuver, low-thrust, and multi-body regimes there are known
better-than-Hohmann solutions — and possibly unknown ones — that the diff-sim
framework is a legitimate tool to search for. Each regime needs its **own**
correct baseline; using Hohmann everywhere would handicap the agent.

## Maneuver experiment list

### Tier 1 — single-body, impulsive (validation against known optima)
1. **Circularize from ellipse** — done (2-D). Port to 3-D as the first attitude test.
2. **Coplanar Hohmann transfer** (e.g. LEO→GEO altitude). Baseline: Hohmann.
3. **Bi-elliptic transfer** (large ratio). Baseline: bi-elliptic (should beat Hohmann).
4. **Pure plane change** at a node. Baseline: `2·v·sin(Δi/2)`. (Requires 3-D.)
5. **Combined plane-change + altitude**. Baseline: combined-maneuver optimum
   (should beat Hohmann + separate plane change).

### Tier 2 — finite / low thrust
6. **Finite-thrust circularize** — quantify gravity losses vs the impulsive lower
   bound. **Done (E1):** `dvr` rises 1.20→1.38 over a 25× thrust drop; no RK4
   horizon wall in the 5e-3→2e-4 band. Eclipse thrust-gate added and tested — no
   gift on circularize (E2 null; needs burn-location freedom). See Current status.
7. **Low-thrust multi-rev transfer** (altitude raise / plane change) — the vehicle
   E2 pointed to: many candidate burn arcs, so eclipse-avoidance is a real degree
   of freedom, and gravity/turn losses vs **Edelbaum** are the yardstick. This is
   where the beat-Edelbaum gift can actually appear. *(Next.)*
8. **Thrust-conditioned generalist** — `A_THRUST` in the observation, one policy
   across the thrust band. **Done (Build A):** ONE 14-input net recovers ~97% of the
   five E1 specialists' band-wide success (82.1% mean vs 84.3%), matching within
   ~1pt on 4/5 cells and beating at 2e-4. Reached by **distillation + DAgger, NOT
   diff-sim** — diff-sim fine-tuning left the thrust feature dead (a direct echo of
   the target-conditioning result). Residual ~11pt gap at the high-thrust
   near-impulsive cell (covariate shift, shrinking each DAgger round).
   `scripts/generalist_sim.py`.
9. **Edelbaum spiral / full SEP LEO→GEO** — needs orbit-averaged dynamics (RK4 at
   dt=10 s can't integrate a weeks-long spiral). Separate model; deferred.
10. **Efficient plane-change strategies** — plane change at apoapsis/nodes;
    apoapsis-raising to cheapen a plane change (a bi-elliptic-like idea for inclination).

### Tier 3 — multi-body (original goal; Hohmann not optimal)
11. **Earth–Moon transfer + capture.** Baselines: patched-conic Hohmann-like transfer
    *and* low-energy/ballistic capture (does the agent find the WSB route?).
12. **Earth–Mars transfer + capture** (heliocentric). Baseline: heliocentric Hohmann /
    porkchop optimum; explore gravity assists.
13. **Gravity assist / flyby** as a maneuver primitive.
14. **Low-energy / manifold transfers** — the discovery frontier.

**Dynamics decision (locked):** Tier-3 reuses the **JPL Horizons ephemeris +
N-body** stack from the original 2021 project — real Sun/planet/Moon states, not
an idealized CR3BP. We already rebuilt the data layer as the bulk/cached/idempotent
`ephemeris.py` (one Horizons query per body-span, disk-cached). Implication for
diff-sim: body positions are an **exogenous time series** (their motion doesn't
depend on the spacecraft), so in the differentiable rollout they are treated as
time-indexed constants — gradients flow through the spacecraft's own dynamics and
controls only, which is exactly what we want and keeps the graph clean. The N-body
gravity on the spacecraft (Sun + planets + Moon, from `BODY_DICT`) is summed each
step as in the original. If a cleaner idealized testbed is ever wanted for
manifold analysis, CR3BP can be added later as an *optional* alternate model, but
the primary Tier-3 dynamics are ephemeris-driven N-body.

## Mission composition — stringing maneuvers together

Worked example (yours): **Falcon 9 → 28.5° inclined parking orbit (KSC) → equatorial
(0°) circular GEO.** The efficient real-world sequence is *not* three separate
maneuvers; it's:
1. Perigee burn to raise apogee to GEO altitude (enter GTO).
2. **Combined** apogee burn: circularize *and* do the 28.5° plane change together,
   at apogee where orbital velocity is lowest (plane change is cheapest there).

That combined apogee burn is itself an optimization (how much plane change to do
at perigee vs apogee, whether to over-raise apogee bi-elliptic-style to cheapen
the plane change further). It's a perfect test case: train it and see whether the
agent rediscovers "do the plane change at apogee" — and whether it beats the naive
Hohmann + separate plane change.

Two ways to build mission-level behavior:
- **Monolithic:** one policy, start state → target state, horizon long enough to
  cover the whole mission. Diff-sim optimizes the entire sequence at once, so it
  *naturally* discovers combined maneuvers and timing. Best for well-defined
  single missions (the KSC case). Start here.
- **Hierarchical:** a library of trained sub-maneuver policies + a higher-level
  sequencer that picks which sub-maneuver and when. More modular; better for long
  multi-phase missions (Earth→Moon transfer→capture→circularize) where a single
  differentiable rollout is too long/stiff. Add when Tier-3 needs it.

## Baselines to implement (so each regime is judged correctly)
- [x] Hohmann, bi-elliptic, apoapsis-circularize (2-D, done).
- [x] Pure plane change `2·v·sin(Δi/2)`; combined plane-change+altitude optimum
  (`orbital3d.plane_change_dv`, `combined_circularize_plane_dv`, `combined_plane_altitude_dv`).
- [x] Edelbaum low-thrust Δv (`orbital3d.edelbaum_dv`).
- [ ] Patched-conic lunar/Mars transfer; note where low-energy transfers should win.

## Immediate next steps
1. ~~2-D diff-sim (Milestone 1)~~, ~~3-D circularize~~, ~~target-conditioning~~,
   ~~combined circularize + plane-change~~, ~~low-thrust circularize + eclipse
   plumbing (E1/E2)~~ — done (see Current status).
2. **Low-thrust multi-rev transfer (Tier-2 #7):** the vehicle E2 pointed to — an
   altitude-raise or plane-change spiral with many candidate burn arcs, so
   eclipse-avoidance is a real degree of freedom. Train vs `edelbaum_dv`; turn the
   eclipse gate back on and test whether the policy defers thrust to sunlit arcs
   (the concrete beat-Edelbaum gift a fixed continuous-yaw law can't take).
3. **Thrust-conditioned generalist (Tier-2 #8):** `A_THRUST` into the observation,
   one policy across the thrust band (E1 found specialists don't transfer).
4. Remaining environmental fidelity: real Sun ephemeris into `deriv` (moves the
   fixed `SUN_DIR` shadow to true geometry, and brings SRP); drag (aerobraking).
   J2 and the (fixed-Sun) eclipse gate already in.
5. The KSC two-burn combined transfer (LEO→GEO) as the first "sequenced" mission,
   using `combined_plane_altitude_dv` as the baseline.
6. Then Tier-3 (Earth–Moon / low-energy transfers, ephemeris N-body); Edelbaum
   full-spiral needs orbit-averaged dynamics (separate model).
