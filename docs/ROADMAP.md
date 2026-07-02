# Roadmap — maneuvers, missions, and the optimality question

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
6. **Finite-thrust circularize/transfer** — quantify gravity losses vs the
   impulsive lower bound.
7. **Edelbaum spiral** (continuous combined altitude+plane). Baseline: Edelbaum Δv.
8. **Efficient plane-change strategies** — plane change at apoapsis/nodes;
   apoapsis-raising to cheapen a plane change (a bi-elliptic-like idea for inclination).

### Tier 3 — multi-body (original goal; Hohmann not optimal)
9. **Earth–Moon transfer + capture.** Baselines: patched-conic Hohmann-like transfer
   *and* low-energy/ballistic capture (does the agent find the WSB route?).
10. **Earth–Mars transfer + capture** (heliocentric). Baseline: heliocentric Hohmann /
    porkchop optimum; explore gravity assists.
11. **Gravity assist / flyby** as a maneuver primitive.
12. **Low-energy / manifold transfers** — the discovery frontier.

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
- [ ] Pure plane change `2·v·sin(Δi/2)`; combined plane-change+altitude optimum.
- [ ] Edelbaum low-thrust Δv.
- [ ] Patched-conic lunar/Mars transfer; note where low-energy transfers should win.

## Immediate next steps
1. Finish + record the 2-D diff-sim result (Milestone 1 close-out).
2. 3-D env with full quaternion attitude control; re-validate circularize in 3-D.
3. Plane change (3-D) with the correct `2·v·sin(Δi/2)` baseline.
4. The KSC combined-maneuver mission as the first "sequenced" test.
5. Then Tier-2 (low thrust / Edelbaum) and Tier-3 (Earth–Moon).
