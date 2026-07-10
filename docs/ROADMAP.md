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
  it now *beats the naive decomposition* on the combined maneuver. The open
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
  analysis can't — the discovery question is still open here.

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
7. **Low-thrust multi-rev transfer** (altitude raise). **Done (Build B):** diff-sim
   learns the circular r1→r2 spiral at **67.6% success, Δv/edelbaum 1.08** (near the
   coplanar low-thrust bound). The eclipse-gift test **refuted H-B2**: the
   eclipse-aware policy does NOT avoid the umbra (shadow 37% ≈ blind 38%) —
   **cancelled-in-shadow thrust costs no fuel, so there is no Δv incentive to dodge
   shadow**, only a thrust-time cost. What the transfer's burn-location freedom DOES
   buy is *robustness*: it recovers success under eclipse (46→74%) where the pinned
   circularize couldn't (E2: 68→69%). A real shadow-avoidance skill needs a binding
   cost (deadline / mass penalty). `scripts/transfer_sim.py`. **Build D follow-up:**
   even giving the policy shadow *observability* (sun in the obs) + an artificial
   shadow-command penalty only yields MODEST avoidance (shadow 38→34% at a gentle
   weight; a strong weight backfires into global throttle-down). So the
   eclipse-beats-Edelbaum idea is thoroughly refuted — avoidance is weak even when
   incentivized, because decision-level (200s) control is coarse vs shadow arcs and
   gating a free-cost action gives no gradient pull. `scripts/eclipse_cost_sim.py`.
8. **Thrust-conditioned generalist** — `A_THRUST` in the observation, one policy
   across the thrust band. **Done (Build A):** ONE 14-input net recovers ~97% of the
   five E1 specialists' band-wide success (82.1% mean vs 84.3%), matching within
   ~1pt on 4/5 cells and beating at 2e-4. Reached by **distillation + DAgger, NOT
   diff-sim** — diff-sim fine-tuning left the thrust feature dead (a direct echo of
   the target-conditioning result). Residual ~11pt gap at the high-thrust
   near-impulsive cell (covariate shift, shrinking each DAgger round).
   `scripts/generalist_sim.py`.
9. **Edelbaum spiral / full SEP LEO→GEO** — **Done (Build C):** orbit-averaged
   dynamics built (state = V, i; yaw+throttle control; 2-D, no RK4-per-second wall).
   A diff-sim yaw law **recovers the Edelbaum closed form**: 100% reach target,
   median Δv/edelbaum 1.006, LEO→GEO@28.5° = 6.02 vs 5.95 km/s (ratio 1.01). A
   correctness match (Edelbaum is the provable optimum — can only match, not beat),
   which validates the averaged engine for regimes with no closed form (variable
   thrust, eclipse-in-averaged, J2-coupled Ω̇). `scripts/edelbaum_sim.py`.
10. **Efficient plane-change strategies** — **Done (Build E):** in the averaged
    model, a diff-sim yaw law **rediscovers raise-to-plane-change** for a pure plane
    change — it raises altitude ~10% (median), ~20% for large Δi, turns the plane
    cheaply where V is small, and returns, matching the Edelbaum optimum (0.997) and
    beating the naive constant-altitude law by ~4% (median) to ~7% (Δi>35°). The
    continuous analog of the bi-elliptic-inclination trick, from raw Δv minimization.
    Needed FULL yaw β∈[-π,π] (edelbaum_sim's ±π/2 forbids the return leg — a control
    parameterization hiding a whole strategy class). `scripts/planechange_sim.py`.

### Tier 3 — multi-body (original goal; Hohmann not optimal)
11. **Earth–Moon transfer + capture.** Baselines: patched-conic Hohmann-like transfer
    *and* low-energy/ballistic capture (does the agent find the WSB route?).
    **Started (Build F):** a **verified differentiable CR3BP engine** is built as the
    clean first testbed (`scripts/cr3bp_sim.py` — Lagrange points exact, Jacobi
    conserved to 2.4e-7). A first differentiable departure-burn transfer
    (`scripts/cr3bp_transfer.py`) reaches the Moon vicinity at ~Hohmann cost (2.29 vs
    2.31 km/s). The genuine low-energy CAPTURE beat is NOT yet demonstrated (that
    flyby isn't a verified bounded capture) — it remains the open frontier; next is
    capture verification + a manifold-seeded / multi-shooting search to tame the
    chaotic long-arc gradients. **Build G:** a **verified Moon-relative capture
    criterion** (E_moon<0 + bounded-Hill-sphere propagation) + an honest two-mode
    search (direct pay-the-capture-burn vs ballistic-drive-E<0). **Honest null:** the
    search finds direct arrivals (best 2.585 km/s) but **no ballistic capture** — the
    thin WSB set eludes a gradient+grid search over near-TLI departures (they all
    arrive fast/hyperbolic). The genuine low-energy beat needs L2-manifold seeding /
    multi-shooting — the clear next step, now on a validated engine + criterion.
    `scripts/cr3bp_capture.py`. **Build H — manifold seeding closes G's gap:**
    `scripts/cr3bp_manifold.py` builds the dynamical-systems structure a departure
    grid lacked — closed L1/L2 **Lyapunov orbits** (STM differential correction,
    Jacobi to machine precision), a **symplectic monodromy** (det=1) with the correct
    hyperbolic reciprocal pair via variational equations, and their **stable
    manifolds**. Seeding the search on the L2 stable manifold produces a **VERIFIED
    temporary ballistic capture** (E_moon<0, closest ~289 km, ~2 lunar revs / ~2 weeks
    bound) — the exact object G's grid found **zero** of. So manifold theory reaches
    the thin capture set. H first reported a "null" on the transfer beat, claiming the
    manifold "doesn't reach LEO → needs the Sun" — **Build I corrects that overclaim.**
    **Build I — the modest beat is REAL in the pure CR3BP** (`scripts/cr3bp_lowenergy.py`):
    you don't need the manifold to reach LEO — you PATCH onto it near the Moon. A
    two-impulse LEO→manifold→ballistic-capture transfer, compared to a **steel-manned**
    Hohmann+minimal-capture to the *same* verified captured state, **beats it by ~5%
    (0.18 km/s), dt-robust** — but only for a *temporary* (2.4-rev / ~17-day) capture
    into a *loose high* lunar orbit (periapsis ~12,400 km), saving on the **capture
    side only** (TLI unchanged; no Sun to lower perigee). En route a G-style
    self-deception was caught + removed (an unguarded "beats 0.74×" that was
    sub-lunar-surface collision artifacts + a flyby-vs-real-capture asymmetry; the fix
    was excluding sub-surface plunges and requiring the manifold's "free capture" to
    clear the *same* K≥2-rev bar Hohmann's paid capture does). **Net:** the low-energy
    capture is demonstrated AND yields a small honest Δv beat; the Sun is needed for the
    **big** wins (permanent capture, capture into high orbits where Hohmann's capture
    burn is large, and a lower TLI), next via a 4-body (Sun-perturbed bicircular) or the
    locked ephemeris N-body model below. **Build J — the LEARNED diff-sim policy reaches
    Tier-3 capture** (`scripts/cr3bp_policy.py`): back on the primary method (backprop a
    2-burn control through the CR3BP rollout), it achieves a **dt-robust VERIFIED
    ballistic capture** (E_moon=−0.11, ~9–10 Moon revs / ~35 d bound, ~2.84 km/s) where
    G's 1-burn search **stalled**. The enablers: the manifold/I-derived
    **2-burn structure** (a 1-burn policy provably cannot capture — G's wall) plus a
    **physically-correct approach-speed objective** (naive distance/energy objectives get
    reward-hacked into deep-fast plunges — two such hacks caught and fixed). Manifold-
    *seeding the init* barely mattered (raw ≈ seeded, 2.843 vs 2.828), so the manifold's
    value was its **structural insight**, not warm-starting. Scope: this is a *direct*
    capture (≈G's direct route, more robustly verified), not the low-energy beat. So the
    learning agent now reaches verified multi-body capture — the Tier-3 milestone for the
    method; a learned *low-energy* transfer and the ephemeris port remain open.
    **Build K — first genuine beat of a physics-AWARE strategy (J2 node change), but
    modest** (`scripts/j2_policy.py`, `scripts/j2_node.py`): R-K1 verified J2 secular nodal
    regression numerically vs Vallado (<0.5% err) and the impulsive plane-change baseline.
    R-K2 backprops a smooth low-thrust RTN-frame control (Fourier, 48 DOF, no structure
    baked in) through a J2-on rollout, minimizing TRUE total Δv (low-thrust ∫|a|dt +
    priced impulsive cleanup to the exact target). **Fair-baseline check:** the J2-*blind*
    impulsive plane change (2.89 km/s for -30°) is a strawman — the fair baseline is
    *passive-J2* (coast the budget, let the node drift for free, clean the residual). The
    genuine diff-sim win is ACTIVE drift-shaping (dive to speed the drift) beating PASSIVE
    waiting, and it is only **5–12%**, only in a **~7–9 day budget window** (below ~6 d a
    dive can't amortize its round trip; near ~10 d passive reaches target for free → nothing
    to beat). The ~2–3× "win" over the J2-blind baseline is almost all passive J2-awareness,
    not learning. Caveats: zero-init optimization is basin-sensitive (T=6 d traps in a
    wasteful local min *below* passive — needs multi-start or a min(active,passive)
    fallback); low-thrust Δv is idealized (per-step impulses). Net: the method *can* beat a
    J2-aware optimum, but the margin is small and regime-specific — not the clean large beat
    R-K1 prematurely implied.
    **Build L — the 5-12% was a FLOOR: the J2 beat GROWS to 55-69% at large node changes**
    (`scripts/j2_policy.py` extended; `.rnd/campaign-2026-07-07-j2-eccentric-sweep.md`).
    H-L1 SUPPORTED: at fixed ~80%-passive-coverage budgets the active-beats-passive ratio
    is 0.876 / 0.449 / 0.313 for ΔΩ = 30 / 60 / 90° — the active Δv stays ~flat (~0.55, just
    dive a little and let J2 do more over a longer budget) while passive's residual plane
    change grows super-linearly. So for operationally-relevant large node changes diff-sim
    active drift-shaping beats passive-J2 by more than half. H-L2 REFUTED: eccentricity does
    NOT help (ratio 1.00 vs circular 0.88) — a valid eccentric orbit (periapsis above the
    atmosphere) forces a large a0 that slows the drift, cheapens the residual plane change,
    and makes the active altitude lever expensive; you can't have a tight fast-drifting orbit
    AND high e. H-L3 / correction: Build K's "T=6 d wasteful local min below passive" was an
    **adam best-tracking off-by-one bug** (best_x saved one step after best_loss), not
    physics — fixed, the optimizer never underperforms passive (T=6 d → 0.999 across all
    starts). Honesty guards added: eccentric-drift check vs Vallado (1-e²)⁻² (caught an
    a0-fixed setup that plunged periapsis 863 km sub-surface), bi-elliptic-aware baseline
    (steel-mans plane change above ~49°), best-stopping-time passive baseline.
    **Build M — CORRECTION: the J2 result is REDISCOVERY, not a beat**
    (`scripts/j2_policy.py` `--analytic`; `.rnd/campaign-2026-07-07-j2-dive-baseline.md`).
    A blind-spot pass caught that the K/L steel-man stopped one level short: a J2-aware
    planner with a time budget doesn't just coast (passive) — the known operational move is
    to DIVE (faster nodal drift ∝ a^−7/2), coast low, RETURN, exactly what the policy
    "discovered". H-M: an analytic optimization of that dive-drift strategy (min over dive
    altitude of Hohmann round-trip + residual plane change) beats the diff-sim policy by
    **9-15%** (0.495/0.494/0.494 vs policy 0.540/0.548/0.571 at ΔΩ=30/60/90°), and gets the
    SAME 55-69% margin over passive. So Build K/L's "first genuine beat of a physics-aware
    optimum" was WRONG: diff-sim does NOT beat the J2-aware optimum — it REDISCOVERS the
    dive-drift maneuver from the raw Δv objective (no structure baked in) and matches it to
    ~10-15% (Build-E genre, like rediscovering raise-to-plane-change). The 55-69%-under-passive
    is a property of the dive-drift STRATEGY, not of learning. Policy Δv is dt-converged
    (0.5404 at dt=60 and dt=30). Public README + j2_beat.png corrected to the rediscovery
    framing; cr3bp_capture.png regenerated at trusted dt=5e-5 (verified ≥2-rev capture, was a
    1.5-rev arc at the distrusted dt=1e-4); dead `.rnd/` README reference removed.
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

**Build N — the ephemeris N-body engine is built + verified (R-N1)** (`scripts/nbody_sim.py`;
`.rnd/campaign-2026-07-07-nbody-ephemeris.md`). Starts paying the ephemeris-port debt after
Builds F–M ran on sanctioned testbeds (CR3BP, two-body+J2). JAX differentiable rollout of a
spacecraft under K point-mass bodies at **exogenous** positions (pre-sampled from the verified
`ephemeris.py`, held per RK4 step); physical km/s units, IAU/DE440 GM. Verified OFFLINE
(CI-safe, no network): two-body Kepler limit closes to machine precision (|Δr|/a=9e-13),
energy conserved to 7e-15, differentiable (finite ∂miss/∂Δv). Real-ephemeris bridge (`--fetch`)
propagates a 1 AU heliocentric orbit under real Sun+Earth+Moon, bounded 1.0000 AU over 100 d.
HONEST LIMITATION (real method finding): bodies-held-per-step is valid at heliocentric scale
but BREAKS for close geocentric orbits (Earth moves ~9000 km/step ≫ LEO radius) — near-Earth
phases need substep body interpolation / a patched geocentric frame, added in R-N2. Next:
R-N2 differentiate a departure Δv to a target vs the Lambert optimum (fair in-space Δv metric);
R-N3 epoch sweep on any gain (syzygy gains are epoch-specific).

**Build N R-N2 — departure Δv on real ephemerides vs the two-body Lambert optimum**
(`scripts/nbody_transfer.py`; substep interp added to `scripts/nbody_sim.py`). Added
`rollout_interp` (bodies at the RK4 substages t, t+dt/2, t+dt), fixing R-N1's held-per-step
limitation — verified against the same machine-precision Kepler closure. Fair Δv-vs-Δv test on
matched endpoints (r1 = departure-body @ launch, r2 = target-body @ arrival, TOF fixed):
Lambert (`scripts/lambert.py`, Sun-only) gives the two-body baseline; a differentiable
departure-Δv optimizer solves the same targeting under the real perturbed field. **H-N2a
SUPPORTED (offline gate):** Sun-only, the optimizer recovers the Lambert Δv from a cold Δv=0
start to 1.24e-11 relative (miss → 0). **H-N2b SUPPORTED (real, --fetch):** across 5 direct
Earth→Mars cruises (MRO/Odyssey/TGO/Perseverance/MSL), the two-body Lambert plan misses Mars by
30–85k km under Sun+Jupiter+Saturn+Venus; re-optimizing closes it for a **±0.5–0.8 m/s (±0.02%)**
correction — the third-body perturbation, small & epoch-consistent (a CORRECTION, not a beat:
Lambert is the endpoint-matched two-body optimum). **Honesty guard fired:** MAVEN (long-way 230°
sweep) exposed a non-converged single-rev Lambert baseline (self-miss 3.16e7 km), first mis-read
as a −508 m/s "correction"; added a Sun-only self-consistency gate that flags an invalid baseline
instead of reporting an artifact. The differentiable optimizer, being branch-agnostic, still
solves MAVEN's long-way targeting where fixed-iteration Lambert fails.

**Build N R-N3 — attribute the R-N2 correction, test its assumptions** (`scripts/nbody_transfer.py
--decompose`). The original R-N3 ("epoch sweep on a gain") was moot — R-N2 found a correction, not a
gain — so this round instead stress-tests the three assumptions R-N2's number rests on, by flying the
two-body Lambert plan under single-perturber subsets. **H-N3a SUPPORTED:** Jupiter alone reproduces
90–105% of the ~30–85k km cruise miss (Saturn/Venus each <8k km) — the correction is a Jupiter
third-body perturbation, matching the order estimate. **H-N3b REFUTED-as-phrased but vindicates the
exclusion:** naively adding Earth as a heliocentric perturber blows the miss to ~1e8 km; a gate-radius
probe shows it shrinks monotonically (4.4e6→5.5e4 km from 1×→30× Earth SOI) because the transfer
lingers within a few SOI of Earth at departure — that near-Earth gravity is the patched-conic C3 phase
(vehicle-set, separate), so excluding Earth/Mars from the cruise perturber set is REQUIRED, now
demonstrated not asserted (the target body Mars is fine; the departure body is the outlier). **H-N3c
REFUTED-as-phrased:** d_Jup barely varies (4.99–5.44 AU, near-circular orbit) so the miss spread (2.7×)
is a transfer-geometry/TOF effect, not 1/d² distance — consistent with R-N2's epoch-stable ±0.5–0.8 m/s
across 2001–2020. Next: R-N4 low-thrust / rendezvous on real ephemerides, once the perturbation model
is trusted.

**Build N R-N4 — differentiable low-thrust Earth→Mars on real ephemerides** (`scripts/nbody_lowthrust.py`;
adds a state-dependent RTN rollout `rollout_rtn`). Brings the diff-sim method's core strength (a
continuous bounded-thrust control found by backprop) to the locked Tier-3 dynamics. **H-N4a SUPPORTED
(capability):** the optimized RTN low-thrust control REACHES Mars — real Sun+Jupiter (MRO window) lands
156–193 km from Mars at 5–10× the mean-accel-to-match, and reaches at 3× too; offline Sun-only reaches
at 5×/10×. **H-N4b NOT cleanly testable by single-shooting — an honest method-limit finding:** the
250-day terminal-miss landscape is too stiff for plain Adam (reach fails below ~5× offline — an optimizer
floor, not physical, since the Δv budget is ample; misses non-monotonic), a band-limited Fourier control
can't spike to impulsive (so dvr won't → 1), and the returned Δv is an un-economized UPPER BOUND
(12–21 km/s vs the 4–5.5 km/s impulsive floor) — reporting it as a physical low-thrust cost would be
dishonest. Corrects the pre-registered dvr-vs-thrust prediction: the diff-sim method that excelled on
bounded-horizon problems (circularize / plane-change / capture) hits its limit on long-arc low-thrust
interplanetary economics. Fix = a better-conditioned formulation (collocation / multiple-shooting, or the
orbit-averaged elements Build C already validated for Edelbaum). R-N5 candidate: port `edelbaum_sim`'s
averaged low-thrust engine onto real ephemeris gravity, where the horizon is tractable and Δv trustworthy.

**Build N R-N5 — multiple-shooting (Sims-Flanagan) FIXES R-N4's method limit** (`scripts/nbody_collocation.py`;
user-chosen direction). Node states x_k=(r,v) as decision variables, one bounded impulse per segment,
N-body coast, continuity defects, minimize ΣΔv under defect+boundary penalties (Adam), warm-started on the
Lambert arc. **The crux was NON-DIMENSIONALIZATION:** Adam's per-parameter step is ~lr regardless of scale,
so raw-km node positions (O(1e8)) never moved enough to close defects (gave dvr<1 with multi-Mkm open
defects — meaningless); rescaling nodes to canonical units (LU=AU, VU≈29.78 km/s) so they are O(1) fixed it.
**H-N5a SUPPORTED:** defects close to ≪ Mars SOI (24–98k km) at 10×/5×/3× mean-accel — where R-N4
single-shooting gave garbage. **H-N5b SUPPORTED (the payoff):** real MRO ephemerides give **dvr = 1.026 at
10× (within 3% of the impulsive Lambert floor)**, rising smoothly & monotonically (1.026→1.151→1.729) as
a_max drops — a TRUSTWORTHY finite-thrust penalty, vs R-N4's meaningless backwards 19.5→12.0 on the same
problem. Honest residual: the very-low-thrust corner (≤2×) doesn't fully converge with plain Adam (departure
Δv must spread over 10+ segments) — wants a true NLP solver (SLSQP/IPOPT with hard defect constraints). General
lesson recorded: scale the state to O(1) when gradient-optimizing multi-body trajectories.

**Build N R-N6 — gravity-assist frontier, round 1: forward-model fidelity** (`scripts/nbody_flyby.py`; user
direction "do the gravity-assist frontier"). North star (user): agent DISCOVERS a gravity assist — Voyager-style,
possibly multiple — by numerical differentiation of the environment, then GENERALIZES to future missions
(amortized mission planning). Per memory 418e2e2, assists are near-impossible to beat on Δv, so the honest prize
is *discovery*, not "beat it." R-N6 first tests the load-bearing assumption underneath all of that: **can the
R-N5 engine (fixed-step RK4 + Plummer softening) even reproduce a flyby?** Rig: a two-body flyby vs the EXACT
hyperbola, judged by the conserved invariants (ε→V∞, eccentricity vector→turn δ). **H-N6a SUPPORTED:** at
dt≤500 s / soft=10 km the turn matches to 0.000%, V∞ to 0.0000% — machine precision. **H-N6c SUPPORTED (sharp
step-size ceiling):** error explodes for dt ≳ 0.3·t_peri (t_peri=r_p/v_peri); R-N5's fixed ≈14 h step carries a
WIDE gas-giant graze (0.004%) but gives 27–55% garbage on a TIGHT terrestrial-SOI-scale pass. **H-N6b SUPPORTED
(after fixing a metric bug):** softening suppresses the turn once soft ≳ 0.15·r_p (1.9% at 0.15, 15% at 0.5,
33% at 1.0) — but my first metric (δ from the eccentricity vector) was BLIND to it, because softened gravity is
still central/conservative and conserves ε and h, so the osculating eccentricity is invariant by construction;
the raw velocity turn is the correct probe. **The tension that scopes R-N7:** softening (R-N5 added it for smooth
flyby gradients) must be ≲0.1·r_p or the assist is physically fake, and no single fixed step serves both cruise
and periapsis — so discovery needs small softening + adaptive/local stepping near close approaches, OR a
flyby-NODE transcription (patched V∞-in=V∞-out, bounded turn as a boundary condition) on the R-N5 multiple-shooting
engine, which is how real mission design does it. R-N6 proved this before a discovery round could return a
silently suppressed, coarse-stepped fake assist. Assists are also epoch-specific (belief 8f2ff0a) — generalization
must be over launch windows.

**Build N R-N7 — can differentiation EXPLOIT a known assist?** (`scripts/nbody_flyby_exploit.py`). Offline
Sun + MOVING Jupiter (an assist needs a moving body — a fixed one only bends the path with no heliocentric
energy change). Single-shooting: departure velocity differentiated through a full fine-step (40k-step) rollout
that resolves the flyby (R-N6-clean). Target = a real flyby seed's endpoint (6 AU); fair direct Lambert to the
same (r2, 5.5-yr TOF) is the baseline. **Two honest verdicts.** (1) **The assist is real & large — SUPPORTED:**
a fully-integrated, R-N6-clean trajectory reaches the target for **9.20 km/s vs the direct ≥25.3** (~63% saving
from Jupiter's orbital energy), closest approach 6.5 R_J. (2) **Raw-environment differentiation does NOT robustly
exploit it — the pre-registered NULL fired:** a **2 m/s** departure perturbation (0.02% of the excess) shifts the
Jupiter approach by ~one r_p, the post-flyby arc diverges chaotically, terminal miss explodes to **1.26 billion
km**, and single-shooting gradient descent cannot recover the razor-thin basin (1500 iters). The thin basin IS the
invariant-manifold tube — reachable by phase-space structure (Tisserand/manifold topology), not by gradient
descent through the raw field. Same long-arc single-shooting stiffness R-N5 fixed for the direct arc, amplified by
the flyby. Scopes R-N8: a flyby-NODE transcription (bounded turn as a boundary condition) and/or multiple-shooting
with a node bracketing the close approach, to localize the perturbation and remove the razor-sensitivity.

**Build N R-N8 — flyby-NODE transcription FIXES R-N7's razor-thin basin** (`scripts/nbody_flyby_node.py`).
The fix R-N7 scoped, built and verified. Two smooth Sun-only Kepler legs patched at a node where the
heliocentric velocity rotates by δ (|V∞| conserved, δ bounded by min-periapsis) — the flyby is an algebraic
boundary condition, NOT a chaotic integrated close approach. Decision vars [v_dep (scaled O(1)), δ, flyby
epoch]. **H-N8a SUPPORTED:** from the SAME 2 m/s departure perturbation that sent R-N7's single-shooting to a
1.26e9 km miss, the node optimizer converges feasible (Δv 9.26 km/s, defects ≤0.03 AU, r_p 6.4 R_J) — the
razor-sensitivity is gone. **H-N8b SUPPORTED:** from a direct-to-target seed with δ=0 (no turn presupposed)
the assist EMERGES — δ grows to −126°, defects close to sub-1000 km, Δv 9.29 (vs direct ≥25.3). Honest scope:
the problem offers a Jupiter node at a set epoch and rewards reaching it, so this is exploitation-with-emergence,
not cold sequence discovery. **H-N8c SUPPORTED:** implied r_p=6.4 R_J clears Jupiter and matches R-N7's
integrated flyby (6.5 R_J) — the node turn is R-N6-consistent by construction. Net: structuring the problem by
its patched-conic/manifold geometry (bounded turn as a boundary condition) converts the razor-thin unrecoverable
basin into a smooth landscape where gradient descent robustly finds AND holds the assist. Vindicates the
Hamiltonian/phase-space framing. Remaining for real discovery: the discrete which-body/whether-to-flyby choice
(Tisserand outer loop) + an optimality certificate (primer vector).

**Build N R-N9 — primer vector from the diff-sim adjoint: an optimality CERTIFICATE** (`scripts/primer_vector.py`).
Reverse-mode backprop through the RK4 rollout IS the discrete STM/costate, so Lawden's primer vector (|p|≤1, =1
at impulses; |p|>1 interior ⟺ a midcourse impulse lowers Δv) falls out of the gradient machinery already built —
no separate indirect solver. **H-N9a SUPPORTED:** autodiff cumulative-product STM matches finite-difference to
1.71e-6. **H-N9b SUPPORTED:** an operational midcourse re-solve confirms the primer on every testable case
(optimal |p|=0.997 → no benefit; marginal |p|=1.020 → real 0.25% gain; suboptimal |p|=2.64 → −34% Δv), with a
multi-rev case correctly guard-skipped (test-side limit, not mislabeled). **H-N9c SUPPORTED:** the real direct
Earth→Mars cruise (R-N2/R-N5 geometry) is certified **primer-optimal, |p|=0.998** — no beneficial deep-space
maneuver; an independent optimality certificate for the baseline the whole Δv campaign rests on. The primer is a
necessary (first-order) condition, not sufficient; exactly-180° transfers excluded (Φ_rv singular). Lawden 1963;
Conway 2010.

**Build N R-N10 — Tisserand–Poincaré outer loop for flyby-SEQUENCE discovery** (`scripts/tisserand_graph.py`).
The discovery capstone: the discrete which-body/when structure gradient descent provably can't find (R-N7 null).
Tisserand parameter conserved across a flyby (v∞²/v_P²=3−T_P); flyby walks the craft along a constant-v∞ contour.
**H-N10a SUPPORTED:** the invariant is exact — T constant to ~1e-15 along a contour, = 3−(v∞/v_P)². **H-N10b
SUPPORTED (the Voyager result):** at launch v∞=9 km/s a single Earth flyby reaches r_a=2.88 AU, but the SEQUENCE
(4 Earth v∞-leveraging flybys → cross Jupiter → 1 Jupiter flyby, δmax=148°) achieves SOLAR-SYSTEM ESCAPE (e≥1) —
the classic Jupiter-powered escape, discovered by enumeration; honest negative: v∞=7 km/s stalls at 3.21 AU
(never reaches Jupiter). **H-N10c SUPPORTED:** every encounter's implied r_p clears the surface (min 1.5 R_J).
Deterministic enumeration, NOT learned discovery; a reachability skeleton (resonant phasing / launch window is the
next layer). Strange & Longuski 2002; ties to Build H's CR3BP manifolds (Koon-Lo-Marsden-Ross 2011). **Completes
the discovery toolkit: Tisserand enumerate (R-N10) → flyby-node optimize (R-N8) → primer certify (R-N9)** — the
system-level answer to "can it discover a Voyager-style tour," with each layer's honest scope stated.

**Build N R-N11 — end-to-end grand-tour pipeline on real ephemerides** (`scripts/grand_tour.py`). The three
layers wired into one pipeline: Tisserand PROPOSE (R-N10) → flyby-node OPTIMIZE (R-N8, Lambert reach + bounded
node turn) → primer CERTIFY (R-N9). **H-N11a SUPPORTED:** offline (analytic circular Jupiter, CI-safe) the
pipeline composes end-to-end — propose escape → reach Jupiter to 0 km → post-flyby escape (a=−134 AU, e=1.038)
→ primer certifies the leg (|p|=0.996). **H-N11b SUPPORTED:** on REAL Horizons Earth+Jupiter states
(2005-08-12, cached; real v_Jupiter=12.45 vs 13.06 circular; Jupiter Keplerian-propagated to the flyby epoch —
live Horizons unavailable during a network outage) all three layers run on real data, the real flyby reaches
escape — AND the primer correctly flags that this un-searched epoch is badly phased (sweep ~59°, dep Δv 33 km/s,
|p|=3.14, DSM needed): the CERTIFY layer catching a bad launch window is the pipeline working. **H-N11c
SUPPORTED:** honest scope — fixed epoch + min-Δv TOF pick, no launch-window/phasing search; H-N11b's badly-phased
real result is the concrete signpost to that next layer.

**Build N R-N12 — launch-window / phasing search** (`scripts/launch_window.py`). The layer R-N11's primer
flagged. Earth→Jupiter porkchop (analytic circular coplanar). **H-N12a SUPPORTED:** the search finds the
well-phased window — dep Δv=8.801 km/s (Hohmann ideal 8.79, sweep 182°), windows recurring at 396 d ≈ the
synodic period (398.9 d); worst anti-window 43 km/s (R-N11's regime). Turns R-N11's arbitrary 33 km/s epoch
into an 8.8 km/s transfer. **H-N12b REFUTED-as-phrased → corrected:** the primer certifies DSM-optimality,
which is ORTHOGONAL to phasing — the expensive anti-window is still primer-optimal (|p|=0.989); R-N11's
|p|=3.14 was a specific contorted geometry with a genuine DSM, not a generic bad-phasing signature. Two tools:
porkchop = phasing metric, primer = DSM metric. **H-N12c PARTIAL:** the v∞-leveraging orbits sit near low-order
resonances (7:4, 7:3, 5:1) so the resonant-return mechanism is available, but the greedy energy-max pump doesn't
land on exact resonances — exact multi-flyby phase-closure (tune flybys to resonances) is the remaining piece.

**Build N R-N13 — amortized mission-planner** (`scripts/amortized_planner.py`). The north-star generalization:
learn (mission params) → (warm-start decision variables) so a NEW mission is inference + a few diff-sim steps,
not a full re-search. Family: aim an Earth-departure transfer at a Jupiter flyby point under REAL perturbed
Sun+Jupiter dynamics; mission θ=(Jupiter arrival angle, TOF); labels from backtracking Gauss-Newton through the
differentiable rollout; tiny MLP θ→v_dep*. **H-N13a SUPPORTED:** test RMSE 0.035 km/s ≈ 1.8× train, 16× better
than nearest-neighbour lookup and 148× better than the mean — the solution manifold is smooth and generalizes.
**H-N13b SUPPORTED (modestly):** MLP warm-start miss@0 is 5× lower than a cold Lambert seed and cuts refinement
from 3 to 2 Gauss-Newton steps (honest caveat: Newton is superlinear so the step win is small; the 5× lower
initial residual is the cleaner signal). **H-N13c SUPPORTED (the boundary):** as the target crosses past closest
approach the solution-map sensitivity ‖∂r_end/∂v‖ grows 82× and conditioning 261× — amortization is
REGIME-BOUNDED: learnable in the smooth pre-flyby (node-transcription) regime, unlearnable in R-N7's post-flyby
razor basin. Closes the arc R-N7 (raw gradient can't exploit) → R-N8 (node transcription smooths it) → R-N13 (the
smoothed regime is exactly the amortizable regime).

**Build N R-N14 — exact resonant-return phase closure + V∞-leveraging** (`scripts/resonant_leveraging.py`).
Closes R-N12c's remaining piece. Tune each Earth flyby to an exact N:M resonance (craft N orbits per M
Earth-years → period (M/N)·T_⊕) so the craft returns to the same inertial encounter where Earth also is after
M whole years. **H-N14a SUPPORTED (beyond prediction):** eight tuned resonances (3:2…1:2) return to Earth to
~1e-3–1e-1 km (machine precision, not the predicted <1e5 km) and the low-order ladder is single-flyby-walkable
(adjacent pump-angle gaps 8.8–32.9° < δmax 64.9°) — R-N10's greedy pump only landed NEAR these rungs, this
lands ON them. **H-N14b SUPPORTED:** apoapsis-Δv leverage |Δv∞/Δv| ≈ 5.9 (2:3), 5.4 (3:4), 1.3 (5:4) — the
Sims-Longuski VILM amplification. **H-N14c SUPPORTED, prediction CORRECTED:** Sun-only closure is exact;
Jupiter leaves a bounded, robust residual (0.007 AU over 3 yr); but my pre-registered "departure tweak
re-closes it" was wrong — departure control over M years is razor-ill-conditioned (the same R-N13/R-N7
sensitivity), stalling at 178 m/s; the apoapsis mid-arc TCM re-closes it (27.8 m/s → sub-km) because leverage
makes mid-arc control well-conditioned. Conditioning dictates WHERE to burn — the R-N13 razor resurfacing one
layer up.

**Build N — coplanar gravity-assist arc COMPLETE (R-N6 → R-N14).** Represent (N6) → raw-gradient null (N7) →
flyby-node fix (N8) → primer certificate (N9) → Tisserand sequence discovery (N10) → end-to-end pipeline on real
data (N11) → launch-window/phasing (N12) → amortized mission-planner (N13) → exact resonant phase-closure (N14).

**Build N R-N15 — 3-D inclination-pumping flyby** (`scripts/inclination_flyby.py`). Opens the 3-D frontier by
questioning the coplanar assumption that underlay all 14 prior rounds. Real-world anchor: Ulysses reached a ~79°
solar-polar orbit from one Jupiter flyby. A flyby conserves |v∞| and rotates the v∞ vector in any direction
(≤δmax); tilting it out of the ecliptic buys inclination via v_out = v_planet + v∞_out — free (geometry, not Δv).
**H-N15a SUPPORTED:** the 3-D diff-sim conserves |v∞| (3e-7) and e (1e-8) through a resolved out-of-plane pass,
deflection matches 2·arcsin(1/e) — R-N6 fidelity, out of plane. **H-N15b SUPPORTED (calibration noted):** max
free inclination 17.8°/27.3°/43.5° at v∞=4/6/9 — the Ulysses mechanism, growing toward polar; the pre-registered
">25°" holds for v∞≳5.5 (honestly under 20° below that). **H-N15c SUPPORTED:** the inclination–aphelion trade —
at low inc the flyby can escape (unbounded), at high inc (>35°) aphelion is bounded at 12.2 AU; one flyby cannot
maximize both energy and inclination (they compete on the shared Tisserand contour).

**Build N R-N16 — 3-D Tisserand graph: multi-flyby inclination cranking + the analytic ceiling**
(`scripts/tisserand_3d.py`). Synthesis of R-N10 (in-plane sequence discovery) + R-N15 (single-flyby pump): a
SEQUENCE of same-body flybys cranks inclination further (the Cassini-at-Titan mechanism). **H-N16b SUPPORTED
(clean closed form):** the reachable-inclination ceiling (flybys→∞) is **arcsin(v∞/v_P)**, independent of body
mass — reproduces R-N15 exactly (Jupiter@v∞6 = 27.4°). **H-N16a SUPPORTED, prediction REFINED:** multi-vs-single
gain is Jupiter 1.0× (one big-δmax flyby reaches the ceiling — why Ulysses needed one Jupiter flyby), Mars 2.1×
(genuine small-δmax staircase), Titan 1.5×; my "small δmax ⇒ many flybys" was not universal — near v∞≈v_P the
orbit plane is hypersensitive to v∞, so the true governor is θ*/δmax (geodesic length / step). **H-N16c
SUPPORTED (mission-design law):** min v∞ for target inclination i is **v_P·sin(i)**; polar needs v∞ ≥ v_P — a
heliocentric Jupiter flyby is capped sub-polar, Titan (v∞≈v_P) approaches polar (why solar-polar missions are
hard and Cassini used Titan, not the Sun).

**Build N R-N17 — 3-D diff-sim flyby-node optimizer: target a specified inclined science orbit**
(`scripts/nbody_flyby_node3d.py`). Puts the differentiable simulator (the project's core method) back at the
center of the 3-D frontier: backprop through a 3-D **Rodrigues flyby node** (turn δ≤δmax about an arbitrary
axis, azimuth φ), two smooth Sun-only legs, optimizing to hit orbital ELEMENTS (inclination, aphelion) — not a
position. This is R-N8's node transcription generalized to 3-D + element targeting, and it validates R-N16's
analytic ceiling through the actual diff-sim. **H-N17a SUPPORTED (correction):** from a strictly in-plane Earth
departure the optimizer discovers the out-of-plane turn and hits i*=15° (defect 0.000°, leg-1 ≪ SOI); exact δ=0
is a flat critical point (inclination is second-order in δ), so the flyby is seeded with a small symmetry-
breaking turn — and `arccos(h_z/|h|)` NaNs the gradient at inc=0, fixed by targeting inclination through its
cosine. **H-N17b SUPPORTED (headline):** with leg-1 fixed (v∞=5.84), the diff-sim feasibility boundary in i* is
**26.6° = arcsin(v∞/v_P)** to <0.1° — reached below, floored above. The backprop optimizer independently
confirms R-N16's closed form under the true differentiable dynamics (analytic-graph ↔ diff-sim closed).
**H-N17c SUPPORTED (correction):** the single node lives on ONE Tisserand contour, so inclination and energy
compete — the Pareto frontier of max inclination vs demanded aphelion falls monotonically (aph 5.2→12.0 AU →
max inc 26.6°→12.3°); my "aphelion pins" framing sharpened to a sloped 2-D reachability frontier.

**Scope.** R-N17's demonstration is the patched-conic model — an instantaneous node rotation between two
Sun-only Kepler legs, with v∞ held fixed for H-N17b/c — not a finite-duration integrated flyby with
third-body perturbations on the legs. The ceiling `arcsin(v∞/v_P)` is a model-independent *kinematic* identity
for a single unpowered flyby (`v_out = v_P + v∞`, `|v∞|` conserved; derived independently in R-N15/R-N16); what
is specific to this setup is the diff-sim confirmation and the numeric feasibility boundary. A finite-duration
integrated flyby (R-N7's razor basin) or a multi-node tour would re-test the surrounding claims.

**Build N R-N18 — multi-node diff-sim inclination staircase: realize the R-N16 bound + re-encounter cost**
(`scripts/nbody_tour_incl.py`). The multi-node continuation of R-N17 (single node) and the diff-sim test of
whether R-N16's greedy-geodesic ceiling — flagged there as only an upper bound "pending phasing" — is actually
realizable. A real K-node tour: K bounded Rodrigues rotations (|δ_k|≤δmax) with real Sun-only coast legs
(closure ~1.6e-3 AU) between nodes, backprop-optimized. Real anchor: **Solar Orbiter** cranks heliocentric
inclination to ~33° with ~7-8 **Venus** gravity assists (arcsin(19/35.02)=32.9°). **H-N18a SUPPORTED:** the
diff-sim Venus tour cranks monotonically (K=1→11.8°, 6→32.8°), far exceeding one flyby, every |δ_k|≤δmax — the
Solar Orbiter staircase in the differentiable simulator. **H-N18b SUPPORTED:** it saturates at **32.8°** vs the
ceiling 32.9° (<0.1° short) and does not exceed it — the real tour REALIZES R-N16's greedy bound (achievable,
not just analytic). **H-N18c SUPPORTED (goalpost corrected):** node count tracks **θ*/δmax** — Venus (θ*/δmax≈
5.5) needs ~5 nodes to the ceiling (order Solar Orbiter's 7-8), Jupiter (θ*/δmax≈0.4) reaches it in ONE; my
verdict code first used a stricter bar than the pre-registered falsifier (Venus ≤2 for 0.9·ceiling; it needed
4), corrected to judge against the pre-registration (recurring R-N15 lesson). **DIAG (re-encounter cost):**
enforcing a resonant re-encounter (constant period) caps inclination below the free ceiling — v∞_out lies on a
circle whose max inclination gaps the ceiling by 0.3–4° as the resonant period grows; the R-N17c inclination–
energy competition, now as a phasing price. Real inclination tours pay for re-encounter.

**Build N R-N19 — V∞-leveraging RAISES the inclination ceiling: the Δv–inclination exchange rate**
(`scripts/vinf_leverage_incl.py`). The round that BREAKS the ceiling instead of realizing it — the first Δv
budget in the 3-D thread. Combines R-N14 (V∞-leveraging: apoapsis Δv changes v∞ with measured leverage ≈5.94)
with R-N16/N18's ceiling arcsin(v∞/v_P): since the ceiling grows with v∞, spending Δv to pump v∞ raises it.
**H-N19a SUPPORTED:** a 2 km/s leveraged budget pumps v∞ 5→17 and adds **25°** of ceiling — Δv traded for
inclination above the free single-v∞ bound. **H-N19b SUPPORTED:** the exchange rate d(inc)/d(Δv)=L/(v_P·cos i)
diverges near polar (0.66 vs 0.016 °/(m/s) at 89° vs 45°, cheap at the margin) while the cumulative Δv to polar
is (v_P−v∞₀)/L ≈ 4.1 km/s (real, large). **H-N19c SUPPORTED (honestly bracketed):** leveraged-GA inclination is
2–12× cheaper in Δv than a direct plane change (2·v·sin i/2) — ~10× at favorable leverage/low i, shrinking to
~2× as leverage degrades toward polar (R-N14 L→1.3). Judged against the pre-registered falsifier (ratio<2×),
NOT the "≥5×" point-estimate — the honest headline is "cheaper across the board, order-of-magnitude only at
favorable leverage." Mechanism/exchange-rate study, never a Δv beat of a flown mission; both baselines are
honest brackets. This is why Ulysses / Solar Orbiter crank inclination with assists, not burns.

**Build N R-N20 — diff-sim tour DISCOVERS leverage-then-crank end-to-end (the capstone)**
(`scripts/tour_discover_leverage.py`). The north-star round: DISCOVER, not derive. R-N19 derived the
leverage-then-crank strategy analytically; R-N20 shows a differentiable optimizer — given ONLY a target
inclination and a Δv-minimizing objective, never told to leverage, seeded at ZERO leverage — DISCOVERS it.
**H-N20a SUPPORTED:** target below the free ceiling → pure-crank tour discovered (Δv 7 m/s ≈ 0); doesn't spend
fuel it doesn't need. **H-N20b SUPPORTED (honest scope):** target above the ceiling → discovers it must
leverage, spends 1.19 km/s (R-N19 analytic 1.20, 1% off) pumping v∞ 5→12 then cranking. The 1% match is a
CONSISTENCY CHECK (same reduced physics — L, ceiling — so the optimum IS the R-N19 formula by construction),
NOT independent Δv validation; the genuine result is the STRATEGY discovery. **H-N20c SUPPORTED:** efficient
structure — pumps |v∞| to exactly v_P·sin(i*) (ratio 1.00), no over-pump, then cranks. Robust across targets
35°/50°/65° (0% off). Reduced-order transcription (leverage from R-N14's real-sim 5.94, crank from R-N16);
L=6 flagged optimistic → discovered Δv a lower bound. **[R-N21 CORRECTS this: L=6 is NOT materially optimistic
— it is conservative below i*≈25° and ≤1.3× optimistic above.]**

**Build N R-N21 — the ENVIRONMENT-measured leverage curve L(v∞): a CORRECTION round** (`scripts/real_leverage_curve.py`).
Questioned the load-bearing premise of R-N19/R-N20 by MEASURING leverage from the real Sun-only diff-sim (R-N14
machinery) instead of supplying a constant. A pre-run measurement REFUTED all three record-derived hypotheses.
**H-N21a REFUTED:** leverage does NOT degrade with v∞ — best-resonance L stays 3.1–8.4 across v∞∈[3,25];
R-N14's cited "1.33 at v∞=8" is reproduced EXACTLY but only on the low-apoapsis 5:4 resonance (apoapsis 1.13
AU) — a resonance-selection artifact, since leverage is set by apoapsis distance (Sims-Longuski), not v∞.
**H-N21b REFUTED:** integrating the measured L(v∞), the real cumulative-Δv / constant-L6 ratio is 0.80 (i*=15°)
→ 1.21 (i*=40°) — L=6 is conservative below 25°, ≤1.3× optimistic above; corrects my own R-N20 caveat.
**H-N21c REFUTED:** Earth leveraging does NOT saturate — the 1:2 family is feasible to v∞=25 km/s (ceiling
57°), reaching i=30° for 2.10 km/s; the real bound is rising Δv + shrinking per-flyby δmax (7.2° at v∞=25 →
crank steps get expensive), not leverage collapse. NET: the leverage-degradation premise threaded through
R-N14→N19→N20 was largely a resonance-selection artifact; real best-resonance leverage is robust and set by
apoapsis. Honesty scope: cumulative Δv is a VILM quadrature over measured single-leg leverages, not a monolithic
multi-leg rollout (still the frontier).

**Build N R-N22 — leverage-then-crank COMPOSED in one real multi-leg diff-sim tour** (`scripts/real_leverage_crank_tour.py`).
The monolithic-rollout frontier: compose LEVERAGE (R-N21) + CRANK (R-N18) in ONE real Sun-only RK4 tour with
real resonant re-encounter closure every leg — the environment-grounded version of R-N20's reduced capstone.
**H-N22a SUPPORTED:** from v∞₀=8 (base ceiling 15.6°), a 15-leg real leverage staircase (retrograde apoapsis
burns, 1.50 km/s) pumps v∞ 8→15.24 (ceiling→30.8°, closures ≪ SOI), then a 5-flyby real crank staircase (v∞
rotated about V_E → |v_out| preserved → real 1-yr-resonance closure) reaches 29.7° > 15.6° base (Δ=+14.1°),
crank free. **H-N22b SUPPORTED (corrects R-N20's fixed δmax=35°):** the physical δmax(v∞) shrinks as leverage
climbs v∞ → up to 5× more crank flybys than a fixed 35° (worst at i≈60°). **H-N22c SUPPORTED (corrects R-N18's
single-circle DIAG):** choosing the best resonance each encounter, the resonance-circle inclination reaches
within 0.0–0.2° of the free ceiling arcsin(v∞/v_P) — re-encounter is NOT the binding limit; the near-1:1
resonance dissolves R-N18's throttle. NET: the strategy R-N20 discovered composes end-to-end in the real
environment; the real binding limit is crank flyby count (per-flyby δmax) + leverage Δv, not re-encounter
geometry. Honest scope: forward tour (leverage then crank as phases), not a single backprop-through-everything
joint optimum — that remains the frontier. (Two of three verdicts are corrections surfaced by pre-run probes
overturning my intuitions; H-N22a first REFUTED on a real perihelion-lift bug, caught and fixed before record.)

**Build N R-N23 — does the composition survive a real third body (Jupiter)?** (`scripts/jupiter_perturbed_tour.py`).
Questioned the Sun-only assumption underpinning the whole N15–N22 arc by adding Jupiter (real GM, circular
ecliptic — analytic, CI-safe) to the R-N22 tour. **H-N23a SUPPORTED:** Jupiter's per-leg phasing residual is
bounded and leg-dependent — 0.004–0.016 AU on long leverage legs (1:2, 2 yr, apo 2.2 AU) vs 0.0005–0.0015 AU on
short crank legs (1:1, 1 yr), a 10× ratio; real (≫ machine precision), not divergent. **H-N23b SUPPORTED:** the
mechanism survives — the v∞ pump climbs 8→15.29 (Sun-only 15.24) and the crank still reaches 29.7° (Jupiter
perturbs phasing, not the flyby-geometry-set crank). **H-N23c SUPPORTED:** re-closing each leg with a
well-conditioned apoapsis TCM (R-N14) costs ~19 m/s mean → ~292 m/s ≈ 19% of the 1.50 km/s leverage budget — a
modest, bounded overhead. NET: the Sun-only arc's conclusions survive a real third body with a quantified ~19%
correction budget (a null-of-worry, honestly reported). Honest scope: Jupiter circular/coplanar (not full JPL
ephemeris — inner planets + Jupiter's inclination/eccentricity neglected); TCM is a per-leg pure-closure
estimate, not a single accumulating closed-loop targeted tour.

**Build N R-N24 — does the composition survive the REAL solar system (full JPL ephemeris)? A CORRECTION round**
(`scripts/full_ephemeris_tour.py`). Removed the last big idealization: flew the R-N23 per-leg structure against
real JPL Horizons — eccentric Earth (e=0.0167) + Venus/Mars/Jupiter/Saturn as real heliocentric perturbers
(Earth = the flyby body → patched-conic, not a point mass), launching from where real Earth actually is and
re-encountering it by closest approach (not R-N23's cylinder crossing). Network-gated fetch-and-cache
(`.ephem_cache/`, gitignored); `--verify` offline. **H-N24a SUPPORTED:** the BARE-resonance single-leg residual
(0.0013–0.0118 AU leverage, 0.0004–0.0025 AU crank) is bounded and ~0.8× R-N23 — real eccentric Earth + inner
planets are sub-dominant; the un-leveraged resonance re-encounters real Earth about as well as circular Earth.
**H-N24b REFUTED:** the v∞-pump leverage staircase does NOT survive — open-loop it drifts 8→5.94 DOWN (every leg
missing real Earth by 3–5× SOI), and closed-loop active targeting still drifts 8→5.13 at a 3.6 km/s targeting
cost (≫ the 1.5 km/s of leverage burns). **This CORRECTS R-N22 ("leverage composes") and R-N23's H-N23b ("pump
climbs 8→15.3 unchanged"):** both relied on a circular-Earth cylinder-crossing that treats Earth as present at
every longitude, masking that each leverage burn shifts the encounter ~0.05 AU OFF real ephemeris Earth.
**H-N24c SUPPORTED:** re-closing the UNPUMPED resonance to real Earth is cheap (16 m/s/leg ≈ R-N23) — but this
maintains the bare chain, it does NOT rescue the leverage. NET: the strategy SPLITS against the real solar
system — the crank/resonance half survives, the leverage half does not; a real v∞-leverage tour must co-design
each post-leverage resonance to re-encounter real Earth (a proper Sims-Longuski VILM with resonance hopping).
Not an over-claimed refutation: real VILM missions fly leverage — only the naive fixed-1:2 staircase fails.

**Build N — out-of-plane arc: crank/resonance half REALIZED and now VERIFIED against real ephemeris (R-N15 → R-N20, R-N24a/c); leverage premise measured (R-N21), composed in the Sun-only rollout (R-N22), robust to circular Jupiter (R-N23) — but the leverage staircase CORRECTED as an idealization artifact once real ephemeris Earth is used (R-N24b).** Single-flyby pump (N15) → analytic ceiling + crank
(N16) → diff-sim single-node + inc–energy frontier (N17) → real multi-node tour realizing the ceiling in
~θ*/δmax nodes + re-encounter cost (N18) → V∞-leveraging breaking the ceiling for a Δv price (N19) → optimizer
DISCOVERS the whole leverage-then-crank strategy from a naive objective (N20). Reachable inclination =
arcsin(v∞/v_P), free; raisable by leveraging at ~2–12× the efficiency of a brute plane change; mission-design
law v∞ ≥ v_P·sin(i); and the strategy is discoverable, not just derivable. All under the patched-conic model —
with the caveat (R-N24) that the LEVERAGE half of the ceiling-raising was partly a circular-Earth artifact and
needs a co-designed resonance sequence to hold against real ephemeris; the crank/free-ceiling half is verified.

**Build N — next frontiers (unplanned, open).** A CO-DESIGNED real-ephemeris VILM tour — resonance-hopping so
each post-leverage orbit re-encounters real ephemeris Earth (R-N24 showed the naive fixed-1:2 staircase drifts
off real Earth; this is the corrected frontier, and materially harder); a single accumulating closed-loop
targeted tour (R-N23/R-N24 estimated the TCM budget per-leg, not end-to-end); a JOINT backprop-through-everything
tour optimizer (R-N22 composed leverage + crank as forward phases; R-N24 shows a real-ephemeris joint objective
must include the real-Earth encounter constraint); a learned amortized tour-planner (R-N13 style) over the
(target inclination, Δv) map. (The full-JPL-ephemeris frontier itself is now CLOSED by R-N24 — fetch-and-cache
solves the CI-safety tension.)

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
