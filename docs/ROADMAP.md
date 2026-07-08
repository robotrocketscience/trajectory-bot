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
    G's 1-burn search **stalled**. The enablers, honestly: the manifold/I-derived
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
    priced impulsive cleanup to the exact target). **Crucial honesty:** the J2-*blind*
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
