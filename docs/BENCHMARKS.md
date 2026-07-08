# Historical-trajectory benchmark

Beyond the analytic baselines (Hohmann etc.), we can benchmark the agent against
**real flown spacecraft trajectories**. JPL Horizons serves spacecraft ephemerides
(negative NAIF IDs), so we pull the actual states real missions flew — under the
same N-body perturbations the real spacecraft dealt with — and ask: under matched
boundary conditions and dynamics, can the agent find a lower-Δv trajectory than
the one the pros actually flew?

Because spacecraft are just another Horizons ID, the existing bulk/cached
`ephemeris.py` fetches them unchanged, e.g.:

```bash
python ephemeris.py 2013-12-01T00:00:00 2014-09-20T00:00:00 86400 -202   # MAVEN cruise
```

## Verified candidate missions

Earth→Mars **direct** cruises (no gravity assists → the cleanest Δv comparisons):

| NAIF ID | Mission | Transfer | Approx. cruise window |
|---|---|---|---|
| `-202` | MAVEN | Earth→Mars | 2013-11-18 → 2014-09-22 |
| `-74`  | Mars Reconnaissance Orbiter | Earth→Mars | 2005-08-12 → 2006-03-10 |
| `-53`  | Mars Odyssey | Earth→Mars | 2001-04-07 → 2001-10-24 |
| `-76`  | Curiosity / MSL | Earth→Mars | 2011-11-26 → 2012-08-06 |
| `-168` | Perseverance / Mars 2020 | Earth→Mars | 2020-07-30 → 2021-02-18 |
| `-143` | ExoMars TGO | Earth→Mars | 2016-03-14 → 2016-10-19 |

Gravity-assist tours (see caveats — *not* fair Δv targets, but great for the
"can the agent discover an assist?" exploration):

| NAIF ID | Mission | Notes |
|---|---|---|
| `-32` | Voyager 2 | Jupiter–Saturn–Uranus–Neptune tour |
| `-61` | Juno | Earth→Jupiter with an Earth gravity assist |
| `-98` | New Horizons | Jupiter assist → Pluto |

(All IDs above verified against Horizons.) A machine-usable registry with
windows is in `scripts/missions.py`.

## Methodology

1. **Pull** the flown heliocentric trajectory over the cruise (cached).
2. **Boundary conditions:** take the post-injection Earth-departure state and the
   Mars-arrival state (with the true time-of-flight) from the flown trajectory.
3. **Solve the same problem:** the diff-sim / RL agent optimizes the transfer from
   that departure state to Mars at arrival, under the **ephemeris N-body** dynamics
   (Sun + planets), minimizing Δv.
4. **Compare** total Δv (and time-of-flight) against the flown mission.

Reconstructing the *flown* Δv: Horizons gives states, not maneuvers. Estimate it
from (a) the launch injection C3 (from the flown Earth-relative departure energy,
or published values), (b) deep-space maneuvers / trajectory-correction maneuvers
(TCMs; velocity discontinuities in the flown states, or published budgets), and
(c) Mars orbit insertion if included.

## Caveats (read before claiming "we beat NASA")

- **Real missions optimize a constrained multi-objective problem** — launch-window
  C3 limits, arrival V∞ for capture, mass/cost/risk margins, operational
  constraints — *not* pure transfer Δv. Beating them on Δv does **not** mean the
  flown trajectory was suboptimal; it optimized a different objective.
- **Launch C3 is set by the launch vehicle**, not freely optimizable. The fair,
  apples-to-apples metric is the **in-space Δv** (post-injection + TCMs + MOI), or
  a total that models the same injection.
- **Gravity-assist missions are near-impossible to beat on Δv** and shouldn't be
  used as "beat it" targets — use them to test whether the agent can *discover*
  an assist at all (the open question).
- Best "fair fight" targets are the **direct Earth→Mars cruises** above: matched
  departure/arrival states, same N-body dynamics, Δv-vs-Δv.

This benchmark is the payoff of the Tier-3 ephemeris N-body decision (see
`docs/ROADMAP.md`): it lets us demonstrate real optimality — or report its
absence — against non-ML, professionally flown solutions.
