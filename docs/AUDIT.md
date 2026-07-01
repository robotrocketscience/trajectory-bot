# TrajectoryBot — Code Audit (2026)

Audit of the recovered 2021 graduate astrodynamics course project. Goal: understand what it does,
why it never learned a trajectory, and what to fix to make it a working,
presentable project.

Sources read: `v2/` (working base, Nov 2021), `archive/*` variants, and the
author's own `docs/final-report.md` / `docs/progress-report.md`.

---

## 1. What it's trying to do

Learn spacecraft maneuvers with reinforcement learning on a self-built
orbital-dynamics simulator, and beat the analytically optimal transfer
(Hohmann / bi-elliptic) on Δv.

Pipeline:

1. **Ephemerides** — `planetPositionVelocity()` queries **JPL Horizons**
   (`astroquery.jplhorizons.Horizons`, heliocentric `@sun`) for each body over
   a 1-hour window at 1-second steps, converts AU→km and AU/day→km/s, and
   caches them in an ephemeris dict (`generateEphDict`).
2. **N-body gravity** — `getBodyAccel()` sums `-μ (r_sc − r_body)/|r_sc − r_body|³`
   over all 10 bodies in `BODY_DICT` (Sun + 8 planets + Moon).
3. **Propagation** — `spacecraftEOM()` advances the spacecraft with a first-order
   kinematic step `r += v·t + ½a·t²`, `v += a·t`.
4. **RL** — a from-scratch PPO (`ppo_torch.py`) actor/critic chooses actions;
   the env (`SpaceCraftEnv`) returns state + reward; target maneuver was an
   Earth→Moon transfer (downgraded from Earth→Mars for tractability).
5. **Baseline** — `hohmann_test.py` (in the oct2021 variant) computes the
   analytic transfer to compare against.

---

## 2. Root-cause findings (why it never learned)

### F1 — Fatal: discrete agent driving a continuous environment
`ppo_torch.py` is the discrete-action PPO (final layer `nn.Softmax`, samples from
`torch.distributions.Categorical`, `choose_action` returns a **single integer**).
It is essentially the widely-circulated Phil Tabor PPO tutorial, unmodified.

But `v2/spacecraft/basic_env1.py` defines a **continuous** action space
`spaces.Box(low=[-1,-1,-1,0], high=[1,1,1,1])` (roll, pitch, yaw, throttle) and
`_take_action` indexes `action[0..3]`. `mainPPO.py` sets
`n_actions = env.action_space.shape[0] = 4`, so the Categorical policy emits a
distribution over 4 *indices* and samples one scalar. The env then tries to
subscript that scalar as a 4-vector.

The two halves were never reconciled — V2 was caught mid-migration from an
earlier **discrete** env (the old `basic_env.py`, whose `_take_action` uses
`if action_type == 1: ...`) to a new **continuous** env, and the agent side was
never migrated. This alone makes coherent learning impossible.

**Fix:** pick one. Recommended: continuous control — replace the actor head with
a Gaussian policy (mean + log-std, `Normal` / `TanhNormal`) and remove the
Softmax/Categorical. Alternatively keep discrete PPO and give the env a small
discrete action set (thrust prograde / retrograde / normal / coast). For the
maneuvers on the roadmap, continuous thrust is the right model.

### F2 — Fatal in V2: the reward function is a stub
In `v2/TBot_Functions.py`, `getReward()` is literally:
```python
def getReward(self, action):
    reward = 1
    return reward
```
and `checkDone()` `return False`. So in the newest checkpoint there is **no
learning signal** (constant reward) and episodes never terminate from inside the
env. The real reward attempts were commented out here and live in the archived
variants (`archive/*/gym-push/gym_push/envs/reward1.py`, `reward2.py`, and the
old `basic_env.py`). Any V2 training run learns nothing by construction.

**Fix:** implement a real, potential-based reward (see §4).

### F3 — Hyperparameters guarantee divergence / no training
`v2/mainPPO.py`:
- `alpha = 2` passed as the Adam learning rate (the sane default in `Agent` is
  `3e-4`; `mainPPO` overrides it to **2**). A learning rate of 2 diverges the
  nets immediately.
- `n_games = 1` — a single episode. Nothing can be learned.
- `N = 5`, `batch_size = 64`, `n_epochs = 4` — update horizon far too short.

**Fix:** `alpha≈3e-4`, `n_games` in the thousands, `N` (rollout length) ~2048,
tune `batch_size`/`n_epochs` normally.

### F4 — Model load never enabled (matches the report's "couldn't continue training")
The report says weights couldn't be loaded, so every run restarted from scratch.
In fact `Agent.load_models()` / `save_models()` **exist and work**; they're just
never called — `# agent.load_models()` is commented out in `mainPPO.py`, and
`save_models()` is only reached on a `best_score` improvement that a constant
reward can't produce. The capability is present; the wiring isn't.

**Fix:** call `load_models()` on start (guarded by file existence) and
`save_models()` on a real running-average improvement.

---

## 3. Secondary bugs / smells

- **B1 — ambiguous env wiring.** `mainPPO.py` does `env = gym.make('gym_push:basic-v0')`
  (the pip-registered env from an archive variant) while also
  `from spacecraft.basic_env1 import SpaceCraftEnv` and a commented
  `# env = SpaceCraftEnv()`. Which env actually runs is unclear and depends on
  what's pip-installed. Collapse to one canonical env.
- **B2 — reward computed before the action is applied.** `env.step()` calls
  `getReward()` *then* `_take_action()`, so reward reflects the pre-action state.
- **B3 — `self.action` never set in V2 step.** `_take_action` sets `self.throttle`
  but not `self.action`; then `step()` uses `self.action[3]` when `throttle > 0`,
  and `self.action` is `None` → `TypeError` on any positive thrust.
- **B4 — malformed observation space.** `defineObsSpace()` sets `low = high = 18e9`
  for most dimensions (line ~116 sets the lower bound to `+18e9`, same as upper)
  and skips index 5. The `Box` bounds are degenerate.
- **B5 — reward leaks into the observation.** `observeState()` stacks
  `self.reward` into the state vector — the agent observes its own reward
  (non-Markovian, and a shortcut for the value net).
- **B6 — orientation math is muddled.** `_take_action` mixes unit-vector and
  `pyquaternion.Quaternion`: `pitch*roll*yaw.rotate(self.orientation)` binds
  `.rotate` to `yaw` only, then left-multiplies a vector by quaternions.
  `self.fuelMass -= 5*sum(action[0:3])` can *add* fuel for negative inputs.
- **B7 — first-order, non-symplectic propagation.** `r += v·t + ½a·t²` at 1 s
  steps accumulates energy error over multi-day transfers. Consider RK4 or a
  symplectic integrator, and/or a patched-conic / restricted-three-body frame
  rather than full heliocentric N-body for the training loop (speed + accuracy).
- **B8 — a fresh JPL Horizons HTTP query per reset/refresh.** `generateEphDict`
  hits the network; over thousands of episodes this is slow and rate-limited.
  Cache ephemerides to disk (this is also why `data.csv` exists).

---

## 4. The reward-function problem (the crux)

The author's postmortem nails it: no reward shaping produced approach behavior;
distance to the target kept *increasing*. Two dead ends are visible in the code:

- **`reward1/2.py` (Oct 2021) is effectively supervised imitation.** It computes
  the *analytic* lunar Hohmann transfer (lead angle, TOF, `deltaV = 4.04`
  hardcoded), integrates the ideal transfer ellipse with `odeint`, and rewards
  the agent for *tracking that precomputed ellipse* (`exp(3/(1+Δr))`). As the
  report itself notes, if you can already define the optimal path, using RL to
  follow it defeats the purpose. It also stages the episode with brittle
  `self.beginxfer` flags and magic numbers.
- **Reward scale/shape is explosive.** Terms like
  `exp(1/(0.001+|Δ|))` blow up near the target and are near-flat far away, giving
  almost no gradient over the region the agent actually explores — consistent
  with "improves score slightly but never approaches."

**Recommended direction:**
- Use **potential-based reward shaping** (Ng et al. 1999) around a physically
  meaningful potential — e.g. negative of the Δv-to-target-orbit, or negative
  distance-in-orbital-elements — which is provably policy-invariant and won't
  incentivize the wrong optimum.
- Start with the **single-body, 2-D, continuous-thrust** case (circularize from
  an ellipse), where the optimum is Hohmann and the reward can be
  `−Δv_used` plus a terminal bonus for matching target orbital elements
  (a, e, i) within tolerance. 2-D is a **validation stepping-stone** that strips
  the problem to the fewest failure modes — **not** the final design.
- Then extend to **full 3-D attitude control**, which is the real target:
  inclination / plane-change maneuvers are inherently 3-D and require the agent
  to control orientation, not just in-plane thrust. Scale through plane changes,
  GEO transfer, and finally multi-body (Earth–Moon, Earth–Mars, capture), where a
  patched-conic or CR3BP formulation keeps episodes fast.

---

## 5. Which tree to work from

`v2/` has the cleanest structure (packaged env, ephemeris caching scaffold,
continuous action space, quaternion attitude) **but** its reward is stubbed and
its agent is still discrete. The archived variants have real (if flawed) reward
logic and the Hohmann baseline. Plan: **build on `v2/`'s structure**, port the
Hohmann baseline from `archive/trajectorybot-oct2021/hohmann_test.py`, and write
a *new* reward from §4 rather than reviving `reward2.py`.

---

## 6. Prioritized fix order

1. Decide action paradigm (→ **continuous**) and make agent + env agree (F1).
2. Reduce to single-body 2-D circularization; write the `−Δv` + terminal-bonus
   reward (F2, §4); wire a real `checkDone` (target-orbit tolerance / fuel /
   impact / step cap).
3. Fix hyperparameters and enable save/load (F3, F4).
4. Clean the env mechanics: canonical env (B1), reward-after-action (B2),
   `self.action` handling (B3), obs-space bounds (B4), drop reward-in-obs (B5),
   sane attitude/fuel model (B6).
5. Cache ephemerides to disk (B8); consider RK4/symplectic + patched-conic (B7).
6. Validate against Hohmann on circularize → plane change → GEO transfer, then
   move to multi-body.

---

## 7. Training hardware

- **Primary trainer** — a CUDA GPU workstation (RTX 3060 12 GB), torch 2.x + CUDA,
  `uv`-managed environment.
- **Secondary** — a second GPU box (GTX 1660 Ti 6 GB, 12 cores, 32 GB RAM) for
  parallel runs.

PPO on this problem is small and fits comfortably on either GPU.
