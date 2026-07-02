# TrajectoryBot — RL for Orbital Maneuver / Trajectory Optimization

A from-scratch orbital-dynamics simulation stack driving a reinforcement-learning
agent that learns spacecraft maneuvers, benchmarked against analytically optimal
transfers (Hohmann / bi-elliptic).

Originally built in 2021 as a project for a graduate astrodynamics course
("Orbital Dynamics II"). The agent trained but never converged on a useful
trajectory. This repo revives the code to audit *why* and fix it into a
presentable project.

## The rewrite (2026) — `tbot/`

The revival is a clean, modern package rather than a patch of the 2021 code. It
keeps the original spirit (everything hand-rolled — no Basilisk / poliastro) on
an up-to-date stack: **Python 3.12, Gymnasium, Stable-Baselines3, PyTorch, `uv`**,
with **pyright `strict`** enforced on the shipped library.

```
tbot/
  orbital.py            # planar orbital elements + analytic Δv baselines
  dynamics.py           # two-body RK4 propagation
  envs/circularize2d.py # Gymnasium env: Milestone 1 (2-D circularization)
ephemeris.py            # bulk, cached, idempotent JPL Horizons access
scripts/                # experiment/training entrypoints (PPO, BC, DAgger, …)
tests/                  # pyright-strict library is covered here
```

Run it:

```bash
uv sync --extra dev          # Python 3.12 env, from pyproject/uv.lock
uv run pytest                # physics, baselines, env, solvability
uv run pyright               # strict type check (library)
uv run python scripts/train_circularize2d.py     # model-free PPO
uv run python scripts/dagger.py                  # imitation (DAgger)
```

### Milestone 1 — 2-D circularization (current)

Circularize from an elliptical orbit at minimum Δv, benchmarked against the
analytic single-impulse optimum. The env is correct and provably solvable (a
scripted controller hits 100% at Δv ≈ optimal), and it's the testbed for *how*
to learn the maneuver. Full methodology + analysis in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). Result so far:

| Method | Success | Δv vs optimal |
|---|---|---|
| Scripted expert | 100% | 1.00 |
| Cold PPO (model-free, 8 runs) | 0–3% | — |
| Behavior cloning | ~16% | ~1.37 |
| DAgger | ~45% | ~1.5 |
| **Differentiable-sim policy gradient** | **81%** | **1.26** |

Takeaway: model-free RL and imitation wall out because they discard the known,
**exactly differentiable** dynamics. The **differentiable-simulation policy
gradient** (backprop through the RK4 rollout) exploits them and wins decisively —
and because it optimizes true Δv, it's the method positioned to *beat* analytic
baselines where they aren't optimal (see [`docs/ROADMAP.md`](docs/ROADMAP.md)).

### Roadmap

2-D circularize → full **3-D attitude control** (quaternion orientation; needed
for plane/inclination changes) → GEO transfer → multi-body (Earth–Moon,
Earth–Mars, capture). See [`docs/AUDIT.md`](docs/AUDIT.md) for the audit of the
2021 code and [`docs/BASELINE.md`](docs/BASELINE.md) for the original-code repro.

---

## The 2021 original stack (all hand-rolled — no Basilisk / poliastro)

- **Ephemerides:** real planet/moon states pulled from **JPL Horizons** via
  `astroquery.jplhorizons` (`TBot_Setup.py`).
- **N-body gravity:** `BODY_DICT` of gravitational parameters + radii for the
  Sun, all 8 planets, and the Moon; equations of motion integrated with
  `scipy.integrate.odeint`.
- **RL agent:** **PPO written from scratch** in PyTorch (`ppo_torch.py`) —
  actor + critic MLPs, GAE, clipped surrogate objective. No stable-baselines.
- **Environment:** a custom Gym env (`spacecraft/basic_env1.py`, registered as
  `gym_push:basic-v0`) exposing state (body positions, spacecraft pos/vel,
  orientation, fuel) and discrete thrust/orientation actions, with a shaped
  reward toward a target orbit.
- **Baseline:** `hohmann_test.py` — the analytic transfer the agent must beat.

## Repository layout

| Path | What it is |
|---|---|
| `v2/` | **Working base** — newest iteration (Nov 2021). Packaged; adds `spacecraft/` env, `TBot_{Setup,Functions,Inputs}`, quaternions, `initial_conditions.py`. |
| `archive/trajectorybot-oct2021/` | Fullest earlier tree — includes `2body_ode.py`, `hohmann_test.py`, ODE tests, `reward1/2.py` variants. |
| `archive/trajbot-dev/` | Mid-development iteration. |
| `archive/final-report-may2021/` | The state at course submission, with the writeup. |
| `docs/final-report.md` | The original course final report (author's own postmortem). |
| `docs/progress-report.md` | The mid-project progress report. |

Trained weights, plots, `data.csv`, and other run artifacts are intentionally
**not committed** (see `.gitignore`); they regenerate by training. Raw `.docx`
report sources are kept locally under `docs/reports-src/` (gitignored); the
committed `.md` conversions are the canonical copies.

## Known failure modes (from the author's 2021 postmortem)

These are the starting points for the audit, straight from `docs/final-report.md`:

1. **Model save/load never worked** — every run trained from scratch, so no
   progress accumulated across sessions.
2. **State/action space too large** — full 3D with 8 actions (6 of them
   orientation changes) → the agent burned episodes flipping in place. The
   author concluded it should have been 2D (single yaw DOF).
3. **Training too slow** — an Earth–Mars episode was days long, so the target
   was downgraded to the Moon; even then only ~400–800 episodes/night.
4. **Reward function never converged** — distance to the target *increased*
   regardless of the reward shaping tried; no reward function produced approach
   behavior.

## Suspected bugs to verify first (from a quick code read — unconfirmed)

- `mainPPO.py` sets the PPO learning rate `alpha = 2` (should be ~`3e-4`); a
  learning rate that large would diverge the policy/value nets outright.
- `n_games = 1` / tiny `N` / `batch_size` in the top-level runner — far too
  small to learn anything.
- Reward-sign / gradient direction consistent with "distance keeps increasing."

## Audit / revival roadmap (target maneuvers)

Single-body (Earth-centered), baseline = Hohmann / bi-elliptic:
1. Circularize from an elliptical orbit
2. Plane change
3. GEO transfer

Multi-body:
4. Earth–Moon transfer
5. Earth–Mars transfer
6. Capture maneuver at the target body

## Provenance

Recovered from personal backups. Originally authored in 2021.
