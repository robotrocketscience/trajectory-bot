# TrajectoryBot — RL for Orbital Maneuver / Trajectory Optimization

A from-scratch orbital-dynamics simulation stack driving a reinforcement-learning
agent that learns spacecraft maneuvers, benchmarked against analytically optimal
transfers (Hohmann / bi-elliptic).

Originally built in 2021 as a project for a graduate astrodynamics course
("Orbital Dynamics II"). The agent trained but never converged on a useful
trajectory. This repo revives the code to audit *why* and fix it into a
presentable project.

## The stack (all hand-rolled — no Basilisk / poliastro)

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
