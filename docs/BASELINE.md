# Baseline reproduction (2026)

Goal: get the 2021 code running as-is on current hardware and establish a
known-state starting point *before* changing anything. This documents the
environment, how to run each variant, and exactly what each does today.

## Environment (faithful 2021 stack)

Reproduced in a `uv`-managed **Python 3.9** venv on a CUDA GPU workstation. The
original code uses the pre-0.22 Gym registry API (`registry.env_specs`), so Gym
must be pinned to 0.21.

```bash
uv venv --python 3.9 .venv-baseline
source .venv-baseline/bin/activate
# gym 0.21 has a malformed 'opencv-python (>=3.)' specifier that uv's strict
# parser rejects; install via pip with build isolation off + pinned setuptools:
python -m pip install "pip<24.1" "setuptools==65.5.0" "wheel<0.40"
python -m pip install "gym==0.21.0" --no-build-isolation
python -m pip install "numpy<1.24" scipy pandas joblib matplotlib pyquaternion "astropy<6" astroquery
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Resolved versions: `gym 0.21.0`, `numpy 1.23.5`, `scipy 1.13.1`, `astropy 5.3.4`,
`astroquery 0.4.11`, `pyquaternion 0.9.9`, `torch 2.8.0+cu128`.

The env queries **JPL Horizons over the network** on reset and on each hourly
(sim-time) ephemeris refresh, so runs require internet and are network-bound.

## Baseline A — `archive/trajectorybot-oct2021/` (discrete). **Runs.**

```bash
python -m pip install -e archive/trajectorybot-oct2021/gym-push   # registers gym_push:basic-v0
cd archive/trajectorybot-oct2021
MPLBACKEND=Agg python mainPPO.py
```

This is the historically-runnable variant: a **discrete** action env
(`env.action_space.n`) matched to the discrete PPO. It executes end-to-end —
resets, pulls ephemerides, the agent chooses actions, reward/fuel/distance are
computed each step.

**It reproduces the original failure, live:**
- `distance to target` **increases monotonically** (observed 379.9 → 393.8 Mm
  over the captured steps) — the exact symptom from the 2021 report ("distance
  to the target steadily increases rather than decreases").
- The agent stays stuck in the reward function's **"STAGE 1"** (pre-transfer
  wait) and never progresses to the transfer/tracking stages.
- A **single episode does not terminate** within 240 s of wall-clock (many
  thousands of 1-second sim steps, each network-influenced). This is the
  "too long to train" problem, also reproduced — `checkDone` almost never fires.

So Baseline A is a *running-but-non-learning* baseline. Exactly the state we need
to improve from.

## Baseline B — `v2/` (continuous env + discrete agent). **Crashes at startup.**

```bash
python -m pip install -e v2
cd v2
MPLBACKEND=Agg python mainPPO.py
```

```
Traceback (most recent call last):
  File ".../v2/mainPPO.py", line 33, in <module>
    agent = Agent(n_actions=env.action_space.shape[0], ...)
IndexError: tuple index out of range
```

Root cause = the **ambiguous env wiring** flagged in the audit (B1). V2's
`mainPPO.py` does `env = gym.make('gym_push:basic-v0')`, which resolves to the
**discrete** env registered by whatever `gym_push` package is pip-installed —
*not* V2's own continuous `spacecraft.basic_env1:SpaceCraftEnv`. A `Discrete`
space has an empty `.shape`, so `.shape[0]` raises `IndexError`. V2 never even
reaches its own environment. (This is also why the two variants must not be
installed and run in the same interpreter without care — the registration
collides on `basic-v0`.)

## Takeaways for the fix work

1. **Build the first working loop on the oct2021 discrete variant's mechanics** —
   it already runs; the problems are reward + dynamics + wiring, not plumbing.
2. **Keep V2's cleaner structure** (packaged env, ephemeris scaffold) but fix the
   env-wiring so the intended env is the one that loads, and reconcile the
   agent's action paradigm with the env's action space (audit F1).
3. The reproduced "distance increases / stuck in STAGE 1 / episode never ends"
   trifecta maps directly onto audit findings F1–F4 and the reward crux (§4).

## Milestone framing (2D → 3D)

First target is a **verified working loop in 2-D** (single-body circularize),
because it strips the problem down to the fewest failure modes while we get the
reward + agent + termination correct. **2-D is a validation stepping-stone, not
the destination** — full 3-D attitude control is the real goal, since
inclination / plane-change maneuvers are inherently 3-D. Once 2-D learns and
beats a Hohmann baseline, extend to 3-D orientation.
