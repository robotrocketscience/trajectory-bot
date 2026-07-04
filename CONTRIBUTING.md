# Contributing

trajectory-bot is a one-author research/portfolio project: a from-scratch
orbital-dynamics simulator plus RL / differentiable-simulation agents,
benchmarked against analytic transfers. The bar for changes is "is the claim
or the code measurably better afterward, with a test or an experiment to show
it."

## Ground rules

- **Python via [uv](https://docs.astral.sh/uv/).** `uv sync --extra dev` to
  set up; `uv run pytest` to test; `uv run pyright` for types (strict on
  `tbot/`, see `pyproject.toml`). CI runs exactly these.
- **The 2021 code under `v2/` and `archive/` is historical.** It is kept for
  the project's postmortem narrative and is intentionally excluded from
  typing and CI. Don't fix it; it is the "before" picture.
- **Experiments are pre-registered.** Research changes (reward shaping,
  optimizer/aggregation knobs, curricula) should state a hypothesis, a
  prediction, and a refute-by criterion in the PR body — this repo's results
  came from that discipline and PRs that skip it are hard to evaluate.
- **Claims need verification fidelity.** Any "beats the analytic baseline" /
  fuel-efficiency claim must survive `scripts/verify_probe.py` (float64,
  dt=1 s, clean-latch filtering, both baselines). Training-fidelity numbers
  (`dvr`, in-run success) are telemetry, not claims.
- **No large binaries or results files in git.** Plots in `docs/media/` are
  the small, regenerable exception — `scripts/viz_readme.py` rebuilds them.

## Commit messages

Conventional-commit prefixes are required:
`feat:` `fix:` `perf:` `refactor:` `test:` `docs:` `build:` `ci:` `style:`
`revert:` `exp:` (research code) `chore:` (narrow housekeeping) `release:`.
Keep commits atomic; body concise (one paragraph or less).

## Branches

`feat/`, `fix/`, `exp/`, `docs:`-style prefixes mirror the commit vocabulary
(`feat/…`, `fix/…`, `exp/…`, `docs/…`, `ci/…`, `chore/…`). One concern per
branch; PRs target `main`.

## Filing issues

Title: one line, lowercase, no trailing period. Body: what happened (exact
command + output), what you expected, environment (`python --version`, OS,
GPU if relevant), and a minimal repro. For simulation-correctness issues,
include the initial orbital elements and the integrator settings.
