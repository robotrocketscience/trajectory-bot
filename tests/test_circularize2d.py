#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circularize-2D env: API conformance, invariants, and solvability."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import orbital as orb  # noqa: E402
from tbot.envs.circularize2d import Circularize2DConfig, Circularize2DEnv  # noqa: E402


def test_reset_and_step_api():
    env = Circularize2DEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["baseline_dv"] > 0.0

    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "dv_used" in info2


def test_coast_preserves_orbit():
    env = Circularize2DEnv()
    env.reset(seed=1)
    e0 = orbital_e(env)
    zero = np.zeros(2, dtype=np.float32)
    for _ in range(100):
        env.step(zero)
    # No thrust => eccentricity should be essentially unchanged.
    assert abs(orbital_e(env) - e0) < 1e-3


def test_fuel_budget_gate():
    env = Circularize2DEnv(Circularize2DConfig(dv_budget=0.0))
    env.reset(seed=2)
    full = np.ones(2, dtype=np.float32)
    _, _, _, _, info = env.step(full)
    assert info["dv_step"] == 0.0            # no fuel => no thrust applied
    assert info["dv_used"] == 0.0


def orbital_e(env: Circularize2DEnv) -> float:
    return orb.orbital_elements(env._state, env.cfg.mu).e


def test_scripted_apoapsis_burn_is_solvable():
    """A hand-coded 'coast to apoapsis, then burn prograde to circular speed'
    policy must solve the task with Δv close to the analytic single-impulse
    baseline. Run at near-impulsive thrust so the burn is concentrated at the
    apsis; the default (harder) thrust is what the RL agent must master.
    """
    for seed in (7, 0, 1, 2, 3):
        env = Circularize2DEnv(Circularize2DConfig(max_steps=6000, thrust_acc_max=0.1))
        env.reset(seed=seed)
        r_target = env._r_target
        vc_target = orb.speed_circular(r_target, env.cfg.mu)
        baseline = orb.circularize_apoapsis_dv(
            orb.orbital_elements(env._state, env.cfg.mu).r_p, r_target, env.cfg.mu)
        dv_per_step = env.cfg.thrust_acc_max * env.cfg.dt

        prev_rdot = None
        burning = False
        terminated = truncated = False
        info: dict = {}
        while not (terminated or truncated):
            s = env._state
            r = float(np.hypot(s[0], s[1]))
            v = float(np.hypot(s[2], s[3]))
            rdot = (s[0] * s[2] + s[1] * s[3]) / r     # radial speed; 0 at apsis
            el = orb.orbital_elements(s, env.cfg.mu)
            if not burning and prev_rdot is not None and prev_rdot > 0.0 >= rdot:
                burning = True                          # just crossed apoapsis
            prev_rdot = rdot
            if burning and el.e >= env.cfg.e_tol and v < vc_target:
                mag = min(1.0, (vc_target - v) / dv_per_step)   # avoid overshoot
                action = np.array([mag, 0.0], dtype=np.float32)  # pure prograde
            else:
                action = np.zeros(2, dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)

        assert info["success"] is True, f"seed {seed} did not circularize"
        assert info["dv_used"] < 1.2 * baseline
