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
    """A hand-coded 'coast to apoapsis, then burn prograde' policy must succeed
    with Δv within 1.6x of the analytic single-impulse baseline."""
    env = Circularize2DEnv(Circularize2DConfig(max_steps=4000))
    env.reset(seed=7)
    baseline = orb.circularize_apoapsis_dv(
        orb.orbital_elements(env._state, env.cfg.mu).r_p,
        env._r_target, env.cfg.mu)

    terminated = truncated = False
    info: dict = {}
    dv_per_step = env.cfg.thrust_acc_max * env.cfg.dt
    while not (terminated or truncated):
        s = env._state
        r = float(np.hypot(s[0], s[1]))
        v = float(np.hypot(s[2], s[3]))
        el = orb.orbital_elements(s, env.cfg.mu)
        # burn only very near apoapsis: the band must be tighter than the
        # success tolerance on a, else we circularize at the wrong radius.
        near_apo = abs(r - env._r_target) / env._r_target < 0.01
        v_circ = orb.speed_circular(r, env.cfg.mu)
        if el.e < env.cfg.e_tol or not near_apo or v >= v_circ:
            action = np.zeros(2, dtype=np.float32)
        else:
            # burn prograde, but only deliver the remaining velocity deficit so we
            # settle onto the circular speed instead of overshooting it.
            mag = min(1.0, (v_circ - v) / dv_per_step)
            vhat = np.array([s[2], s[3]]) / v
            action = (vhat * mag).astype(np.float32)
        _, _, terminated, truncated, info = env.step(action)

    assert info["success"] is True
    assert info["dv_used"] < 1.6 * baseline
