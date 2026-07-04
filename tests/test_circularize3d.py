#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circularize-3D env (decision-layer): API + that commanded directions take effect."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import attitude  # noqa: E402
from tbot import orbital as orb  # noqa: E402
from tbot import orbital3d as orb3  # noqa: E402
from tbot.envs.circularize3d import Circularize3DEnv  # noqa: E402


def test_reset_and_step_api():
    env = Circularize3DEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["r_target"] > 0
    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert abs(float(np.linalg.norm(env._state[6:10])) - 1.0) < 1e-6


def test_commanded_prograde_burn_raises_orbit():
    # command "thrust prograde, full throttle"; the deterministic controller slews
    # the vehicle prograde and the burn should raise the orbit.
    env = Circularize3DEnv()
    env.reset(seed=1)
    r = 7000.0
    vc = orb.speed_circular(r)
    q0 = env._state[6:10]
    env._state = np.concatenate([[r, 0.0, 0.0], [0.0, vc, 0.0], q0, np.zeros(3)])
    a0 = orb3.orbital_elements3d(env._state[0:3], env._state[3:6], env.cfg.mu).a
    for _ in range(15):
        env.step(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32))  # prograde, full
    a1 = orb3.orbital_elements3d(env._state[0:3], env._state[3:6], env.cfg.mu).a
    assert a1 > a0


def test_controller_slews_thrust_axis_toward_command():
    from tbot import quaternion as quat
    env = Circularize3DEnv()
    env.reset(seed=2)
    tax = np.array(env.cfg.sc.thrust_axis)
    for _ in range(15):
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # point prograde, no burn
    # measure against the CURRENT prograde (it drifts as the craft coasts; the
    # controller tracks the moving direction, so compare to where it is now)
    r, v = env._state[0:3], env._state[3:6]
    d_now = attitude.desired_direction(r, v, np.array([1.0, 0.0, 0.0]))
    align = float(np.dot(quat.rotate(env._state[6:10], tax), d_now))
    assert align > 0.9                            # thrust axis well aligned with prograde
