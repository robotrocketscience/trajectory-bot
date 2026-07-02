#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Circularize-3D env: API conformance + pointed-burn physics sanity."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import orbital as orb  # noqa: E402
from tbot import orbital3d as orb3  # noqa: E402
from tbot import quaternion as quat  # noqa: E402
from tbot.envs.circularize3d import Circularize3DConfig, Circularize3DEnv  # noqa: E402


def _q_point(v):
    """Quaternion rotating body +x onto the direction of v."""
    x = np.array([1.0, 0.0, 0.0])
    vh = v / np.linalg.norm(v)
    axis = np.cross(x, vh)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(x, vh))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else \
            quat.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi)
    return quat.from_axis_angle(axis, float(np.arctan2(s, c)))


def test_reset_and_step_api():
    env = Circularize3DEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["r_target"] > 0
    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs2)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert abs(float(np.linalg.norm(env._state[6:10])) - 1.0) < 1e-6   # q stays unit


def test_pointed_prograde_burn_raises_orbit():
    env = Circularize3DEnv()
    env.reset(seed=1)
    r = 7000.0
    vc = orb.speed_circular(r)
    r_vec = np.array([r, 0.0, 0.0])
    v_vec = np.array([0.0, vc, 0.0])
    q = _q_point(v_vec)
    env._state = np.concatenate([r_vec, v_vec, q, np.zeros(3)])
    a0 = orb3.orbital_elements3d(r_vec, v_vec, env.cfg.mu).a
    for _ in range(10):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))  # zero torque, full throttle
    a1 = orb3.orbital_elements3d(env._state[0:3], env._state[3:6], env.cfg.mu).a
    assert a1 > a0                                   # prograde thrust raises the orbit


def test_torque_slews_attitude():
    env = Circularize3DEnv()
    env.reset(seed=2)
    tdir0 = env._thrust_dir().copy()
    for _ in range(5):
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # pure +x torque
    tdir1 = env._thrust_dir()
    assert float(np.linalg.norm(tdir1 - tdir0)) > 1e-3    # attitude actually changed
