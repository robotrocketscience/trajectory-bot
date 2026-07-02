#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attitude layer: orbit frame + pointing controller convergence."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import attitude  # noqa: E402
from tbot import dynamics3d as dyn  # noqa: E402
from tbot import quaternion as quat  # noqa: E402


def test_orbit_frame_orthonormal():
    r = np.array([7000.0, 1000.0, -500.0])
    v = np.array([1.0, 7.0, 2.0])
    t, w, s = attitude.orbit_frame(r, v)
    for u in (t, w, s):
        assert float(np.linalg.norm(u)) == pytest_approx(1.0)
    assert abs(float(np.dot(t, w))) < 1e-9
    assert abs(float(np.dot(t, s))) < 1e-9
    assert abs(float(np.dot(w, s))) < 1e-9


def test_desired_direction_selects_axes():
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    t, w, s = attitude.orbit_frame(r, v)
    np.testing.assert_allclose(attitude.desired_direction(r, v, np.array([1.0, 0, 0])), t, atol=1e-9)
    np.testing.assert_allclose(attitude.desired_direction(r, v, np.array([0, 1.0, 0])), w, atol=1e-9)


def test_pointing_controller_converges():
    # start with an arbitrary attitude; the controller should slew the thrust axis
    # onto a fixed desired direction over time.
    sc = dyn.Spacecraft()
    tax = np.array(sc.thrust_axis)
    desired = np.array([0.0, 1.0, 0.0])
    state = np.concatenate([[7000.0, 0, 0], [0, 7.5, 0],
                            quat.from_axis_angle(np.array([1.0, 1, 1]), 2.0), np.zeros(3)])
    for _ in range(400):
        omega = attitude.point_rate_command(state[6:10], desired, tax, k_p=0.5, max_rate=0.05)
        state = dyn.rk4_step(state, 1.0, omega, 0.0, sc)
    b_in = quat.rotate(state[6:10], tax)
    assert float(np.dot(b_in, desired)) > 0.999    # aligned to within ~2.5 deg


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-9)
