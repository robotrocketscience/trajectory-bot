#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-D coupled orbit+attitude dynamics tests."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import dynamics3d as dyn  # noqa: E402
from tbot import orbital as orb  # noqa: E402
from tbot import orbital3d as orb3  # noqa: E402

IDENT_Q = np.array([1.0, 0.0, 0.0, 0.0])
NO_TORQUE = np.zeros(3)


def _state(r_vec, v_vec, q=IDENT_Q, w=np.zeros(3)):
    return np.concatenate([r_vec, v_vec, q, w])


def test_circular_orbit_closes_and_conserves_energy():
    r = 7000.0
    vc = orb.speed_circular(r)
    s = _state(np.array([r, 0.0, 0.0]), np.array([0.0, vc, 0.0]))
    sc = dyn.Spacecraft()
    period = 2 * np.pi * np.sqrt(r**3 / orb.MU_EARTH)
    dt = 1.0
    n = int(round(period / dt))
    for _ in range(n):
        s = dyn.rk4_step(s, dt, NO_TORQUE, 0.0, sc)
    np.testing.assert_allclose(s[0:3], [r, 0.0, 0.0], atol=5.0)
    e0 = orb3.orbital_elements3d(np.array([r, 0.0, 0.0]), np.array([0.0, vc, 0.0])).energy
    e1 = orb3.orbital_elements3d(s[0:3], s[3:6]).energy
    assert abs(e1 - e0) / abs(e0) < 1e-6


def test_quaternion_stays_unit():
    s = _state(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]),
               w=np.array([0.01, 0.0, 0.02]))
    sc = dyn.Spacecraft()
    for _ in range(500):
        s = dyn.rk4_step(s, 1.0, NO_TORQUE, 0.0, sc)
        assert abs(float(np.linalg.norm(s[6:10])) - 1.0) < 1e-6


def test_rate_tracking_reaches_commanded_rate():
    # angular velocity should track the commanded body rate (first-order response)
    s = _state(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0]),
               w=np.zeros(3))
    sc = dyn.Spacecraft()
    cmd = np.array([0.0, 0.0, 0.04])
    for _ in range(300):
        s = dyn.rk4_step(s, 1.0, cmd, 0.0, sc)     # 300 s >> 1/rate_gain = 10 s
    np.testing.assert_allclose(s[10:13], cmd, atol=2e-3)


def test_thrust_along_body_axis_accelerates_prograde():
    # attitude = identity => body +x points along inertial +x; thrust adds +x accel
    r = 7000.0
    vc = orb.speed_circular(r)
    s = _state(np.array([0.0, r, 0.0]), np.array([-vc, 0.0, 0.0]))  # at +y, moving -x
    sc = dyn.Spacecraft()
    a0 = orb3.orbital_elements3d(s[0:3], s[3:6]).a
    for _ in range(20):
        s = dyn.rk4_step(s, 1.0, NO_TORQUE, 1.0, sc)   # full throttle, body +x
    # thrust in +x while velocity is -x is retrograde here -> lowers energy/a.
    assert orb3.orbital_elements3d(s[0:3], s[3:6]).a < a0
