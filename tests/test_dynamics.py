#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Propagation tests: a coasting orbit conserves energy and closes after one period."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import dynamics as dyn  # noqa: E402
from tbot import orbital as orb  # noqa: E402


def test_circular_orbit_closes_after_one_period():
    r = 7000.0
    v = orb.speed_circular(r)
    state = np.array([r, 0.0, 0.0, v])
    period = 2.0 * np.pi * np.sqrt(r**3 / orb.MU_EARTH)

    dt = 1.0
    n = int(round(period / dt))
    path = dyn.propagate(state, dt, n)
    final = path[-1]

    # Position closes on itself to within a few km after a full RK4 period.
    assert np.linalg.norm(final[:2] - state[:2]) < 5.0

    e0 = orb.orbital_elements(state).energy
    e1 = orb.orbital_elements(final).energy
    assert abs(e1 - e0) / abs(e0) < 1e-6          # energy drift negligible


def test_thrust_raises_orbit():
    # A prograde burn increases speed and raises the orbit's apoapsis.
    r = 7000.0
    v = orb.speed_circular(r)
    state = np.array([r, 0.0, 0.0, v])
    a0 = orb.orbital_elements(state).a

    thrust = np.array([0.0, 1e-3])                # +y (prograde here) accel
    s = state.copy()
    for _ in range(50):
        s = dyn.rk4_step(s, 1.0, thrust)
    assert orb.orbital_elements(s).a > a0


def test_gravity_accel_points_inward():
    r_vec = np.array([7000.0, 0.0])
    acc = dyn.gravity_accel(r_vec)
    assert acc[0] < 0.0 and abs(acc[1]) < 1e-12
    assert acc[0] == pytest.approx(-orb.MU_EARTH / 7000.0**2, rel=1e-9)
