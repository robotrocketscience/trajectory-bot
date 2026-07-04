#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-D orbital-element and plane-change baseline tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import orbital as orb  # noqa: E402
from tbot import orbital3d as orb3  # noqa: E402


def test_equatorial_circular_orbit():
    r = 7000.0
    vc = orb.speed_circular(r)
    el = orb3.orbital_elements3d(np.array([r, 0.0, 0.0]), np.array([0.0, vc, 0.0]))
    assert el.e == pytest.approx(0.0, abs=1e-9)
    assert el.a == pytest.approx(r, rel=1e-9)
    assert el.inc == pytest.approx(0.0, abs=1e-9)     # motion in xy-plane


def test_polar_orbit_inclination():
    r = 7000.0
    vc = orb.speed_circular(r)
    # velocity in +z at position +x => orbit plane contains z -> inclination 90°
    el = orb3.orbital_elements3d(np.array([r, 0.0, 0.0]), np.array([0.0, 0.0, vc]))
    assert el.inc == pytest.approx(np.pi / 2, abs=1e-9)
    assert el.e == pytest.approx(0.0, abs=1e-9)


def test_inclined_orbit():
    r = 7000.0
    vc = orb.speed_circular(r)
    inc = np.radians(28.5)
    v = np.array([0.0, vc * np.cos(inc), vc * np.sin(inc)])
    el = orb3.orbital_elements3d(np.array([r, 0.0, 0.0]), v)
    assert el.inc == pytest.approx(inc, abs=1e-6)


def test_plane_change_dv_formula():
    v = 3.0
    di = np.radians(30.0)
    assert orb3.plane_change_dv(v, di) == pytest.approx(2 * v * np.sin(di / 2), rel=1e-12)
    # a 60° plane change at circular speed costs exactly the orbital speed (2 sin30 = 1)
    assert orb3.plane_change_dv(v, np.radians(60.0)) == pytest.approx(v, rel=1e-9)
