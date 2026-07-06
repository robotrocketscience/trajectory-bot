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


# --- Edelbaum low-thrust baseline ------------------------------------------

R_E = 6378.137
LEO = R_E + 300.0
GEO = 42164.0


def test_edelbaum_pure_altitude_is_speed_difference():
    # Δi=0 → |V0 - Vf|, the pure low-thrust altitude change
    dv = orb3.edelbaum_dv(LEO, GEO, 0.0)
    assert dv == pytest.approx(abs(orb.speed_circular(LEO) - orb.speed_circular(GEO)), rel=1e-12)


def test_edelbaum_pure_plane_change():
    # same radius → 2V·sin(π/4·Δi); the continuous-turn form
    r = 7000.0; V = orb.speed_circular(r); di = np.radians(30.0)
    assert orb3.edelbaum_dv(r, r, di) == pytest.approx(2 * V * np.sin(np.pi / 4 * di), rel=1e-9)


def test_edelbaum_leo_geo_28deg_anchor():
    # literature anchor: LEO→GEO with 28.5° plane change ≈ 5.96 km/s
    dv = orb3.edelbaum_dv(LEO, GEO, np.radians(28.5))
    assert dv == pytest.approx(5.96, abs=0.05)
    # coplanar low-thrust LEO→GEO ≈ 4.66 km/s
    assert orb3.edelbaum_dv(LEO, GEO, 0.0) == pytest.approx(4.66, abs=0.05)


def test_edelbaum_monotonic_in_plane_change():
    dvs = [orb3.edelbaum_dv(LEO, GEO, np.radians(d)) for d in (0, 10, 20, 30, 40)]
    assert all(b > a for a, b in zip(dvs, dvs[1:]))
    # continuous small plane change costs π/2× the impulsive one (known penalty)
    r = 20000.0; V = orb.speed_circular(r); di = np.radians(1.0)
    assert orb3.edelbaum_dv(r, r, di) / orb3.plane_change_dv(V, di) == pytest.approx(np.pi / 2, abs=0.02)
