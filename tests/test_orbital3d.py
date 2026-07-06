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


# --- combined altitude + plane-change baseline -----------------------------

R_E = 6378.137
LEO = R_E + 300.0     # 300 km circular
GEO = 42164.0


def test_combined_zero_inclination_is_hohmann():
    # with no plane change, every strategy must collapse to the pure Hohmann Δv
    d = orb3.combined_plane_altitude_dv(LEO, GEO, 0.0)
    hoh = orb.hohmann_dv(LEO, GEO)["total"]
    assert d["naive"] == pytest.approx(hoh, rel=1e-12)
    assert d["combined_apogee"] == pytest.approx(hoh, rel=1e-12)
    assert d["split_optimal"] == pytest.approx(hoh, rel=1e-9)


def test_combined_strategy_ordering():
    # cheapest strategy last: split ≤ combined-at-apogee ≤ naive-separate
    d = orb3.combined_plane_altitude_dv(LEO, GEO, np.radians(28.5))
    assert d["split_optimal"] <= d["combined_apogee"] + 1e-9
    assert d["combined_apogee"] <= d["naive"] + 1e-9
    # naive = Hohmann + a *separate* plane change at r2, by construction
    assert d["naive"] == pytest.approx(
        d["hohmann_total"] + d["plane_change_at_r2"], rel=1e-12)


def test_combined_28deg_leo_geo_margin():
    # the marquee case: combining beats the naive decomposition by ~21%
    d = orb3.combined_plane_altitude_dv(LEO, GEO, np.radians(28.5))
    assert d["naive"] == pytest.approx(5.406, abs=5e-3)
    assert d["combined_apogee"] == pytest.approx(4.256, abs=5e-3)
    margin = 1.0 - d["combined_apogee"] / d["naive"]
    assert margin == pytest.approx(0.213, abs=0.01)
    # Vallado's worked optimum puts ~2.2° of the 28.5° plane change at perigee
    assert np.degrees(d["split_frac"] * np.radians(28.5)) == pytest.approx(2.2, abs=0.5)


def test_combined_apogee_matches_vector_dv():
    # independent cross-check: the combined apogee burn is the vector difference
    # between the transfer-ellipse apogee velocity (inclined) and the target
    # circular velocity (equatorial), NOT the scalar sum of speed + plane change
    di = np.radians(28.5)
    a_t = 0.5 * (LEO + GEO)
    v_apo = orb.vis_viva(GEO, a_t)
    v_geo = orb.speed_circular(GEO)
    dv_apogee = np.sqrt(v_apo**2 + v_geo**2 - 2 * v_apo * v_geo * np.cos(di))
    dv_peri = abs(orb.vis_viva(LEO, a_t) - orb.speed_circular(LEO))
    d = orb3.combined_plane_altitude_dv(LEO, GEO, di)
    assert d["combined_apogee"] == pytest.approx(dv_peri + dv_apogee, rel=1e-12)
