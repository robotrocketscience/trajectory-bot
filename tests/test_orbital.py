#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orbital-mechanics unit tests: elements from state, and analytic Δv baselines."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import orbital as orb  # noqa: E402


def _circular_state(r, mu=orb.MU_EARTH):
    v = orb.speed_circular(r, mu)
    return np.array([r, 0.0, 0.0, v])


def test_circular_orbit_elements():
    r = 7000.0
    el = orb.orbital_elements(_circular_state(r))
    assert el.e == pytest.approx(0.0, abs=1e-9)
    assert el.a == pytest.approx(r, rel=1e-9)
    assert el.r_p == pytest.approx(r, rel=1e-6)
    assert el.r_a == pytest.approx(r, rel=1e-6)


def test_elliptical_elements_roundtrip():
    r_p, r_a = 7000.0, 14000.0
    a = 0.5 * (r_p + r_a)
    v_p = orb.vis_viva(r_p, a)
    # at periapsis: position +x, velocity +y (prograde)
    state = np.array([r_p, 0.0, 0.0, v_p])
    el = orb.orbital_elements(state)
    assert el.a == pytest.approx(a, rel=1e-9)
    assert el.r_p == pytest.approx(r_p, rel=1e-6)
    assert el.r_a == pytest.approx(r_a, rel=1e-6)
    assert el.e == pytest.approx((r_a - r_p) / (r_a + r_p), rel=1e-6)


def test_circularize_apoapsis_dv_matches_manual():
    r_p, r_a = 7000.0, 14000.0
    a = 0.5 * (r_p + r_a)
    dv = orb.circularize_apoapsis_dv(r_p, r_a)
    expected = orb.speed_circular(r_a) - orb.vis_viva(r_a, a)
    assert dv == pytest.approx(expected, rel=1e-12)
    assert dv > 0.0


def test_hohmann_leo_to_geo():
    r_leo = orb.R_EARTH + 300.0
    r_geo = 42164.0
    dv = orb.hohmann_dv(r_leo, r_geo)
    # Textbook LEO(300km)->GEO Hohmann total is ~3.9 km/s.
    assert dv["total"] == pytest.approx(3.9, abs=0.15)
    assert dv["dv1"] > dv["dv2"]


def test_bielliptic_can_beat_hohmann_at_high_ratio():
    # For r2/r1 beyond ~11.94, bi-elliptic (large r_b) can beat Hohmann.
    r1 = 7000.0
    r2 = r1 * 15.0
    r_b = r1 * 60.0
    h = orb.hohmann_dv(r1, r2)["total"]
    be = orb.bielliptic_dv(r1, r2, r_b)["total"]
    assert be < h
