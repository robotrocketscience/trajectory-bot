#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quaternion attitude math tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tbot import quaternion as q  # noqa: E402

IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def test_identity_rotation():
    v = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(q.rotate(IDENT, v), v, atol=1e-12)


def test_rotate_x_about_z_90deg():
    qz = q.from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(q.rotate(qz, np.array([1.0, 0.0, 0.0])),
                               [0.0, 1.0, 0.0], atol=1e-9)


def test_multiply_identity():
    qz = q.from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.7)
    np.testing.assert_allclose(q.multiply(IDENT, qz), qz, atol=1e-12)
    np.testing.assert_allclose(q.multiply(qz, IDENT), qz, atol=1e-12)


def test_rotate_then_inverse_roundtrip():
    qz = q.from_axis_angle(np.array([1.0, 2.0, 3.0]), 1.1)
    v = np.array([0.5, -1.0, 2.0])
    rotated = q.rotate(qz, v)
    back = q.rotate(q.conjugate(qz), rotated)
    np.testing.assert_allclose(back, v, atol=1e-9)


def test_normalize_and_norm_preserved_under_multiply():
    a = q.normalize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert float(np.linalg.norm(a)) == pytest.approx(1.0, abs=1e-12)
    b = q.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.3)
    prod = q.multiply(a, b)
    assert float(np.linalg.norm(prod)) == pytest.approx(1.0, abs=1e-9)


def test_kinematics_integrates_to_rotation():
    # constant body rate about z: after time T the attitude should have rotated
    # ~ omega*T about z. Integrate q̇ = ½ q ⊗ [0,ω] with small steps.
    omega = np.array([0.0, 0.0, 0.5])
    qq = IDENT.copy()
    dt = 1e-3
    for _ in range(1000):                          # T = 1 s -> angle 0.5 rad
        qq = q.normalize(qq + dt * q.kinematics_deriv(qq, omega))
    # body x-axis should now point at angle 0.5 rad in the xy-plane
    x_in = q.rotate(qq, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(x_in, [np.cos(0.5), np.sin(0.5), 0.0], atol=1e-3)
