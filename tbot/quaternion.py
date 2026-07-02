#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit-quaternion attitude math (Hamilton convention, scalar-first ``[w,x,y,z]``).

Attitude is represented as a unit quaternion mapping the body frame to the
inertial frame. No Euler angles (gimbal lock) and no accumulating rotation
matrices (drift). The quaternion double-cover (``q ≡ -q``) is handled by callers
feeding rotated body vectors to the policy rather than raw ``q``.

Kinematics: for body angular velocity ω, ``q̇ = ½ · q ⊗ [0, ω]``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]
Quat = npt.NDArray[np.float64]


def multiply(q1: Quat, q2: Quat) -> Quat:
    """Hamilton product ``q1 ⊗ q2`` (scalar-first)."""
    w1, x1, y1, z1 = (float(q1[i]) for i in range(4))
    w2, x2, y2, z2 = (float(q2[i]) for i in range(4))
    out: Quat = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)
    return out


def conjugate(q: Quat) -> Quat:
    out: Quat = np.array([float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])],
                         dtype=np.float64)
    return out


def normalize(q: Quat) -> Quat:
    n = float(np.sqrt(sum(float(q[i]) ** 2 for i in range(4))))
    if n < 1e-12:
        out: Quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return out
    out2: Quat = np.asarray(q, dtype=np.float64) / n
    return out2


def rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector ``v`` from body frame to inertial frame by ``q``."""
    qv: Quat = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)
    r = multiply(multiply(q, qv), conjugate(q))
    out: Vec3 = np.array([float(r[1]), float(r[2]), float(r[3])], dtype=np.float64)
    return out


def from_axis_angle(axis: Vec3, angle: float) -> Quat:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2))
    if n < 1e-12:
        out: Quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return out
    s = float(np.sin(angle / 2.0))
    out2: Quat = np.array([
        float(np.cos(angle / 2.0)),
        float(a[0]) / n * s, float(a[1]) / n * s, float(a[2]) / n * s,
    ], dtype=np.float64)
    return out2


def kinematics_deriv(q: Quat, omega: Vec3) -> Quat:
    """``q̇ = ½ · q ⊗ [0, ω]`` for body angular velocity ω."""
    omega_q: Quat = np.array([0.0, float(omega[0]), float(omega[1]), float(omega[2])],
                             dtype=np.float64)
    out: Quat = 0.5 * multiply(q, omega_q)
    return out
