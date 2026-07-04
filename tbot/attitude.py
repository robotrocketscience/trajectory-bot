#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic attitude layer: point-to-direction control + orbit-frame axes.

This sits *below* the RL policy. The policy decides a desired thrust direction
(in the orbit frame) and a throttle; this module turns "point the thrust axis at
direction d" into the body angular-rate command that slews the vehicle there. The
policy therefore never commands rotation directly — it only makes the decision,
and this deterministic controller effects it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import quaternion as quat

Vec = npt.NDArray[np.float64]


def _unit(v: Vec) -> Vec:
    n = float(np.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    out: Vec = np.asarray(v, dtype=np.float64) / n
    return out


def _cross(a: Vec, b: Vec) -> Vec:
    out: Vec = np.array([
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ], dtype=np.float64)
    return out


def orbit_frame(r_vec: Vec, v_vec: Vec) -> tuple[Vec, Vec, Vec]:
    """Orthonormal orbit frame: (prograde t̂, orbit-normal ŵ, t̂×ŵ ŝ).

    A desired thrust direction ``c_t·t̂ + c_w·ŵ + c_s·ŝ`` covers all maneuvers:
    prograde/retrograde (±t̂), plane change (±ŵ), radial (ŝ).
    """
    t_hat = _unit(v_vec)
    w_hat = _unit(_cross(r_vec, v_vec))
    s_hat = _cross(t_hat, w_hat)
    return t_hat, w_hat, s_hat


def desired_direction(r_vec: Vec, v_vec: Vec, coeffs: Vec) -> Vec:
    """Inertial unit vector for orbit-frame coefficients ``[c_t, c_w, c_s]``."""
    t_hat, w_hat, s_hat = orbit_frame(r_vec, v_vec)
    d: Vec = coeffs[0] * t_hat + coeffs[1] * w_hat + coeffs[2] * s_hat
    return _unit(d)


def point_rate_command(q: Vec, desired_dir: Vec, thrust_axis: Vec,
                       k_p: float, max_rate: float) -> Vec:
    """Body angular-rate command to slew the thrust axis toward ``desired_dir``.

    Proportional pointing law: the rate is along the (body-frame) error axis
    ``b × d`` with magnitude ∝ sin(angle), capped at ``max_rate``.
    """
    b_in = quat.rotate(q, thrust_axis)          # current thrust axis, inertial
    d = _unit(desired_dir)
    err_in = _cross(b_in, d)                     # inertial rotation axis · sin θ
    err_body = quat.rotate(quat.conjugate(q), err_in)   # express in body frame
    omega: Vec = k_p * err_body
    n = float(np.sqrt(float(omega[0]) ** 2 + float(omega[1]) ** 2
                      + float(omega[2]) ** 2))
    if n > max_rate:
        omega = omega * (max_rate / n)
    return omega
