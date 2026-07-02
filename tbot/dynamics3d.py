#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-D rigid-body spacecraft dynamics: orbit + attitude, integrated together.

State is a 13-vector ``[r(3), v(3), q(4), ω(3)]`` — inertial position [km] and
velocity [km/s], body→inertial attitude quaternion, and body angular velocity
[rad/s]. Thrust acts along a fixed body axis (default body +x), so the agent must
*slew the vehicle* (via torque, through the quaternion kinematics and Euler's
equations) to point the thrust before it can burn in a chosen direction.

Translational:  v̇ = −μ r/|r|³ + throttle · a_thrust · R(q)·b̂
Attitude:       q̇ = ½ q ⊗ [0, ω]
Rotational:     ω̇ = I⁻¹ (τ − ω × (I ω))            (Euler's equations)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from . import quaternion as quat
from .orbital import MU_EARTH

Vec = npt.NDArray[np.float64]


@dataclass
class Spacecraft:
    mu: float = MU_EARTH                       # [km^3/s^2]
    a_thrust: float = 5e-3                      # [km/s^2] full-throttle accel
    inertia: tuple[float, float, float] = (1.0, 1.0, 1.5)   # diagonal I (abstract units)
    max_torque: float = 2e-3                    # commanded torque scale
    thrust_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # body frame


def _deriv(state: Vec, torque: Vec, throttle: float, sc: Spacecraft) -> Vec:
    r = state[0:3]
    v = state[3:6]
    q = state[6:10]
    w = state[10:13]

    rmag = float(np.sqrt(float(r[0]) ** 2 + float(r[1]) ** 2 + float(r[2]) ** 2))
    grav: Vec = -sc.mu * r / rmag**3

    b_hat: Vec = np.array(sc.thrust_axis, dtype=np.float64)
    thrust_dir = quat.rotate(q, b_hat)         # body axis in inertial frame
    acc: Vec = grav + throttle * sc.a_thrust * thrust_dir

    qdot = quat.kinematics_deriv(q, w)

    # Euler's equations with diagonal inertia: ω̇ = I⁻¹(τ − ω × (Iω))
    ix, iy, iz = sc.inertia
    iw: Vec = np.array([ix * float(w[0]), iy * float(w[1]), iz * float(w[2])],
                       dtype=np.float64)
    gyro: Vec = np.array([                       # ω × (Iω)
        float(w[1]) * float(iw[2]) - float(w[2]) * float(iw[1]),
        float(w[2]) * float(iw[0]) - float(w[0]) * float(iw[2]),
        float(w[0]) * float(iw[1]) - float(w[1]) * float(iw[0]),
    ], dtype=np.float64)
    wdot: Vec = np.array([
        (float(torque[0]) - float(gyro[0])) / ix,
        (float(torque[1]) - float(gyro[1])) / iy,
        (float(torque[2]) - float(gyro[2])) / iz,
    ], dtype=np.float64)

    out: Vec = np.concatenate([v, acc, qdot, wdot])
    return out


def rk4_step(state: Vec, dt: float, torque: Vec, throttle: float,
             sc: Spacecraft) -> Vec:
    """One RK4 step of the coupled orbit+attitude dynamics; renormalizes q."""
    s: Vec = np.asarray(state, dtype=np.float64)
    k1 = _deriv(s, torque, throttle, sc)
    k2 = _deriv(s + 0.5 * dt * k1, torque, throttle, sc)
    k3 = _deriv(s + 0.5 * dt * k2, torque, throttle, sc)
    k4 = _deriv(s + dt * k3, torque, throttle, sc)
    out: Vec = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[6:10] = quat.normalize(out[6:10])      # keep attitude a unit quaternion
    return out
