#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-D rigid-body spacecraft dynamics: orbit + attitude, integrated together.

State is a 13-vector ``[r(3), v(3), q(4), omega(3)]`` -- inertial position [km]
and velocity [km/s], body->inertial attitude quaternion, and body angular
velocity [rad/s]. Thrust acts along a fixed body axis (default body +x), so the
agent must slew the vehicle to point the thrust before it can burn.

Attitude uses **bounded angular-rate control with a first-order response**: the
agent commands a body rate ``omega_cmd`` (the low-level ADCS torque loop is
abstracted), the vehicle's rate tracks it with time constant ``1/rate_gain``, and
the quaternion integrates that rate. This is unconditionally stable at the orbital
step size -- unlike raw torque + Euler's equations, whose stiff rotational
dynamics blow up when integrated at dt ~ 10 s -- while still requiring the agent
to point the vehicle.

  v_dot     = -mu r/|r|^3 + throttle * a_thrust * R(q) b_hat
  q_dot     = 1/2 q (x) [0, omega]
  omega_dot = rate_gain * (omega_cmd - omega)
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
    rate_gain: float = 0.1                      # [1/s] rate-tracking gain (k*dt < 2.8)
    thrust_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)  # body frame


def _deriv(state: Vec, omega_cmd: Vec, throttle: float, sc: Spacecraft) -> Vec:
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
    wdot: Vec = sc.rate_gain * (np.asarray(omega_cmd, dtype=np.float64) - w)

    out: Vec = np.concatenate([v, acc, qdot, wdot])
    return out


def rk4_step(state: Vec, dt: float, omega_cmd: Vec, throttle: float,
             sc: Spacecraft) -> Vec:
    """One RK4 step of the coupled orbit+attitude dynamics; renormalizes q.

    ``omega_cmd`` is the commanded body angular rate [rad/s].
    """
    s: Vec = np.asarray(state, dtype=np.float64)
    k1 = _deriv(s, omega_cmd, throttle, sc)
    k2 = _deriv(s + 0.5 * dt * k1, omega_cmd, throttle, sc)
    k3 = _deriv(s + 0.5 * dt * k2, omega_cmd, throttle, sc)
    k4 = _deriv(s + dt * k3, omega_cmd, throttle, sc)
    out: Vec = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    out[6:10] = quat.normalize(out[6:10])      # keep attitude a unit quaternion
    return out
