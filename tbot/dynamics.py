#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planar two-body propagation with RK4 and a constant thrust acceleration.

State is ``[x, y, vx, vy]`` (km, km/s). Thrust is modelled as a constant
acceleration ``[ax, ay]`` [km/s^2] held over the step (zero-order hold), which is
how the RL agent's per-step action is applied.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .orbital import MU_EARTH

Vec = npt.NDArray[np.float64]


def gravity_accel(r_vec: Vec, mu: float = MU_EARTH) -> Vec:
    """Two-body gravitational acceleration ``-mu r / |r|^3`` [km/s^2]."""
    r = float(np.hypot(float(r_vec[0]), float(r_vec[1])))
    out: Vec = np.array(
        [-mu * float(r_vec[0]) / r**3, -mu * float(r_vec[1]) / r**3],
        dtype=np.float64,
    )
    return out


def _deriv(state: Vec, thrust_acc: Vec, mu: float) -> Vec:
    ax = -mu * float(state[0]) / float(np.hypot(float(state[0]), float(state[1]))) ** 3
    ay = -mu * float(state[1]) / float(np.hypot(float(state[0]), float(state[1]))) ** 3
    out: Vec = np.array(
        [float(state[2]), float(state[3]),
         ax + float(thrust_acc[0]), ay + float(thrust_acc[1])],
        dtype=np.float64,
    )
    return out


def rk4_step(state: Vec, dt: float,
             thrust_acc: Vec | None = None,
             mu: float = MU_EARTH) -> Vec:
    """Advance one RK4 step of ``dt`` seconds under gravity + constant thrust."""
    ta: Vec = np.zeros(2, dtype=np.float64) if thrust_acc is None else thrust_acc
    s: Vec = np.asarray(state, dtype=np.float64)
    k1 = _deriv(s, ta, mu)
    k2 = _deriv(s + 0.5 * dt * k1, ta, mu)
    k3 = _deriv(s + 0.5 * dt * k2, ta, mu)
    k4 = _deriv(s + dt * k3, ta, mu)
    out: Vec = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out


def propagate(state: Vec, dt: float, n_steps: int, mu: float = MU_EARTH) -> Vec:
    """Coast (no thrust) for ``n_steps`` of ``dt``; return the full [n_steps+1, 4] path."""
    s: Vec = np.asarray(state, dtype=np.float64)
    out: Vec = np.empty((n_steps + 1, 4), dtype=np.float64)
    out[0] = s
    for i in range(n_steps):
        s = rk4_step(s, dt, None, mu)
        out[i + 1] = s
    return out
