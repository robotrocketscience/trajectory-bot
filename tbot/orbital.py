#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planar two-body orbital mechanics: elements from state, and analytic Δv baselines.

Everything here is frame-agnostic given a gravitational parameter ``mu`` [km^3/s^2];
the RL env uses Earth. State vectors are planar ``[x, y, vx, vy]`` in km and km/s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

# Earth (the central body for the single-body milestones).
MU_EARTH: float = 398600.4418   # [km^3/s^2]
R_EARTH: float = 6378.137       # [km] equatorial radius


def vis_viva(r: float, a: float, mu: float = MU_EARTH) -> float:
    """Orbital speed [km/s] at radius ``r`` on an orbit of semi-major axis ``a``."""
    return float(np.sqrt(mu * (2.0 / r - 1.0 / a)))


def speed_circular(r: float, mu: float = MU_EARTH) -> float:
    """Circular-orbit speed [km/s] at radius ``r``."""
    return float(np.sqrt(mu / r))


@dataclass(frozen=True)
class Elements:
    """Planar orbital elements derived from a state vector."""

    r: float        # radius [km]
    v: float        # speed [km/s]
    energy: float   # specific orbital energy [km^2/s^2]
    h: float        # specific angular momentum (signed, +z) [km^2/s]
    a: float        # semi-major axis [km] (inf for parabolic)
    e: float        # eccentricity [-]
    r_p: float      # periapsis radius [km]
    r_a: float      # apoapsis radius [km] (inf if e>=1)


def orbital_elements(state: Vec, mu: float = MU_EARTH) -> Elements:
    """Compute planar elements from ``state = [x, y, vx, vy]``."""
    x = float(state[0])
    y = float(state[1])
    vx = float(state[2])
    vy = float(state[3])

    r = float(np.hypot(x, y))
    v = float(np.hypot(vx, vy))
    energy = 0.5 * v * v - mu / r
    h = x * vy - y * vx                       # z-component of r x v (planar)
    rv = x * vx + y * vy                      # r . v

    # Eccentricity vector components: ((v^2 - mu/r) r - (r.v) v) / mu
    factor = v * v - mu / r
    e_x = (factor * x - rv * vx) / mu
    e_y = (factor * y - rv * vy) / mu
    e = float(np.hypot(e_x, e_y))

    if abs(energy) < 1e-12:                    # ~parabolic
        a = float("inf")
        r_p = (h * h / mu) / (1.0 + e) if e > 0.0 else r
        r_a = float("inf")
    else:
        a = -mu / (2.0 * energy)
        r_p = a * (1.0 - e)
        r_a = a * (1.0 + e) if e < 1.0 else float("inf")
    return Elements(r=r, v=v, energy=energy, h=h, a=a, e=e, r_p=r_p, r_a=r_a)


# --------------------------------------------------------------------------
# Analytic Δv baselines (impulsive) — the numbers the RL agent must match/beat.
# --------------------------------------------------------------------------
def circularize_apoapsis_dv(r_p: float, r_a: float, mu: float = MU_EARTH) -> float:
    """Single-impulse Δv [km/s] to circularize at apoapsis of an ellipse (r_p, r_a).

    The theoretically optimal maneuver for "circularize from an elliptical orbit"
    is one prograde burn at apoapsis: raise the speed from the elliptical
    apoapsis speed to the local circular speed. This is the baseline for the
    circularize-2D milestone.
    """
    a_ellipse = 0.5 * (r_p + r_a)
    v_apo = vis_viva(r_a, a_ellipse, mu)       # speed at apoapsis on the ellipse
    v_circ = speed_circular(r_a, mu)           # target circular speed at r_a
    return abs(v_circ - v_apo)


def hohmann_dv(r1: float, r2: float, mu: float = MU_EARTH) -> dict[str, float]:
    """Two-burn Hohmann transfer Δv [km/s] between circular orbits r1 -> r2."""
    a_t = 0.5 * (r1 + r2)
    v1 = speed_circular(r1, mu)
    v2 = speed_circular(r2, mu)
    v_peri = vis_viva(r1, a_t, mu)
    v_apo = vis_viva(r2, a_t, mu)
    dv1 = abs(v_peri - v1)
    dv2 = abs(v2 - v_apo)
    return {"dv1": dv1, "dv2": dv2, "total": dv1 + dv2}


def bielliptic_dv(r1: float, r2: float, r_b: float,
                  mu: float = MU_EARTH) -> dict[str, float]:
    """Three-burn bi-elliptic transfer Δv [km/s] r1 -> r2 via apoapsis r_b (> r2)."""
    a1 = 0.5 * (r1 + r_b)
    a2 = 0.5 * (r2 + r_b)
    v1 = speed_circular(r1, mu)
    dv1 = abs(vis_viva(r1, a1, mu) - v1)
    dv2 = abs(vis_viva(r_b, a2, mu) - vis_viva(r_b, a1, mu))
    v2 = speed_circular(r2, mu)
    dv3 = abs(vis_viva(r2, a2, mu) - v2)
    return {"dv1": dv1, "dv2": dv2, "dv3": dv3, "total": dv1 + dv2 + dv3}
