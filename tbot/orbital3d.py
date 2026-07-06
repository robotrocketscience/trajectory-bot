#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-D orbital elements from a Cartesian state, plus 3-D maneuver Δv baselines.

Adds inclination and the angular-momentum vector to the planar elements. State
is inertial ``r`` [km] and ``v`` [km/s] (each a 3-vector).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .orbital import MU_EARTH, speed_circular

Vec = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Elements3D:
    r: float        # radius [km]
    v: float        # speed [km/s]
    energy: float   # specific orbital energy [km^2/s^2]
    a: float        # semi-major axis [km]
    e: float        # eccentricity [-]
    inc: float      # inclination [rad]
    h: float        # specific angular momentum magnitude [km^2/s]
    r_p: float      # periapsis radius [km]
    r_a: float      # apoapsis radius [km] (inf if e>=1)


def orbital_elements3d(r_vec: Vec, v_vec: Vec, mu: float = MU_EARTH) -> Elements3D:
    x, y, z = (float(r_vec[i]) for i in range(3))
    vx, vy, vz = (float(v_vec[i]) for i in range(3))
    r = float(np.sqrt(x * x + y * y + z * z))
    v = float(np.sqrt(vx * vx + vy * vy + vz * vz))
    energy = 0.5 * v * v - mu / r

    # angular momentum h = r x v
    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    h = float(np.sqrt(hx * hx + hy * hy + hz * hz))
    inc = float(np.arccos(max(-1.0, min(1.0, hz / h)))) if h > 0.0 else 0.0

    # eccentricity vector: ((v^2 - mu/r) r - (r.v) v) / mu
    rv = x * vx + y * vy + z * vz
    factor = v * v - mu / r
    ex = (factor * x - rv * vx) / mu
    ey = (factor * y - rv * vy) / mu
    ez = (factor * z - rv * vz) / mu
    e = float(np.sqrt(ex * ex + ey * ey + ez * ez))

    if abs(energy) < 1e-12:
        a = float("inf")
        r_p = (h * h / mu) / (1.0 + e) if e > 0.0 else r
        r_a = float("inf")
    else:
        a = -mu / (2.0 * energy)
        r_p = a * (1.0 - e)
        r_a = a * (1.0 + e) if e < 1.0 else float("inf")
    return Elements3D(r=r, v=v, energy=energy, a=a, e=e, inc=inc, h=h, r_p=r_p, r_a=r_a)


def plane_change_dv(v: float, delta_i: float) -> float:
    """Δv [km/s] for a pure inclination change ``delta_i`` [rad] at orbital speed ``v``.

    ``Δv = 2 v sin(Δi/2)`` — cheapest where v is small (high orbit), which is why
    plane changes are done at apoapsis / combined with a transfer's apogee burn.
    """
    return 2.0 * v * abs(np.sin(delta_i / 2.0))


def edelbaum_dv(r0: float, rf: float, delta_i: float, mu: float = MU_EARTH) -> float:
    """Edelbaum low-thrust Δv [km/s] for a combined altitude + plane change between
    circular orbits: ``ΔV = √(V₀² − 2 V₀ V_f cos(π/2·Δi) + V_f²)`` (Δi [rad]).

    The closed-form optimum for a constant-acceleration, quasi-circular, two-body
    spiral with a piecewise-constant yaw law (Edelbaum 1961). It is the correct
    baseline for the low-thrust regime — the yardstick a diff-sim low-thrust policy
    is judged against (and the analog of Hohmann/combined for impulsive maneuvers).

    Limits: Δi=0 → |V₀−V_f| (pure altitude); r0=rf → 2V·sin(π/4·Δi) (pure plane
    change, the π/2 continuous-turn penalty over impulsive 2V·sin(Δi/2) at small Δi).
    Anchor: LEO(≈7.73)→GEO(≈3.07) at 28.5° → ≈5.96 km/s (matches the literature).
    """
    v0 = speed_circular(r0, mu)
    vf = speed_circular(rf, mu)
    # arg = (V₀−V_f)² at Δi=0; clamp so float rounding can't make it slightly
    # negative → NaN (the pure-altitude edge case).
    arg = v0 * v0 - 2.0 * v0 * vf * np.cos(0.5 * np.pi * delta_i) + vf * vf
    return float(np.sqrt(max(arg, 0.0)))
