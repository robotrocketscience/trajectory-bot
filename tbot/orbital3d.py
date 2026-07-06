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

from .orbital import (
    MU_EARTH,
    circularize_apoapsis_dv,
    hohmann_dv,
    speed_circular,
    vis_viva,
)

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


def combined_circularize_plane_dv(
    r_p: float, r_a: float, delta_i: float, mu: float = MU_EARTH,
) -> dict[str, float]:
    """Δv [km/s] baselines for a single-burn combined circularize + plane change.

    An inclined ellipse ``(r_p, r_a)`` at inclination ``delta_i`` [rad] →
    circular *and equatorial* at ``r_a``, in one apoapsis burn. The single-burn
    analogue of :func:`combined_plane_altitude_dv`; both yardsticks last:

    - ``naive``: circularize at apoapsis (prograde) and then do a *separate*
      plane change at the resulting circular speed — the from-the-textbook
      decomposition, the number the agent is expected to beat.
    - ``combined``: one vector burn at apoapsis that both circularizes and
      removes the inclination. By the triangle inequality this can never cost
      more than ``naive`` (the direct velocity change ≤ any two-leg route), so
      the margin is guaranteed and grows with ``delta_i``.

    Both collapse to :func:`~tbot.orbital.circularize_apoapsis_dv` at
    ``delta_i = 0``. Over-raising apoapsis to cheapen the plane change
    (a bi-elliptic-with-plane-change idea) can beat ``combined`` — a policy that
    does so is genuine discovery, not bookkeeping.
    """
    a_ell = 0.5 * (r_p + r_a)
    v_apo = vis_viva(r_a, a_ell, mu)          # elliptical apoapsis speed (inclined)
    v_circ = speed_circular(r_a, mu)          # target circular speed (equatorial)
    naive = abs(v_circ - v_apo) + 2.0 * v_circ * abs(np.sin(delta_i / 2.0))
    combined = float(
        np.sqrt(v_apo * v_apo + v_circ * v_circ
                - 2.0 * v_apo * v_circ * np.cos(delta_i)))
    return {
        "naive": naive,
        "combined": combined,
        "circularize": circularize_apoapsis_dv(r_p, r_a, mu),
        "plane_change_at_r_a": 2.0 * v_circ * abs(np.sin(delta_i / 2.0)),
    }


def combined_plane_altitude_dv(
    r1: float, r2: float, delta_i: float, mu: float = MU_EARTH,
) -> dict[str, float]:
    """Δv [km/s] baselines for a combined altitude-raise + plane-change transfer.

    Circular ``r1`` (inclined ``delta_i`` [rad]) → circular ``r2`` (equatorial),
    via a Hohmann transfer ellipse. Three yardsticks the diff-sim agent is judged
    against, cheapest strategy last:

    - ``naive``: a competent-but-naive practitioner — Hohmann in-plane, then a
      *separate* plane change at ``r2`` where the circular speed is lowest.
      This is the number a from-the-textbook decomposition gives, and the one
      the agent is expected to *beat*.
    - ``combined_apogee``: fold the whole plane change into the apogee
      circularization burn (vector Δv, not scalar sum). The standard textbook
      "combined maneuver" — cheaper because it never pays the plane change and
      the speed change separately.
    - ``split_optimal``: the true two-impulse optimum — split the plane change
      between the perigee and apogee burns (a small fraction at perigee), found
      by golden-section on the split. Beats ``combined_apogee`` slightly.

    All three share the same Hohmann transfer ellipse (a = (r1+r2)/2). Valid for
    ``r2/r1 < 11.94`` where the in-plane optimum is genuinely Hohmann (above that,
    a bi-elliptic base would beat it and this baseline understates the optimum).

    Note the split optimum still fixes the transfer apoapsis at ``r2``;
    over-raising apoapsis bi-elliptic-style can cheapen the plane change further,
    so a policy beating ``split_optimal`` is genuine discovery, not a bookkeeping
    artifact.
    """
    a_t = 0.5 * (r1 + r2)
    v1 = speed_circular(r1, mu)          # circular speed at r1 (in inclined plane)
    v2 = speed_circular(r2, mu)          # circular speed at r2 (equatorial target)
    v_peri = vis_viva(r1, a_t, mu)       # transfer-ellipse speed at perigee
    v_apo = vis_viva(r2, a_t, mu)        # transfer-ellipse speed at apogee

    hoh = hohmann_dv(r1, r2, mu)
    naive = hoh["total"] + 2.0 * v2 * abs(np.sin(delta_i / 2.0))

    def combined(va: float, vb: float, di: float) -> float:
        # magnitude of the vector Δv that changes speed va→vb and rotates by di
        return float(np.sqrt(va * va + vb * vb - 2.0 * va * vb * np.cos(di)))

    dv1_in = abs(v_peri - v1)                     # in-plane perigee raise
    combined_apogee = dv1_in + combined(v_apo, v2, delta_i)

    def split_total(frac: float) -> float:
        # frac of the plane change absorbed at perigee, the rest at apogee
        dv1 = combined(v1, v_peri, frac * delta_i)
        dv2 = combined(v_apo, v2, (1.0 - frac) * delta_i)
        return dv1 + dv2

    # golden-section minimize split_total over frac ∈ [0, 1]
    gr = 0.5 * (np.sqrt(5.0) - 1.0)
    lo, hi = 0.0, 1.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = split_total(c), split_total(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = split_total(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = split_total(d)
    split_frac = 0.5 * (lo + hi)
    split_optimal = split_total(split_frac)

    return {
        "naive": naive,
        "combined_apogee": combined_apogee,
        "split_optimal": split_optimal,
        "split_frac": split_frac,
        "hohmann_total": hoh["total"],
        "plane_change_at_r2": 2.0 * v2 * abs(np.sin(delta_i / 2.0)),
    }


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
