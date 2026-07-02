#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false, reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
"""Untyped-library boundary: the single JPL Horizons network call.

astropy / astroquery ship only partial type information, so pyright cannot see
through `Horizons(...).vectors()` or `Time(...).iso` without emitting
"unknown type" errors. Rather than weaken strict mode for the whole project, the
one function that actually touches those libraries is isolated here with a
file-scoped relaxation of the four unavoidable "unknown" rules. Everything this
returns is coerced to concrete `NDArray[np.float64]`, so callers stay fully
strict-typed.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

_KM_PER_AU: float = 1.495978707e8              # [km]
_KMS_PER_AU_PER_DAY: float = _KM_PER_AU / 86400.0   # [km/s]


def horizons_vectors(body_id: str, start: str, stop: str, step_s: int,
                     location: str) -> tuple[Vec, Vec, Vec]:
    """One bulk JPL Horizons vectors query. Returns (times_jd, r_km, v_kms)."""
    from astropy.table import Table
    from astropy.time import Time
    from astroquery.jplhorizons import Horizons

    start_t = Time(start, format="isot", scale="utc")
    stop_t = Time(stop, format="isot", scale="utc")
    total_s = float((stop_t - start_t).to_value("s"))
    n_intervals = max(1, int(round(total_s / step_s)))

    obj = Horizons(
        id=body_id, location=location,
        epochs={"start": start_t.iso, "stop": stop_t.iso, "step": str(n_intervals)},
    )
    vec = Table(obj.vectors())
    times_jd: Vec = np.asarray(vec["datetime_jd"], dtype=np.float64)
    r: Vec = np.column_stack(
        [np.asarray(vec[c], dtype=np.float64) for c in ("x", "y", "z")]
    ) * _KM_PER_AU
    v: Vec = np.column_stack(
        [np.asarray(vec[c], dtype=np.float64) for c in ("vx", "vy", "vz")]
    ) * _KMS_PER_AU_PER_DAY
    return times_jd, r, v
