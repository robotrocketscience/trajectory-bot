#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry of real spacecraft missions for the historical-trajectory benchmark.

Each entry is a JPL Horizons NAIF ID plus the cruise window. Fetched via the
cached `ephemeris.py` (spacecraft are just another Horizons id). See
`docs/BENCHMARKS.md` for methodology and caveats. All IDs verified against Horizons.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mission:
    naif_id: str
    name: str
    transfer: str
    launch: str        # ISO date
    arrival: str       # ISO date
    gravity_assist: bool


MISSIONS: dict[str, Mission] = {
    # Earth->Mars direct cruises (clean, no gravity assists) — fair Δv targets.
    "maven": Mission("-202", "MAVEN", "Earth->Mars", "2013-11-18", "2014-09-22", False),
    "mro": Mission("-74", "Mars Reconnaissance Orbiter", "Earth->Mars", "2005-08-12", "2006-03-10", False),
    "odyssey": Mission("-53", "Mars Odyssey", "Earth->Mars", "2001-04-07", "2001-10-24", False),
    "msl": Mission("-76", "Curiosity / MSL", "Earth->Mars", "2011-11-26", "2012-08-06", False),
    "perseverance": Mission("-168", "Perseverance / Mars 2020", "Earth->Mars", "2020-07-30", "2021-02-18", False),
    "tgo": Mission("-143", "ExoMars TGO", "Earth->Mars", "2016-03-14", "2016-10-19", False),
    # Gravity-assist tours — NOT fair Δv targets; assist-discovery exploration only.
    "voyager2": Mission("-32", "Voyager 2", "outer-planet tour", "1977-08-20", "1989-08-25", True),
    "juno": Mission("-61", "Juno", "Earth->Jupiter (EGA)", "2011-08-05", "2016-07-05", True),
    "new_horizons": Mission("-98", "New Horizons", "Jupiter assist -> Pluto", "2006-01-19", "2015-07-14", True),
}


if __name__ == "__main__":
    for key, m in MISSIONS.items():
        ga = " [gravity assist]" if m.gravity_assist else ""
        print(f"{key:14} {m.naif_id:>5}  {m.name:28} {m.launch}..{m.arrival}{ga}")
