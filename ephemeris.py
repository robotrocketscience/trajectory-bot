#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bulk, on-disk-cached, idempotent JPL Horizons ephemeris access.

The original TrajectoryBot code queried JPL Horizons on every environment reset
and on every hourly (sim-time) ephemeris refresh — 10 separate HTTP requests per
refresh, repeated across thousands of steps and episodes. That spams the Horizons
service and re-downloads identical data.

This module replaces that with a single **bulk** query per body covering the
entire requested time span, cached to disk and **idempotent**: a repeated request
for the same (or an already-covered) span is served from disk and makes zero
network calls. A JSON manifest indexes cached spans so overlap is deduplicated
across runs.

Design goals (per project rule: batch, cache, idempotency):
  - One network call per unique (body, start, stop, step) span. Never per-step.
  - Cache-hit path is pure disk I/O — no ``astroquery``/network import needed.
  - Concurrent-safe writes (temp file + atomic rename); a half-written cache
    entry is never observed.
  - The single network-touching function (``_horizons_vectors``) is isolated so
    tests can monkeypatch it and assert call counts.

Units: positions returned in km, velocities in km/s, times as astropy Time.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]

DEFAULT_CACHE_DIR: str = os.environ.get(
    "TBOT_EPHEM_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ephem_cache"),
)


@dataclass(frozen=True)
class SpanSpec:
    """Immutable description of a requested ephemeris span.

    body_id: JPL Horizons body id string (e.g. '399' for Earth).
    start, stop: ISO-8601 UTC strings (astropy Time .isot), inclusive endpoints.
    step_s: sample spacing in seconds. The span is sampled at
            start, start+step, ... up to (and including) stop.
    location: Horizons coordinate origin (e.g. '@sun').
    """

    body_id: str
    start: str
    stop: str
    step_s: int
    location: str = "@sun"

    def key(self) -> str:
        raw = f"{self.body_id}|{self.start}|{self.stop}|{self.step_s}|{self.location}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


@dataclass
class Ephemeris:
    """Sampled state history for one body over a span.

    times_jd: [N] Julian dates of each sample.
    r: [N, 3] position in km (in the ``location`` frame).
    v: [N, 3] velocity in km/s.
    """

    spec: SpanSpec
    times_jd: Vec
    r: Vec
    v: Vec

    def state_at_jd(self, jd: float) -> tuple[Vec, Vec]:
        """Linear-interpolate (r, v) at Julian date ``jd``.

        Clamps to the span endpoints rather than extrapolating; callers should
        request spans that cover their whole trajectory horizon so clamping is
        never silently hit.
        """
        t = self.times_jd
        if jd <= float(t[0]):
            return self.r[0].copy(), self.v[0].copy()
        if jd >= float(t[-1]):
            return self.r[-1].copy(), self.v[-1].copy()
        i = int(np.searchsorted(t, jd)) - 1
        frac = (jd - float(t[i])) / (float(t[i + 1]) - float(t[i]))
        r_out: Vec = self.r[i] + frac * (self.r[i + 1] - self.r[i])
        v_out: Vec = self.v[i] + frac * (self.v[i + 1] - self.v[i])
        return r_out, v_out


# --------------------------------------------------------------------------
# The only function that touches the network. Isolated for testing/mocking.
# --------------------------------------------------------------------------
def _horizons_vectors(spec: SpanSpec) -> tuple[Vec, Vec, Vec]:
    """One bulk JPL Horizons vectors query for a span. Returns (times_jd, r_km, v_kms).

    Delegates to ``horizons_backend`` (the isolated untyped-library boundary).
    This wrapper is the single network entry point; tests monkeypatch it to
    avoid hitting the server.
    """
    from horizons_backend import horizons_vectors

    return horizons_vectors(spec.body_id, spec.start, spec.stop,
                            spec.step_s, spec.location)


def _cache_path(cache_dir: str, spec: SpanSpec) -> str:
    return os.path.join(cache_dir, f"{spec.body_id}_{spec.key()}.npz")


def _manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "manifest.json")


def _load_manifest(cache_dir: str) -> dict[str, Any]:
    p = _manifest_path(cache_dir)
    if os.path.exists(p):
        with open(p) as f:
            data: dict[str, Any] = json.load(f)
            return data
    return {}


def _update_manifest(cache_dir: str, spec: SpanSpec) -> None:
    manifest = _load_manifest(cache_dir)
    manifest[spec.key()] = {
        "body_id": spec.body_id,
        "start": spec.start,
        "stop": spec.stop,
        "step_s": spec.step_s,
        "location": spec.location,
        "file": os.path.basename(_cache_path(cache_dir, spec)),
    }
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, _manifest_path(cache_dir))


def get_span(spec: SpanSpec, *, cache_dir: str = DEFAULT_CACHE_DIR,
             allow_network: bool = True) -> Ephemeris:
    """Return the ephemeris for ``spec``, from cache if present else one bulk fetch.

    Idempotent: calling repeatedly with the same spec makes exactly one network
    call total (on first miss); every later call is a disk load. If the cache is
    absent and ``allow_network`` is False, a miss raises instead of querying.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, spec)

    if os.path.exists(path):
        data = np.load(path)
        return Ephemeris(
            spec,
            np.asarray(data["times_jd"], dtype=np.float64),
            np.asarray(data["r"], dtype=np.float64),
            np.asarray(data["v"], dtype=np.float64),
        )

    if not allow_network:
        raise LookupError(
            f"ephemeris span not cached and allow_network=False: {spec.key()}"
        )

    times_jd, r, v = _horizons_vectors(spec)

    # Atomic write: temp file + rename, so a concurrent reader never sees a
    # partial .npz and a re-run is idempotent.
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".npz.tmp")
    os.close(fd)
    np.savez(tmp, times_jd=times_jd, r=r, v=v)
    if os.path.exists(tmp + ".npz"):
        os.replace(tmp + ".npz", path)
        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        os.replace(tmp, path)
    _update_manifest(cache_dir, spec)
    return Ephemeris(spec, times_jd, r, v)


def bulk_fetch(body_ids: Sequence[str], start: str, stop: str, step_s: int, *,
               location: str = "@sun", cache_dir: str = DEFAULT_CACHE_DIR,
               allow_network: bool = True) -> dict[str, Ephemeris]:
    """Fetch/lookup many bodies over one span. One network call per uncached body.

    Returns {body_id: Ephemeris}. This is the entry point training code should
    use once, up front, to pre-populate the cache for a whole run.
    """
    out: dict[str, Ephemeris] = {}
    for bid in body_ids:
        spec = SpanSpec(str(bid), start, stop, int(step_s), location)
        out[str(bid)] = get_span(spec, cache_dir=cache_dir, allow_network=allow_network)
    return out


def _main() -> None:
    import sys

    if len(sys.argv) < 5:
        print("usage: ephemeris.py START_ISO STOP_ISO STEP_S BODY_ID [BODY_ID ...]")
        raise SystemExit(2)
    start_iso, stop_iso, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
    bodies = sys.argv[4:]
    eph = bulk_fetch(bodies, start_iso, stop_iso, step)
    for bid, e in eph.items():
        print(f"body {bid}: {len(e.times_jd)} samples cached "
              f"[{float(e.times_jd[0]):.3f} .. {float(e.times_jd[-1]):.3f} JD]")
    print(f"cache dir: {DEFAULT_CACHE_DIR}")


if __name__ == "__main__":
    _main()
