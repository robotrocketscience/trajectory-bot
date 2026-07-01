#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the bulk/idempotent ephemeris cache. No network access.

The single network-touching function (`ephemeris._horizons_vectors`) is
monkeypatched with a call-counting fake, so these tests prove the caching /
idempotency contract without ever contacting JPL Horizons.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ephemeris as eph  # noqa: E402


def _fake_vectors_factory(counter):
    """Return a fake _horizons_vectors that counts calls and yields synthetic data."""
    def _fake(spec):
        counter["n"] += 1
        n = 11
        times_jd = np.linspace(2459337.5, 2459338.5, n)  # 1-day span, 11 samples
        r = np.tile(np.array([1.0, 2.0, 3.0]), (n, 1)) * np.arange(1, n + 1)[:, None]
        v = np.tile(np.array([0.1, 0.2, 0.3]), (n, 1))
        return times_jd, r, v
    return _fake


@pytest.fixture
def spec():
    return eph.SpanSpec(body_id="399", start="2021-05-03T00:00:00",
                        stop="2021-05-04T00:00:00", step_s=8640, location="@sun")


def test_first_call_fetches_once_then_caches(tmp_path, monkeypatch, spec):
    counter = {"n": 0}
    monkeypatch.setattr(eph, "_horizons_vectors", _fake_vectors_factory(counter))

    e1 = eph.get_span(spec, cache_dir=str(tmp_path))
    assert counter["n"] == 1                      # exactly one network call
    assert os.path.exists(eph._cache_path(str(tmp_path), spec))
    assert e1.r.shape == (11, 3)

    # Second identical request must NOT hit the network again (idempotent).
    e2 = eph.get_span(spec, cache_dir=str(tmp_path))
    assert counter["n"] == 1                      # still one — served from disk
    np.testing.assert_array_equal(e1.r, e2.r)
    np.testing.assert_array_equal(e1.times_jd, e2.times_jd)


def test_manifest_indexes_span(tmp_path, monkeypatch, spec):
    counter = {"n": 0}
    monkeypatch.setattr(eph, "_horizons_vectors", _fake_vectors_factory(counter))
    eph.get_span(spec, cache_dir=str(tmp_path))

    manifest = eph._load_manifest(str(tmp_path))
    assert spec.key() in manifest
    entry = manifest[spec.key()]
    assert entry["body_id"] == "399"
    assert entry["step_s"] == 8640


def test_allow_network_false_raises_on_miss(tmp_path, monkeypatch, spec):
    counter = {"n": 0}
    monkeypatch.setattr(eph, "_horizons_vectors", _fake_vectors_factory(counter))
    with pytest.raises(LookupError):
        eph.get_span(spec, cache_dir=str(tmp_path), allow_network=False)
    assert counter["n"] == 0                       # never queried


def test_bulk_fetch_one_call_per_body_then_zero(tmp_path, monkeypatch):
    counter = {"n": 0}
    monkeypatch.setattr(eph, "_horizons_vectors", _fake_vectors_factory(counter))
    bodies = ["399", "301", "10"]

    eph.bulk_fetch(bodies, "2021-05-03T00:00:00", "2021-05-04T00:00:00", 8640,
                   cache_dir=str(tmp_path))
    assert counter["n"] == 3                        # one per body

    # Re-run over the same span: fully served from cache, zero new calls.
    eph.bulk_fetch(bodies, "2021-05-03T00:00:00", "2021-05-04T00:00:00", 8640,
                   cache_dir=str(tmp_path))
    assert counter["n"] == 3                        # unchanged — idempotent


def test_state_interpolation_and_clamp(tmp_path, monkeypatch, spec):
    counter = {"n": 0}
    monkeypatch.setattr(eph, "_horizons_vectors", _fake_vectors_factory(counter))
    e = eph.get_span(spec, cache_dir=str(tmp_path))

    # Midpoint between sample 0 and 1: linear interpolation.
    mid_jd = 0.5 * (e.times_jd[0] + e.times_jd[1])
    r_mid, _ = e.state_at_jd(mid_jd)
    np.testing.assert_allclose(r_mid, 0.5 * (e.r[0] + e.r[1]))

    # Before the span start clamps to the first sample (no extrapolation).
    r_before, _ = e.state_at_jd(e.times_jd[0] - 10.0)
    np.testing.assert_array_equal(r_before, e.r[0])
    # After the span end clamps to the last sample.
    r_after, _ = e.state_at_jd(e.times_jd[-1] + 10.0)
    np.testing.assert_array_equal(r_after, e.r[-1])


def test_cache_key_is_stable_and_param_sensitive():
    a = eph.SpanSpec("399", "2021-05-03T00:00:00", "2021-05-04T00:00:00", 3600)
    b = eph.SpanSpec("399", "2021-05-03T00:00:00", "2021-05-04T00:00:00", 3600)
    c = eph.SpanSpec("399", "2021-05-03T00:00:00", "2021-05-04T00:00:00", 60)
    assert a.key() == b.key()          # same params -> same key (idempotent)
    assert a.key() != c.key()          # different step -> different key
