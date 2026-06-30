"""Tests for posterior JSON extraction."""

from __future__ import annotations

import json

import numpy as np
import pytest
from astropy.table import Table

from darkhunter_sed import posterior


def _write_synthetic_ums_fits(path, n=500):
    rng = np.random.default_rng(42)
    mass = rng.normal(1.05, 0.04, n)
    tbl = Table(
        {
            "initial_Mass": mass,
            "EEP": rng.normal(300, 10, n),
            "Teff": rng.normal(6150, 50, n),
            "dist": rng.normal(331, 5, n),
            "vrad_0": rng.normal(-11.0, 0.5, n),
            "vrad_1": rng.normal(22.5, 0.5, n),
        }
    )
    tbl.write(path, overwrite=True)


def test_summarize_samples_fits(tmp_path):
    fits = tmp_path / "test_ums.fits"
    _write_synthetic_ums_fits(fits)
    summary = posterior.summarize_samples_fits(fits, fit_type="ums")
    assert summary["fit_type"] == "ums"
    assert summary["parameters"]["initial_Mass"]["median"] == pytest.approx(1.05, abs=0.02)
    assert len(summary["vrad_epochs"]) == 2


def test_write_and_read_sed_summary(tmp_path):
    fits = tmp_path / "ums.fits"
    _write_synthetic_ums_fits(fits)
    out = tmp_path / "Gaia_DR3_123_sed_summary.json"
    posterior.write_sed_summary("123", ums_samples=fits, out_path=out)
    doc = posterior.read_sed_summary(out)
    assert doc["gaia_source_id"] == "123"
    assert doc["m1_msun"]["median"] == pytest.approx(1.05, abs=0.02)
    m1 = posterior.m1_msun_from_summary(doc)
    assert m1 == pytest.approx(1.05, abs=0.02)


def test_percentile_stats_empty():
    stats = posterior.percentile_stats([np.nan, np.nan])
    assert stats["n_finite"] == 0
    assert np.isnan(stats["median"])
