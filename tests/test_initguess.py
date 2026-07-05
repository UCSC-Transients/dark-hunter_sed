"""Tests for Gaia-informed UMS init guessing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from darkhunter_sed import initguess


def _mock_mist_monotone_age(eep, mass, feh, afe, verbose=False):
    del mass, feh, afe, verbose
    age_gyr = 0.05 + (float(eep) - 250.0) * 0.1
    log_age = 9.0 + float(np.log10(age_gyr))
    return {"log(Age)": log_age, "log(Teff)": np.log10(5800.0)}


def test_eep_from_age_on_grid_interpolates():
    grid = np.linspace(250.0, 350.0, 5)
    eep = initguess.eep_from_age_on_grid(_mock_mist_monotone_age, 1.5, 1.0, -0.5, -0.2, grid)
    assert math.isfinite(eep)
    assert 250.0 <= eep <= 500.0


def test_guess_eep_from_age_missing_age_returns_default():
    eep = initguess.guess_eep_from_age(
        "/nonexistent/mist.h5",
        float("nan"),
        1.0,
        0.0,
        -0.2,
        gen_mist_fn=_mock_mist_monotone_age,
    )
    assert eep == initguess.DEFAULT_EEP


def test_guess_eep_from_age_uses_grid():
    eep = initguess.guess_eep_from_age(
        "/nonexistent/mist.h5",
        1.5,
        1.0,
        -0.64,
        -0.2,
        gen_mist_fn=_mock_mist_monotone_age,
    )
    assert 250.0 <= eep <= 500.0


def test_build_ums_initpars_uses_gaia_mass_and_feh(monkeypatch):
    monkeypatch.setattr(initguess, "guess_eep_from_age", lambda *a, **k: 310.0)
    monkeypatch.setattr(initguess, "mist_teff_kelvin", lambda *a, **k: 5788.0)
    fit_data = {
        "Teff": 5788.0,
        "[Fe/H]": -0.64,
        "[a/Fe]": -0.2,
        "Mass": 1.05,
        "Age_Gyr": 2.0,
    }
    initpars = initguess.build_ums_initpars(
        fit_data,
        "/nonexistent/mist.h5",
        dist_init=87.0,
        fix_photjitter=False,
        fix_specjitter=False,
        dospec=False,
        nspec=0,
    )
    assert initpars["initial_[Fe/H]"] == pytest.approx(-0.64)
    assert initpars["initial_Mass"] == pytest.approx(1.05)
    assert initpars["dist"] == pytest.approx(87.0)
    assert initpars["photjitter"] == pytest.approx(1e-4)
