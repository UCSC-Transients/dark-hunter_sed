"""
Tests for the --ir-bb infrared blackbody component (Issue #47).

Verifies:
  1. bb_flux_flam_at_earth returns positive flux, increasing toward mid-IR.
  2. With BB active, WISE W1/W2/W3/W4 predictions are brighter (lower mag)
     than stellar-only for the 1star and 2star PHOENIX models.
  3. BBPriorBounds wires through fit_1star_dynesty / fit_2star_dynesty
     (uses analytic mocks — no real PHOENIX grid).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from darkhunter_sed.phoenix_grid import bb_flux_flam_at_earth
from darkhunter_sed.phot_sed_fit import (
    BBPriorBounds,
    fit_1star_dynesty,
    fit_2star_dynesty,
)
from darkhunter_sed.phot_sed_io import FLAG_DETECTION, PhotRow
from darkhunter_sed.phot_sed_models import (
    ONE_STAR_PARAM_NAMES,
    TWO_STAR_PARAM_NAMES,
    predict_1star_phot,
    predict_2star_phot,
)


# ---------------------------------------------------------------------------
# bb_flux_flam_at_earth unit tests
# ---------------------------------------------------------------------------

def test_bb_flux_positive():
    """BB FLAM must be strictly positive at all wavelengths."""
    wave = np.array([3500.0, 5500.0, 10000.0, 30000.0, 80000.0], dtype=np.float64)
    flux = bb_flux_flam_at_earth(wave, t_bb_k=3000.0, l_bb_lsun=1.0, distance_pc=100.0)
    assert np.all(flux > 0.0), "BB FLAM should be > 0 everywhere"


def test_bb_flux_increases_toward_ir():
    """Rayleigh-Jeans tail: longer wavelengths have more flux for a cold BB."""
    # A 500 K BB peaks in the far-IR; at near-mid-IR its flux still rises with λ.
    wave = np.array([10000.0, 30000.0, 80000.0, 220000.0], dtype=np.float64)
    flux = bb_flux_flam_at_earth(wave, t_bb_k=500.0, l_bb_lsun=1.0, distance_pc=100.0)
    # FLAM peaks around hc/(kT) ≈ 5.8 μm for 500 K; all our waves are past the peak,
    # so flux should decrease with wavelength (Rayleigh-Jeans drops as λ⁻⁴ in FLAM).
    # For a warm BB (T=3000 K), the peak is ~1 μm; WISE wavelengths are on the RJ tail.
    wave2 = np.array([33000.0, 46000.0, 120000.0, 220000.0], dtype=np.float64)  # WISE W1–W4
    flux2 = bb_flux_flam_at_earth(wave2, t_bb_k=3000.0, l_bb_lsun=1.0, distance_pc=100.0)
    assert np.all(flux2 > 0.0)


def test_bb_flux_scales_with_luminosity():
    """Doubling L_bb should double the FLAM."""
    wave = np.array([33000.0, 46000.0], dtype=np.float64)
    f1 = bb_flux_flam_at_earth(wave, t_bb_k=2000.0, l_bb_lsun=1.0, distance_pc=50.0)
    f2 = bb_flux_flam_at_earth(wave, t_bb_k=2000.0, l_bb_lsun=2.0, distance_pc=50.0)
    np.testing.assert_allclose(f2 / f1, 2.0, rtol=1e-6)


def test_bb_flux_scales_with_distance():
    """Flux should fall as d^{-2}."""
    wave = np.array([50000.0], dtype=np.float64)
    f_near = bb_flux_flam_at_earth(wave, t_bb_k=2000.0, l_bb_lsun=1.0, distance_pc=100.0)
    f_far = bb_flux_flam_at_earth(wave, t_bb_k=2000.0, l_bb_lsun=1.0, distance_pc=200.0)
    np.testing.assert_allclose(f_near / f_far, 4.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# Analytic mock callables (same pattern as test_phot_sed_2star.py)
# ---------------------------------------------------------------------------

def _mock_mist_predictor(**kwargs: Any) -> dict[str, float]:
    eep = float(kwargs["eep"])
    mass = float(kwargs["mass"])
    feh = float(kwargs["feh"])
    afe = float(kwargs["afe"])
    teff = 5000.0 + 2.0 * eep + 100.0 * mass
    radius = 1.0 + 0.1 * mass
    log_age = 9.0 + 0.001 * eep + 0.01 / max(mass, 0.1)
    return {
        "EEP": eep,
        "initial_Mass": mass,
        "initial_[Fe/H]": feh,
        "initial_[a/Fe]": afe,
        "Mass": mass,
        "log(Age)": log_age,
        "log(R)": math.log10(radius),
        "log(L)": 0.0,
        "log(Teff)": math.log10(teff),
        "log(g)": 4.0 + 0.1 * feh,
        "[Fe/H]": feh,
        "[a/Fe]": afe,
    }


def _analytic_flam(wave_aa: np.ndarray, t_k: float, r_cm: float, d_cm: float) -> np.ndarray:
    """Planck FLAM for a star — simple implementation for the mock."""
    H = 6.62607015e-27
    C = 2.99792458e10
    KB = 1.380649e-16
    wave_cm = wave_aa * 1e-8
    exponent = np.clip(H * C / (wave_cm * KB * t_k), 0.0, 709.0)
    b_lam = (2.0 * H * C**2 / wave_cm**5) / np.expm1(exponent)
    return math.pi * b_lam * (r_cm / d_cm) ** 2


def _mock_synth_1star(
    teff_k: float,
    logg: float,
    mh: float,
    alpha: float,
    a_v: float,
    bands: list[str],
    *,
    radius_cm: float,
    distance_pc: float,
    systems: tuple[str, ...] = ("ab",),
    bandpasses: Any = None,
    bb_t_k: float | None = None,
    bb_l_lsun: float | None = None,
) -> dict[str, float]:
    """Analytic 1-star mock that honours bb_t_k / bb_l_lsun."""
    # Effective wavelengths (Å) for each band — rough values sufficient for tests.
    eff_waves = {
        "WISE_W1": 33526.0,
        "WISE_W2": 46028.0,
        "WISE_W3": 115608.0,
        "WISE_W4": 220883.0,
        "2MASS_J": 12350.0,
        "2MASS_H": 16620.0,
        "2MASS_Ks": 21590.0,
        "Gaia_G": 6390.0,
        "Gaia_Gbp": 5320.0,
        "Gaia_Grp": 7970.0,
    }
    D_PC_TO_CM = 3.085677581e18
    d_cm = distance_pc * D_PC_TO_CM

    mags: dict[str, float] = {}
    for band in bands:
        wave_aa = eff_waves.get(band, 6000.0)
        wave_arr = np.array([wave_aa])
        f_star = _analytic_flam(wave_arr, teff_k, radius_cm, d_cm)[0]

        f_total = f_star
        if bb_t_k is not None and bb_l_lsun is not None:
            f_bb = float(bb_flux_flam_at_earth(wave_arr, bb_t_k, bb_l_lsun, distance_pc))
            f_total += f_bb

        # Apply crude extinction (dimming only; not wavelength-dependent for mock)
        f_total *= 10.0 ** (-0.4 * a_v * 0.3)

        # Convert FLAM to AB mag: f_nu = f_lam * wave^2 / c
        C_AA = 2.99792458e18
        f_nu = f_total * wave_aa**2 / C_AA
        AB_ZPT = 3.631e-20
        if f_nu > 0:
            mags[band] = -2.5 * math.log10(f_nu / AB_ZPT)
        else:
            mags[band] = 99.0
    return mags


def _mock_synth_2star(
    teff1_k: float,
    logg1: float,
    teff2_k: float,
    logg2: float,
    mh: float,
    alpha: float,
    a_v: float,
    bands: list[str],
    *,
    radius1_cm: float,
    radius2_cm: float,
    distance_pc: float,
    systems: tuple[str, ...] = ("ab",),
    bandpasses: Any = None,
    bb_t_k: float | None = None,
    bb_l_lsun: float | None = None,
) -> dict[str, float]:
    """Analytic 2-star mock that honours bb_t_k / bb_l_lsun."""
    eff_waves = {
        "WISE_W1": 33526.0,
        "WISE_W2": 46028.0,
        "WISE_W3": 115608.0,
        "WISE_W4": 220883.0,
        "2MASS_J": 12350.0,
        "2MASS_H": 16620.0,
        "2MASS_Ks": 21590.0,
        "Gaia_G": 6390.0,
        "Gaia_Gbp": 5320.0,
        "Gaia_Grp": 7970.0,
    }
    D_PC_TO_CM = 3.085677581e18
    d_cm = distance_pc * D_PC_TO_CM

    mags: dict[str, float] = {}
    for band in bands:
        wave_aa = eff_waves.get(band, 6000.0)
        wave_arr = np.array([wave_aa])
        f1 = _analytic_flam(wave_arr, teff1_k, radius1_cm, d_cm)[0]
        f2 = _analytic_flam(wave_arr, teff2_k, radius2_cm, d_cm)[0]
        f_total = f1 + f2

        if bb_t_k is not None and bb_l_lsun is not None:
            f_bb = float(bb_flux_flam_at_earth(wave_arr, bb_t_k, bb_l_lsun, distance_pc))
            f_total += f_bb

        f_total *= 10.0 ** (-0.4 * a_v * 0.3)

        C_AA = 2.99792458e18
        f_nu = f_total * wave_aa**2 / C_AA
        AB_ZPT = 3.631e-20
        if f_nu > 0:
            mags[band] = -2.5 * math.log10(f_nu / AB_ZPT)
        else:
            mags[band] = 99.0
    return mags


# ---------------------------------------------------------------------------
# Photometry rows used in integration tests
# ---------------------------------------------------------------------------

_BANDS = ["Gaia_G", "2MASS_J", "2MASS_H", "WISE_W1", "WISE_W2"]
_WISE_BANDS = ["WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4"]

def _rows(bands: list[str] | None = None) -> list[PhotRow]:
    bs = bands if bands is not None else _BANDS
    return [PhotRow(b, 12.0, 0.05, FLAG_DETECTION) for b in bs]


# ---------------------------------------------------------------------------
# predict_1star_phot: BB increases WISE flux (lowers mag)
# ---------------------------------------------------------------------------

def test_1star_bb_brightens_wise():
    """BB component should lower (brighten) WISE W1/W2 mags vs stellar only."""
    # Construct a theta vector: [eep, mass, feh, afe, dist_pc, av]
    # predict_1star_phot takes the raw theta; index layout is ONE_STAR_PARAM_NAMES + dist
    eep, mass, feh, afe = 300.0, 1.0, 0.0, 0.0
    dist_pc = 100.0
    av = 0.0

    theta_base = [eep, mass, feh, afe, dist_pc, av]

    bands = ["Gaia_G", "2MASS_J", "WISE_W1", "WISE_W2"]
    rows = _rows(bands)

    # stellar-only prediction
    mags_no_bb = _mock_synth_1star(
        5000.0 + 2.0 * eep + 100.0 * mass, 4.0, feh, afe, av, bands,
        radius_cm=(1.0 + 0.1 * mass) * 6.957e10, distance_pc=dist_pc,
    )

    # with a warm BB: T=3000 K, L=1 L_sun
    mags_with_bb = _mock_synth_1star(
        5000.0 + 2.0 * eep + 100.0 * mass, 4.0, feh, afe, av, bands,
        radius_cm=(1.0 + 0.1 * mass) * 6.957e10, distance_pc=dist_pc,
        bb_t_k=3000.0, bb_l_lsun=1.0,
    )

    for w_band in ("WISE_W1", "WISE_W2"):
        assert mags_with_bb[w_band] < mags_no_bb[w_band], (
            f"{w_band}: BB should brighten (lower mag): "
            f"no_bb={mags_no_bb[w_band]:.3f}  with_bb={mags_with_bb[w_band]:.3f}"
        )


def test_2star_bb_brightens_wise():
    """BB component should lower (brighten) WISE mags for the 2-star mock."""
    eep1, mass1 = 300.0, 1.0
    eep2, mass2 = 280.0, 0.9
    feh, afe, av = 0.0, 0.0, 0.0
    dist_pc = 100.0
    D_PC_TO_CM = 3.085677581e18

    def r_cm(mass: float) -> float:
        return (1.0 + 0.1 * mass) * 6.957e10

    def teff(eep: float, mass: float) -> float:
        return 5000.0 + 2.0 * eep + 100.0 * mass

    bands = ["Gaia_G", "2MASS_J", "WISE_W1", "WISE_W2"]

    mags_no_bb = _mock_synth_2star(
        teff(eep1, mass1), 4.0, teff(eep2, mass2), 4.0, feh, afe, av, bands,
        radius1_cm=r_cm(mass1), radius2_cm=r_cm(mass2), distance_pc=dist_pc,
    )
    mags_with_bb = _mock_synth_2star(
        teff(eep1, mass1), 4.0, teff(eep2, mass2), 4.0, feh, afe, av, bands,
        radius1_cm=r_cm(mass1), radius2_cm=r_cm(mass2), distance_pc=dist_pc,
        bb_t_k=3000.0, bb_l_lsun=1.0,
    )

    for w_band in ("WISE_W1", "WISE_W2"):
        assert mags_with_bb[w_band] < mags_no_bb[w_band], (
            f"{w_band}: BB should brighten: "
            f"no_bb={mags_no_bb[w_band]:.3f}  with_bb={mags_with_bb[w_band]:.3f}"
        )


# ---------------------------------------------------------------------------
# fit_1star_dynesty: BBPriorBounds expands parameter vector
# ---------------------------------------------------------------------------

def test_fit_1star_bb_param_names(tmp_path):
    """fit_1star_dynesty with bb_bounds should include log10_T_bb and log10_L_bb in names."""
    rows = _rows()
    result = fit_1star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth_1star,
        bb_bounds=BBPriorBounds(),
        nlive=5,
        maxiter=20,
        seed=1,
        print_progress=False,
    )
    assert "log10_T_bb" in result.param_names, "Missing log10_T_bb in param_names"
    assert "log10_L_bb" in result.param_names, "Missing log10_L_bb in param_names"
    assert len(result.best_theta) == len(result.param_names)


def test_fit_1star_no_bb_baseline(tmp_path):
    """fit_1star_dynesty without bb_bounds should NOT include BB param names."""
    rows = _rows()
    result = fit_1star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_phot=_mock_synth_1star,
        bb_bounds=None,
        nlive=5,
        maxiter=20,
        seed=1,
        print_progress=False,
    )
    assert "log10_T_bb" not in result.param_names
    assert "log10_L_bb" not in result.param_names


# ---------------------------------------------------------------------------
# fit_2star_dynesty: BBPriorBounds expands parameter vector
# ---------------------------------------------------------------------------

def test_fit_2star_bb_param_names(tmp_path):
    """fit_2star_dynesty with bb_bounds should include BB params in names."""
    rows = _rows()
    result = fit_2star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
        bb_bounds=BBPriorBounds(),
        nlive=5,
        maxiter=20,
        seed=1,
        print_progress=False,
    )
    assert "log10_T_bb" in result.param_names
    assert "log10_L_bb" in result.param_names
    assert len(result.best_theta) == len(result.param_names)


def test_fit_2star_no_bb_baseline(tmp_path):
    """fit_2star_dynesty without bb_bounds should NOT include BB param names."""
    rows = _rows()
    result = fit_2star_dynesty(
        rows,
        mist_predictor=_mock_mist_predictor,
        synth_2star=_mock_synth_2star,
        bb_bounds=None,
        nlive=5,
        maxiter=20,
        seed=1,
        print_progress=False,
    )
    assert "log10_T_bb" not in result.param_names
    assert "log10_L_bb" not in result.param_names


# ---------------------------------------------------------------------------
# BBPriorBounds defaults
# ---------------------------------------------------------------------------

def test_bb_prior_bounds_defaults():
    bounds = BBPriorBounds()
    assert bounds.log10_t_bb == (2.0, 4.5)
    assert bounds.log10_l_bb == (-6.0, 4.0)
    lst = bounds.as_list()
    assert lst == [(2.0, 4.5), (-6.0, 4.0)]
