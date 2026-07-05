"""Tests for RV prior parsing and spectrum utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_sed import priors, spectrum


FIXTURE_SUMMARY = Path(__file__).resolve().parents[2].parent / "rvs" / "dark-hunter_rv" / "validation_output" / "blaze_only_1702" / "Gaia_DR3_1702370142434513152_summary.txt"


@pytest.mark.skipif(not FIXTURE_SUMMARY.is_file(), reason="RV validation summary not present")
def test_parse_pipeline_rv_epochs():
    epochs = priors.parse_pipeline_rv_epochs(FIXTURE_SUMMARY)
    assert len(epochs) >= 2
    assert all(e.rv_kms == e.rv_kms for e in epochs)  # finite
    assert epochs[0].epoch_num >= 1


def test_build_vrad_normal_prior_inflates_error():
    ep = priors.RvEpoch("f.txt", 1, 58000.0, -11.0, 0.05)
    init, pr = priors.build_vrad_init_and_priors(
        [ep], prior_kind="normal", err_inflate=2.0, err_floor_kms=0.01
    )
    assert init["vrad_0"] == pytest.approx(-11.0)
    assert pr["vrad_0"] == ["normal", [-11.0, 0.1]]


def test_build_vrad_prior_respects_floor():
    ep = priors.RvEpoch("f.txt", 1, 58000.0, 5.0, 0.01)
    _, pr = priors.build_vrad_init_and_priors([ep], err_inflate=1.0, err_floor_kms=2.0)
    assert pr["vrad_0"][1][1] == 2.0


def test_stellar_priors_flame_mass_and_age():
    meta = {
        "Teff": 5788.0,
        "logg": 4.17,
        "MH": -0.64,
        "Parallax": 11.41,
        "Parallax_Error": 0.032,
        "Mass_FLAME": 1.05,
        "Age_FLAME": 2.3,
        "Flags_FLAME": "00",
    }
    out = priors.stellar_priors_from_summary_metadata(meta)
    assert out["Mass"] == pytest.approx(1.05)
    assert out["Age_Gyr"] == pytest.approx(2.3)
    assert out["[Fe/H]"] == pytest.approx(-0.64)
    assert out["Flags_FLAME"] == "00"


def test_match_rv_epochs_to_spectrum_paths():
    epochs = [
        priors.RvEpoch("Gaia_DR3_1_epoch_2.txt", 2, 1.0, 10.0, 1.0),
        priors.RvEpoch("Gaia_DR3_1_epoch_1.txt", 1, 1.0, -5.0, 1.0),
    ]
    paths = [Path("Gaia_DR3_1_epoch_1.txt"), Path("Gaia_DR3_1_epoch_2.txt")]
    matched = priors.match_rv_epochs_to_spectrum_paths(epochs, paths)
    assert matched[0].rv_kms == pytest.approx(-5.0)
    assert matched[1].rv_kms == pytest.approx(10.0)


def test_epoch_sort():
    paths = [
        Path("Gaia_DR3_x_epoch_10.txt"),
        Path("Gaia_DR3_x_epoch_2.txt"),
    ]
    ordered = spectrum.sort_spectrum_paths(paths)
    assert "epoch_2" in ordered[0].name
