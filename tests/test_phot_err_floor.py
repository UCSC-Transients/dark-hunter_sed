"""Photometry uncertainty floors for SED fitting."""

from __future__ import annotations

from darkhunter_sed import data


def test_apply_photometry_error_floors_gaia() -> None:
    phot = {
        "GaiaDR3_G": [9.1, 0.0002],
        "2MASS_J": [7.9, 0.022],
    }
    out = data.apply_photometry_error_floors(phot, default_floor_mag=0.02, gaia_floor_mag=0.03)
    assert out["GaiaDR3_G"][1] == 0.03
    assert out["2MASS_J"][1] == 0.022
