"""PS1 photometry from the Gaia archive join (no network)."""

from __future__ import annotations

import inspect

from astropy.table import Table

from darkhunter_sed import photometry_gather
from darkhunter_sed.photometry_gather import _append_ps1_from_gaia_archive

_ARCHIVE_PS1_COLUMNS = {
    "g_mean_psf_mag", "g_mean_psf_mag_error", "r_mean_psf_mag", "r_mean_psf_mag_error",
    "i_mean_psf_mag", "i_mean_psf_mag_error", "z_mean_psf_mag", "z_mean_psf_mag_error",
    "y_mean_psf_mag", "y_mean_psf_mag_error", "n_detections",
}


def _row(n_detections: int):
    t = Table(
        {
            "g_mean_psf_mag": [15.1], "g_mean_psf_mag_error": [0.01],
            "r_mean_psf_mag": [14.7], "r_mean_psf_mag_error": [0.01],
            "i_mean_psf_mag": [14.5], "i_mean_psf_mag_error": [0.01],
            "z_mean_psf_mag": [14.4], "z_mean_psf_mag_error": [0.01],
            "y_mean_psf_mag": [14.3], "y_mean_psf_mag_error": [0.02],
            "n_detections": [n_detections],
        }
    )
    return t[0]


def test_ps1_archive_appends_all_bands_with_enough_detections() -> None:
    phot: list[tuple[str, float, float]] = []
    assert _append_ps1_from_gaia_archive(_row(12), phot) == 5
    assert [b for b, _, _ in phot] == ["PS_g", "PS_r", "PS_i", "PS_z", "PS_y"]


def test_ps1_archive_skips_single_detection_objects() -> None:
    phot: list[tuple[str, float, float]] = []
    assert _append_ps1_from_gaia_archive(_row(1), phot) == 0
    assert phot == []


def test_ps1_archive_query_requests_only_existing_columns() -> None:
    src = inspect.getsource(photometry_gather.query_catalogs)
    assert "num_sigma_clipped" not in src
    for col in _ARCHIVE_PS1_COLUMNS:
        assert f"ps.{col}" in src
