"""Unit tests for WISE and DECam-u photometry in gather_phot."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.table import Table

from darkhunter_sed.photometry_gather import (
    _append_decam_u_vizier,
    _append_wise_from_irsa,
    _decam_u_from_gaia_archive,
    query_catalogs,
)


def test_append_wise_from_irsa_filters_bad_values(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Table(
        {
            "w1mpro": [12.1],
            "w1sigmpro": [0.05],
            "w2mpro": [np.nan],
            "w2sigmpro": [0.04],
        }
    )

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Irsa.query_region",
        lambda *args, **kwargs: table,
    )
    phot: list[tuple[str, float, float]] = []
    n = _append_wise_from_irsa(MagicMock(), phot, MagicMock())
    assert n == 1
    assert phot == [("WISE_W1", 12.1, 0.05)]


def test_decam_u_from_gaia_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = Table({"u_psf": [18.2], "e_u_psf": [0.03]})

    class _Job:
        def get_results(self):
            return rows

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Gaia.launch_job",
        lambda _q: _Job(),
    )
    pair = _decam_u_from_gaia_archive("123")
    assert pair == (18.2, 0.03)


def test_append_decam_u_vizier(monkeypatch: pytest.MonkeyPatch) -> None:
    table = Table({"uPSF": [17.5], "e_uPSF": [0.02]})

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Vizier.query_region",
        lambda *args, **kwargs: {"II/358/dr2": table},
    )
    phot: list[tuple[str, float, float]] = []
    n = _append_decam_u_vizier(MagicMock(), phot, MagicMock())
    assert n == 1
    assert phot[0][0] == "DECam_u"


def test_query_catalogs_includes_wise_and_decam_u(monkeypatch: pytest.MonkeyPatch) -> None:
    gaia_row = Table(
        {
            "ra": [10.0],
            "dec": [-20.0],
            "ref_epoch": [2016.0],
            "pmra": [1.0],
            "pmdec": [-1.0],
            "phot_g_mean_mag": [11.0],
            "phot_g_mean_flux": [1000.0],
            "phot_g_mean_flux_error": [10.0],
            "phot_bp_mean_mag": [11.5],
            "phot_bp_mean_flux": [900.0],
            "phot_bp_mean_flux_error": [10.0],
            "phot_rp_mean_mag": [10.5],
            "phot_rp_mean_flux": [1100.0],
            "phot_rp_mean_flux_error": [10.0],
        }
    )
    empty = Table()

    def fake_launch_job(query: str):
        class _Job:
            def get_results(self_inner):
                if "gaia_source" in query:
                    return gaia_row
                return empty

        return _Job()

    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Gaia.launch_job", fake_launch_job
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather._append_wise_from_irsa",
        lambda position, phot, radius: phot.extend(
            [("WISE_W1", 8.0, 0.01), ("WISE_W2", 7.5, 0.01)]
        )
        or 2,
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather._decam_u_from_gaia_archive",
        lambda _sid: (16.0, 0.02),
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Irsa.query_region",
        lambda *args, **kwargs: empty,
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.SDSS.query_crossid",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Catalogs.query_region",
        lambda *args, **kwargs: empty,
    )
    monkeypatch.setattr(
        "darkhunter_sed.photometry_gather.Vizier.query_region",
        lambda *args, **kwargs: None,
    )

    phot = query_catalogs("999")
    bands = [b for b, _m, _e in phot]
    assert "WISE_W1" in bands
    assert "WISE_W2" in bands
    assert "DECam_u" in bands
