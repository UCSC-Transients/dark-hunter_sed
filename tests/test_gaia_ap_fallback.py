"""Tests for Gaia GSP-Phot coalesce (gaia_source → astrophysical_parameters)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy.table import Table

from darkhunter_sed import stellar_data as sd


def test_coalesce_float_prefers_first_finite():
    assert sd._coalesce_float(float("nan"), 5800.0, 6000.0) == pytest.approx(5800.0)
    assert sd._coalesce_float(None, np.ma.masked, 4.2) == pytest.approx(4.2)
    assert math.isnan(sd._coalesce_float(float("nan"), None))


def test_query_gaia_uses_ap_when_gs_atmosphere_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate ESAC result already coalesced via ADQL COALESCE(gs, ap)."""
    result = Table(
        rows=[[8164.0, 4.1, -0.1, 10.0, 20.0, 1.7, 0.5, "00", 2.5, 0.1]],
        names=[
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "ra",
            "dec",
            "mass_flame",
            "age_flame",
            "flags_flame",
            "final_parallax",
            "final_parallax_error",
        ],
    )

    class _Job:
        def get_results(self):
            return result

    class _Gaia:
        @staticmethod
        def launch_job(query: str):
            assert "COALESCE(gs.teff_gspphot, ap.teff_gspphot)" in query
            assert "gaiadr3.astrophysical_parameters" in query
            return _Job()

    monkeypatch.setattr(sd, "Gaia", _Gaia)
    out = sd.query_gaia_stellar_priors("4144743775455688576")
    assert out["Teff"] == pytest.approx(8164.0)
    assert out["log(g)"] == pytest.approx(4.1)
    assert out["[Fe/H]"] == pytest.approx(-0.1)
    assert out["Mass_FLAME"] == pytest.approx(1.7)
    assert out["Age_Gyr"] == pytest.approx(0.5)


def test_fallback_coalesces_ap_over_null_gs(monkeypatch: pytest.MonkeyPatch) -> None:
    gs = Table(
        rows=[[float("nan"), float("nan"), float("nan"), 1.0, 2.0, 3.0, 0.1]],
        names=[
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "ra",
            "dec",
            "parallax",
            "parallax_error",
        ],
    )
    ap = Table(
        rows=[[7500.0, 3.9, 0.05, 1.6, 1.2, "01"]],
        names=[
            "teff_gspphot",
            "logg_gspphot",
            "mh_gspphot",
            "mass_flame",
            "age_flame",
            "flags_flame",
        ],
    )

    def fake_tap(sync_url: str, adql: str):
        if "gaiadr3.gaia_source" in adql:
            return gs
        if "gaiadr3.astrophysical_parameters" in adql:
            return ap
        if "nss_two_body_orbit" in adql or "nss_acceleration_astro" in adql:
            return None
        raise AssertionError(f"unexpected ADQL: {adql}")

    monkeypatch.setattr(sd, "_tap_sync_votable_table", fake_tap)
    tab = sd._query_gaia_stellar_priors_fallback(123, sync_url="http://example/tap")
    assert float(tab["teff_gspphot"][0]) == pytest.approx(7500.0)
    assert float(tab["logg_gspphot"][0]) == pytest.approx(3.9)
    assert float(tab["mh_gspphot"][0]) == pytest.approx(0.05)
    assert float(tab["mass_flame"][0]) == pytest.approx(1.6)
