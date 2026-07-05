"""Tests for dust-map Av priors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from darkhunter_sed import dust_prior as dp


@dataclass
class _MockBackend:
    bayestar: tuple[float, float] | None = None
    decaps: tuple[float, float] | None = None
    edenhofer: tuple[float, float] | None = None
    chen_3d: tuple[float, float] | None = None
    chen_los: float | None = None
    csfd: float | None = None

    def query_bayestar(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.bayestar

    def query_decaps(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.decaps

    def query_edenhofer(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.edenhofer

    def query_chen_3d(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.chen_3d

    def query_chen_los(self, l_deg: float, b_deg: float) -> float | None:
        return self.chen_los

    def query_csfd(self, l_deg: float, b_deg: float) -> float | None:
        return self.csfd


def test_tnormal_informative_bounds():
    prior = dp._tnormal_informative(0.2, 0.05)
    assert prior[0] == "tnormal"
    loc, scale, low, high = prior[1]
    assert loc == pytest.approx(0.2)
    assert scale == pytest.approx(0.05)
    assert low == pytest.approx(0.05)
    assert high == pytest.approx(0.35)


def test_tnormal_upper_limit():
    prior = dp._tnormal_upper_limit(0.3)
    loc, scale, low, high = prior[1]
    assert loc == 0.0
    assert scale == pytest.approx(0.1)
    assert high == pytest.approx(0.3)


def test_ebv_to_av():
    assert dp.ebv_to_av(0.1) == pytest.approx(0.332)


def test_decaps_footprint_wrap():
    assert dp.in_decaps_footprint(250.0, 5.0)
    assert dp.in_decaps_footprint(3.0, -2.0)
    assert not dp.in_decaps_footprint(90.0, 5.0)


def test_edenhofer_distance_clip():
    assert dp.in_edenhofer_distance_range(100.0)
    assert not dp.in_edenhofer_distance_range(50.0)
    assert not dp.in_edenhofer_distance_range(2000.0)


def test_build_av_prior_bayestar_chain():
    backend = _MockBackend(bayestar=(0.15, 0.02))
    result = dp.build_av_prior(
        ra_deg=180.0,
        dec_deg=45.0,
        parallax_mas=10.0,
        backend=backend,
    )
    assert result.map_used == "bayestar2019"
    assert result.prior_kind == "informative_3d"
    assert result.a_v_med == pytest.approx(0.15)


def test_build_av_prior_fallback_csfd():
    backend = _MockBackend(csfd=0.05)
    result = dp.build_av_prior(
        ra_deg=30.0,
        dec_deg=-50.0,
        parallax_mas=5.0,
        backend=backend,
    )
    assert result.map_used == "csfd"
    assert result.prior_kind == "upper_limit"
    assert result.prior[1][0] == 0.0
    assert result.prior[1][3] == pytest.approx(0.05)


def test_build_av_prior_legacy_when_disabled():
    result = dp.build_av_prior(
        ra_deg=0.0,
        dec_deg=0.0,
        parallax_mas=1.0,
        use_dustmaps=False,
    )
    assert result.map_used == "legacy"


def test_build_av_prior_from_fit_data():
    fit_data = {
        "RA": 180.0,
        "Dec": 30.0,
        "parallax": [5.0, 0.1],
    }
    result = dp.build_av_prior_from_fit_data(fit_data, use_dustmaps=False)
    assert result.map_used == "legacy"
    assert math.isfinite(result.init_av)


def test_init_from_prior_av_avoids_exact_zero():
    prior = ["tnormal", [0.0, 0.1, 0.0, 0.5]]
    init = dp._init_from_prior(prior)
    assert init > 0.0
    assert init <= 0.5
