"""Tests for Path-2 dust and astrometric prior construction (phot_sed_priors)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from darkhunter_sed import phot_sed_priors as psp


# ---------------------------------------------------------------------------
# Mock CSFD / dust backend
# ---------------------------------------------------------------------------


@dataclass
class _MockBackend:
    """Minimal injectable DustQueryBackend for tests."""

    csfd_ebv: float | None = None
    bayestar: tuple[float, float] | None = None
    edenhofer: tuple[float, float] | None = None
    chen_3d: tuple[float, float] | None = None

    def query_csfd(self, l_deg: float, b_deg: float) -> float | None:
        if self.csfd_ebv is None:
            return None
        from darkhunter_sed.dust_prior import RV_DUST

        return self.csfd_ebv * RV_DUST  # dust_prior returns A_V at R_V=3.32

    def query_bayestar(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.bayestar

    def query_decaps(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return None

    def query_edenhofer(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.edenhofer

    def query_chen_3d(self, l_deg: float, b_deg: float, d_pc: float) -> tuple[float, float] | None:
        return self.chen_3d

    def query_chen_los(self, l_deg: float, b_deg: float) -> float | None:
        return None


# ---------------------------------------------------------------------------
# R_V constant
# ---------------------------------------------------------------------------


def test_rv_f99_value() -> None:
    assert psp.R_V_F99 == pytest.approx(3.1)


# ---------------------------------------------------------------------------
# get_av_sf
# ---------------------------------------------------------------------------


def test_get_av_sf_converts_ebv_at_rv31() -> None:
    # CSFD returns E(B-V)=0.1 → query_csfd returns A_V at R_V=3.32 → back-convert to 3.1.
    be = _MockBackend(csfd_ebv=0.1)
    av_sf = psp.get_av_sf(45.0, 30.0, backend=be)
    assert av_sf == pytest.approx(psp.R_V_F99 * 0.1)


def test_get_av_sf_nan_on_none_csfd() -> None:
    be = _MockBackend(csfd_ebv=None)
    av_sf = psp.get_av_sf(45.0, 30.0, backend=be)
    assert math.isnan(av_sf)


# ---------------------------------------------------------------------------
# build_av_prior_spec — default flat
# ---------------------------------------------------------------------------


def test_default_flat_prior_uses_csfd_cap() -> None:
    be = _MockBackend(csfd_ebv=0.5)  # E(B-V)=0.5 → Av_SF = 3.1*0.5 = 1.55
    spec = psp.build_av_prior_spec(45.0, 30.0, d_pc=None, prior_dust=False, backend=be)
    assert spec.kind == "flat"
    assert spec.av_lo == 0.0
    assert spec.av_hi == pytest.approx(psp.R_V_F99 * 0.5)


def test_default_flat_prior_fallback_when_csfd_fails() -> None:
    be = _MockBackend(csfd_ebv=None)
    spec = psp.build_av_prior_spec(45.0, 30.0, d_pc=None, prior_dust=False, backend=be)
    assert spec.kind == "flat"
    assert spec.av_hi == pytest.approx(psp._AV_FALLBACK_HI)


def test_flat_av_prior_av_sf_override() -> None:
    spec = psp.build_av_prior_spec(0.0, 0.0, d_pc=None, prior_dust=False, av_sf=2.5)
    assert spec.av_hi == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# build_av_prior_spec — --prior-dust (Gaussian with CSFD cap)
# ---------------------------------------------------------------------------


def test_dust_prior_gaussian_kind() -> None:
    # Bayestar returns (A_V_mean=0.3, A_V_sigma=0.05); CSFD gives Av_SF=1.55.
    be = _MockBackend(csfd_ebv=0.5, bayestar=(0.3, 0.05))
    spec = psp.build_av_prior_spec(
        45.0, 30.0, d_pc=200.0, prior_dust=True, backend=be
    )
    assert spec.kind == "gaussian"
    assert spec.av_mean == pytest.approx(0.3)
    assert spec.av_sigma == pytest.approx(0.05 * 2.0)  # default inflate=2


def test_dust_prior_av_cap_hard() -> None:
    """Av hard cap: av_hi must not exceed Av_SF from CSFD (R_V=3.1 F99)."""
    av_sf = psp.R_V_F99 * 0.5  # = 1.55
    be = _MockBackend(csfd_ebv=0.5, bayestar=(1.0, 0.5))  # wide sigma
    spec = psp.build_av_prior_spec(
        45.0, 30.0, d_pc=200.0, prior_dust=True, backend=be
    )
    assert spec.av_hi <= av_sf + 1e-9, (
        f"av_hi={spec.av_hi:.4f} exceeds Av_SF={av_sf:.4f}"
    )


def test_dust_prior_sample_bounded_at_u1() -> None:
    """unit_to_av(u=0.9999) must be <= av_hi (hard cap respected at boundary)."""
    be = _MockBackend(csfd_ebv=0.5, bayestar=(0.3, 0.05))
    spec = psp.build_av_prior_spec(45.0, 30.0, d_pc=200.0, prior_dust=True, backend=be)
    val = psp.unit_to_av(0.9999, spec)
    assert val <= spec.av_hi + 1e-9


def test_dust_prior_fallback_to_flat_when_no_3d_map() -> None:
    """No 3D map available → falls back to flat [0, Av_SF]."""
    be = _MockBackend(csfd_ebv=0.4)  # only CSFD, no 3D map
    spec = psp.build_av_prior_spec(
        45.0, 30.0, d_pc=200.0, prior_dust=True, backend=be
    )
    # dec=30 is in Bayestar footprint but backend returns None → fallback
    assert spec.kind == "flat"


# ---------------------------------------------------------------------------
# build_plx_prior_spec
# ---------------------------------------------------------------------------


def test_plx_prior_flat_default() -> None:
    spec = psp.build_plx_prior_spec(5.0, 0.1, prior_gaia_vac=False)
    assert spec.kind == "flat"
    assert spec.plx_lo == pytest.approx(5.0 - psp._PLX_SIGMA_CLIP * 0.1)
    assert spec.plx_hi == pytest.approx(5.0 + psp._PLX_SIGMA_CLIP * 0.1)


def test_plx_prior_gaussian_with_gaia_vac() -> None:
    spec = psp.build_plx_prior_spec(5.0, 0.1, prior_gaia_vac=True)
    assert spec.kind == "gaussian"
    assert spec.plx_obs == pytest.approx(5.0)
    assert spec.plx_err == pytest.approx(0.1)


def test_plx_prior_gaussian_sample_in_bounds() -> None:
    spec = psp.build_plx_prior_spec(5.0, 0.2, prior_gaia_vac=True)
    for u in (0.0, 0.01, 0.5, 0.99, 1.0):
        val = psp.unit_to_plx(u, spec)
        assert spec.plx_lo <= val <= spec.plx_hi + 1e-9, (
            f"u={u}: val={val:.4f} outside [{spec.plx_lo:.4f}, {spec.plx_hi:.4f}]"
        )


# ---------------------------------------------------------------------------
# load_uberms_prior_spec
# ---------------------------------------------------------------------------


def test_load_uberms_prior_spec_full(tmp_path: Path) -> None:
    data = {"feh": -0.25, "feh_sigma": 0.1, "mass": 1.1, "mass_sigma": 0.15}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data))
    spec = psp.load_uberms_prior_spec(p)
    assert spec.feh_mean == pytest.approx(-0.25)
    assert spec.feh_sigma == pytest.approx(0.1)
    assert spec.mass_mean == pytest.approx(1.1)
    assert spec.mass_sigma == pytest.approx(0.15)


def test_load_uberms_missing_fields(tmp_path: Path) -> None:
    data = {"feh": -0.1}  # no sigma, no mass
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data))
    spec = psp.load_uberms_prior_spec(p)
    assert spec.feh_mean == pytest.approx(-0.1)
    assert spec.feh_sigma is None
    assert spec.mass_mean is None
    assert spec.mass_sigma is None


def test_load_uberms_nonfinite_ignored(tmp_path: Path) -> None:
    data = {"feh": float("nan"), "feh_sigma": 0.1, "mass": None, "mass_sigma": -1.0}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(data))
    spec = psp.load_uberms_prior_spec(p)
    assert spec.feh_mean is None
    assert spec.mass_mean is None
    assert spec.mass_sigma is None  # negative sigma


# ---------------------------------------------------------------------------
# build_prior_spec — uberMS gating
# ---------------------------------------------------------------------------


def test_uberms_gating_1star_applied(tmp_path: Path) -> None:
    """uberMS applied to 1-star model."""
    data = {"feh": -0.3, "feh_sigma": 0.1}
    p = tmp_path / "ums.json"
    p.write_text(json.dumps(data))
    be = _MockBackend(csfd_ebv=0.3)
    spec = psp.build_prior_spec(
        45.0, 30.0, 5.0, 0.1,
        prior_dust=False,
        prior_gaia_vac=False,
        uberms_path=p,
        model="1star",
        backend=be,
    )
    assert spec.uberms is not None
    assert spec.uberms.feh_mean == pytest.approx(-0.3)


def test_uberms_gating_2star_ignored(tmp_path: Path) -> None:
    """uberMS silently skipped for 2-star model (spec.uberms is None)."""
    data = {"feh": -0.3, "feh_sigma": 0.1}
    p = tmp_path / "ums.json"
    p.write_text(json.dumps(data))
    be = _MockBackend(csfd_ebv=0.3)
    spec = psp.build_prior_spec(
        45.0, 30.0, 5.0, 0.1,
        prior_dust=False,
        prior_gaia_vac=False,
        uberms_path=p,
        model="2star",
        backend=be,
    )
    assert spec.uberms is None


def test_uberms_gating_wd_ignored(tmp_path: Path) -> None:
    """uberMS silently skipped for wd model."""
    data = {"mass": 0.6, "mass_sigma": 0.05}
    p = tmp_path / "ums.json"
    p.write_text(json.dumps(data))
    be = _MockBackend(csfd_ebv=0.2)
    spec = psp.build_prior_spec(
        45.0, 30.0, 5.0, 0.1,
        uberms_path=p,
        model="wd",
        backend=be,
    )
    assert spec.uberms is None


# ---------------------------------------------------------------------------
# plx_in_prior flag
# ---------------------------------------------------------------------------


def test_plx_in_prior_false_by_default() -> None:
    be = _MockBackend(csfd_ebv=0.3)
    spec = psp.build_prior_spec(45.0, 30.0, 5.0, 0.1, backend=be)
    assert spec.plx_in_prior is False


def test_plx_in_prior_true_with_gaia_vac() -> None:
    be = _MockBackend(csfd_ebv=0.3)
    spec = psp.build_prior_spec(45.0, 30.0, 5.0, 0.1, prior_gaia_vac=True, backend=be)
    assert spec.plx_in_prior is True


# ---------------------------------------------------------------------------
# unit_to_av flat/gaussian consistency
# ---------------------------------------------------------------------------


def test_unit_to_av_flat_endpoints() -> None:
    spec = psp.AvPriorSpec(kind="flat", av_lo=0.0, av_hi=2.0)
    assert psp.unit_to_av(0.0, spec) == pytest.approx(0.0)
    assert psp.unit_to_av(1.0, spec) == pytest.approx(2.0)
    assert psp.unit_to_av(0.5, spec) == pytest.approx(1.0)


def test_unit_to_av_gaussian_median() -> None:
    spec = psp.AvPriorSpec(
        kind="gaussian", av_lo=0.0, av_hi=2.0, av_mean=0.5, av_sigma=0.1
    )
    # u=0.5 → median of truncated normal ≈ mean for symmetric truncation
    val = psp.unit_to_av(0.5, spec)
    assert 0.0 <= val <= 2.0
    assert abs(val - 0.5) < 0.05


def test_unit_to_av_gaussian_always_bounded() -> None:
    spec = psp.AvPriorSpec(
        kind="gaussian", av_lo=0.0, av_hi=1.5, av_mean=0.4, av_sigma=0.15
    )
    for u in (0.0001, 0.5, 0.9999):
        val = psp.unit_to_av(u, spec)
        assert 0.0 - 1e-9 <= val <= 1.5 + 1e-9, f"u={u}: val={val}"


# ---------------------------------------------------------------------------
# _trunc_normal_icdf edge cases
# ---------------------------------------------------------------------------


def test_trunc_normal_icdf_zero_sigma_linear() -> None:
    val = psp._trunc_normal_icdf(0.3, mu=1.0, sigma=0.0, lo=0.0, hi=2.0)
    assert val == pytest.approx(0.6)  # linear 0.3 * 2.0


def test_trunc_normal_icdf_clamps() -> None:
    val_lo = psp._trunc_normal_icdf(0.0, mu=1.0, sigma=0.5, lo=0.0, hi=2.0)
    val_hi = psp._trunc_normal_icdf(1.0, mu=1.0, sigma=0.5, lo=0.0, hi=2.0)
    assert val_lo >= 0.0
    assert val_hi <= 2.0
