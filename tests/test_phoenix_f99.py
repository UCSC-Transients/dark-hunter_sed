"""Tests for F99 extinction and PHOENIX HiRes grid (mocked tiny λ; no 42GB tree)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from synphot import SpectralElement
from synphot.models import Empirical1D

from darkhunter_sed.config import DEFAULT_PHOENIX_DIR, phoenix_dir
from darkhunter_sed.extinction_f99 import (
    DEFAULT_R_V,
    apply_f99_extinction,
    f99_alambda_over_av,
    f99_model,
)
from darkhunter_sed.phoenix_grid import (
    PhoenixGrid,
    PhoenixPoint,
    phoenix_synth_phot,
    scale_surface_to_earth,
    synthesize_mags,
)


def _tiny_wave() -> np.ndarray:
    return np.linspace(4000.0, 9000.0, 64, dtype=np.float64)


def _flat_bp(wave: np.ndarray, name: str = "mock") -> SpectralElement:
    """Unit-throughput bandpass spanning ``wave`` (for CI without CDBS)."""
    thru = np.ones_like(wave, dtype=np.float64)
    thru[0] = 0.0
    thru[-1] = 0.0
    return SpectralElement(
        Empirical1D,
        points=wave * u.AA,
        lookup_table=thru,
        meta={"name": name},
    )


def test_phoenix_dir_default_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PHOENIX_DIR", raising=False)
    assert phoenix_dir() == DEFAULT_PHOENIX_DIR.expanduser().resolve()
    monkeypatch.setenv("PHOENIX_DIR", str(tmp_path))
    assert phoenix_dir() == tmp_path.resolve()
    assert phoenix_dir(tmp_path / "nested") == (tmp_path / "nested").resolve()


def test_f99_uses_dust_extinction_extinguish() -> None:
    """A_V dims blue flux more than red; matches F99.extinguish directly."""
    wave = _tiny_wave()
    flux = np.ones_like(wave)
    a_v = 1.0
    out = apply_f99_extinction(wave, flux, a_v, r_v=DEFAULT_R_V)
    ext = f99_model(r_v=DEFAULT_R_V)
    expected = ext.extinguish(wave * u.AA, Av=a_v)
    np.testing.assert_allclose(out, expected, rtol=1e-10)
    assert out[0] < out[-1]  # more extinction at blue end
    assert apply_f99_extinction(wave, flux, 0.0)[0] == pytest.approx(1.0)


def test_f99_rejects_negative_av() -> None:
    wave = _tiny_wave()
    with pytest.raises(ValueError, match="a_v"):
        apply_f99_extinction(wave, np.ones_like(wave), -0.1)


def test_f99_edge_hold_outside_validity() -> None:
    """λ outside F99 window uses clipped edge A_λ/A_V (no crash)."""
    wave = np.array([500.0, 5000.0, 50000.0], dtype=np.float64)
    al = f99_alambda_over_av(wave)
    assert np.all(np.isfinite(al))
    assert al[0] == pytest.approx(f99_alambda_over_av(np.array([1000.0]))[0])
    assert al[2] == pytest.approx(f99_alambda_over_av(np.array([1.0e4 / 0.3]))[0])


def _mock_grid() -> PhoenixGrid:
    """2×2×1×2 tiny grid in (Teff, logg, [M/H], [α/Fe]) with analytic fluxes."""
    wave = _tiny_wave()
    # Flux encodes parameters so linear interp is exact to recover.
    # F = teff + 100*logg + 1000*mh + 5000*alpha  (constant spectrum)
    specs: dict[tuple[float, float, float, float], np.ndarray] = {}
    points: list[PhoenixPoint] = []
    for teff in (5000.0, 5200.0):
        for logg in (4.0, 4.5):
            for mh in (0.0,):
                for alpha in (0.0, 0.2):
                    key = (teff, logg, mh, alpha)
                    val = teff + 100.0 * logg + 1000.0 * mh + 5000.0 * alpha
                    specs[key] = np.full_like(wave, val)
                    points.append(
                        PhoenixPoint(
                            teff,
                            logg,
                            mh,
                            alpha,
                            Path(f"/mock/lte{teff:.0f}-{logg:.2f}-{mh}-{alpha}.fits"),
                        )
                    )

    def loader(path: Path) -> np.ndarray:
        # Parse encoded path tokens
        name = path.name
        # lte5000-4.00-0.0-0.0.fits style from Path above
        body = name.replace(".fits", "").removeprefix("lte")
        teff_s, logg_s, mh_s, alpha_s = body.split("-")
        key = (float(teff_s), float(logg_s), float(mh_s), float(alpha_s))
        return specs[key]

    return PhoenixGrid(
        root="/tmp/mock-phoenix",
        wavelength=wave,
        points=points,
        flux_loader=loader,
        to_flam=False,
    )


def test_phoenix_multilinear_interp_recovers_analytic() -> None:
    grid = _mock_grid()
    wave, flux = grid.spectrum(5100.0, 4.25, 0.0, 0.1, interpolate=True)
    assert wave.shape == flux.shape
    expected = 5100.0 + 100.0 * 4.25 + 1000.0 * 0.0 + 5000.0 * 0.1
    np.testing.assert_allclose(flux, expected, rtol=1e-10)


def test_phoenix_alpha_required_when_missing() -> None:
    wave = _tiny_wave()
    pts = [
        PhoenixPoint(5000.0, 4.5, 0.0, 0.0, Path("/mock/a.fits")),
        PhoenixPoint(5200.0, 4.5, 0.0, 0.0, Path("/mock/b.fits")),
        PhoenixPoint(5000.0, 4.0, 0.0, 0.0, Path("/mock/c.fits")),
        PhoenixPoint(5200.0, 4.0, 0.0, 0.0, Path("/mock/d.fits")),
    ]
    fluxes = {
        Path("/mock/a.fits"): np.ones_like(wave),
        Path("/mock/b.fits"): np.ones_like(wave) * 2,
        Path("/mock/c.fits"): np.ones_like(wave) * 3,
        Path("/mock/d.fits"): np.ones_like(wave) * 4,
    }
    grid = PhoenixGrid(
        wavelength=wave,
        points=pts,
        flux_loader=lambda p: fluxes[p],
        to_flam=False,
    )
    with pytest.raises(ValueError, match=r"α/Fe"):
        grid.spectrum(5100.0, 4.25, 0.0, 0.2)


def test_extincted_spectrum_then_synth_mags() -> None:
    grid = _mock_grid()
    wave = grid.wavelength
    bp = _flat_bp(wave, "mock_band")
    mags0 = phoenix_synth_phot(
        grid,
        5100.0,
        4.25,
        0.0,
        0.0,
        a_v=0.0,
        bands=["mock_band"],
        systems=("ab", "vega"),
        bandpasses={"mock_band": bp},
    )
    mags1 = phoenix_synth_phot(
        grid,
        5100.0,
        4.25,
        0.0,
        0.0,
        a_v=1.0,
        bands=["mock_band"],
        systems=("ab", "vega"),
        bandpasses={"mock_band": bp},
    )
    assert "ab" in mags0["mock_band"] and "vega" in mags0["mock_band"]
    # Extinction makes source fainter → larger (fainter) AB mag.
    assert mags1["mock_band"]["ab"] > mags0["mock_band"]["ab"]


def test_synthesize_mags_ab_vega_keys() -> None:
    wave = _tiny_wave()
    flux = np.full_like(wave, 1.0e-14)
    bp = _flat_bp(wave)
    out = synthesize_mags(
        wave,
        flux,
        ["X"],
        systems=("ab", "vega"),
        bandpasses={"X": bp},
    )
    assert set(out["X"]) == {"ab", "vega"}
    assert np.isfinite(out["X"]["ab"]) and np.isfinite(out["X"]["vega"])


def test_abmag_matches_synphot_observation() -> None:
    """Photon-weighted numpy AB matches synphot Observation on a short grid."""
    from synphot import Observation, SourceSpectrum
    from synphot.models import Empirical1D
    from synphot.units import FLAM

    from darkhunter_sed.phoenix_grid import _abmag_from_flam_on_bandpass

    wave = _tiny_wave()
    flux = np.full_like(wave, 1.0e-14)
    bp = _flat_bp(wave, "X")
    thru = np.asarray(bp(bp.waveset).value, dtype=np.float64)
    wave_bp = np.asarray(bp.waveset.to(u.AA).value, dtype=np.float64)
    m_np = _abmag_from_flam_on_bandpass(wave, flux, wave_bp, thru)
    src = SourceSpectrum(Empirical1D, points=wave * u.AA, lookup_table=flux * FLAM)
    m_sp = float(Observation(src, bp, force="taper").effstim("abmag").value)
    assert m_np == pytest.approx(m_sp, abs=1e-4)


def test_synthesize_mags_bandpass_native_fast_on_long_wave() -> None:
    """Full-HiRes-length array must not make synth take seconds (bandpass-native)."""
    import time

    wave = np.linspace(3000.0, 11000.0, 200_000, dtype=np.float64)
    flux = np.full_like(wave, 1.0e-14)
    bp_wave = np.linspace(4800.0, 5800.0, 200, dtype=np.float64)
    bp = _flat_bp(bp_wave, "g")
    t0 = time.perf_counter()
    out = synthesize_mags(wave, flux, ["g"], systems=("ab",), bandpasses={"g": bp})
    dt = time.perf_counter() - t0
    assert np.isfinite(out["g"]["ab"])
    assert dt < 2.0, f"bandpass-native synth too slow: {dt:.2f}s"


def test_phoenix_flux_lru_cache() -> None:
    wave = _tiny_wave()
    path_a = Path("/mock/cache_a.fits")
    path_b = Path("/mock/cache_b.fits")
    path_c = Path("/mock/cache_c.fits")
    path_d = Path("/mock/cache_d.fits")
    loads: list[Path] = []
    fluxes = {
        path_a: np.ones_like(wave),
        path_b: np.ones_like(wave) * 2,
        path_c: np.ones_like(wave) * 3,
        path_d: np.ones_like(wave) * 4,
    }

    def loader(p: Path) -> np.ndarray:
        loads.append(p)
        return fluxes[p]

    pts = [
        PhoenixPoint(5000.0, 4.5, 0.0, 0.0, path_a),
        PhoenixPoint(5200.0, 4.5, 0.0, 0.0, path_b),
        PhoenixPoint(5000.0, 4.0, 0.0, 0.0, path_c),
        PhoenixPoint(5200.0, 4.0, 0.0, 0.0, path_d),
    ]
    grid = PhoenixGrid(
        wavelength=wave,
        points=pts,
        flux_loader=loader,
        to_flam=False,
        flux_cache_size=8,
    )
    grid.spectrum(5100.0, 4.25, 0.0, 0.0, interpolate=True)
    n_after_first = len(loads)
    assert n_after_first == 4
    grid.spectrum(5100.0, 4.25, 0.0, 0.0, interpolate=True)
    assert len(loads) == n_after_first  # corners served from LRU


def test_scale_surface_to_earth() -> None:
    flux = np.array([1.0, 2.0])
    # R = d → geometric factor 1
    d_pc = 1.0
    r_cm = 3.0856775814913673e18  # 1 pc in cm
    out = scale_surface_to_earth(flux, radius_cm=r_cm, distance_pc=d_pc)
    np.testing.assert_allclose(out, flux, rtol=1e-12)
