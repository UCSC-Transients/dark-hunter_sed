"""
Magnitude-system (AB vs Vega) handling for Path-2 photometry (issue #68).

Covers the per-band registry in ``filters_synphot``, the one-time row conversion
in ``phot_sed_io``, Bergeron WD tables converted to AB, a WD+star likelihood that
sums WD and companion in one system, and that the Path-1 loader stays native.
No network or PHOENIX data needed; the synphot Vega check skips when CALSPEC /
CDBS thruputs are not available locally.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darkhunter_sed import misty_iso, phoenix_grid, stellar_data
from darkhunter_sed.bergeron_wd import _BAND_REGISTRY_DA, BergeronGrid
from darkhunter_sed.cummings_ifmr import CummingsIFMR
from darkhunter_sed.filters_synphot import (
    BAND_REGISTRY,
    MAG_SYSTEM_REGISTRY,
    ab_minus_native,
    load_bandpass,
    nonhst_dir,
)
from darkhunter_sed.phot_sed_io import (
    FLAG_DETECTION,
    FLAG_UPPER_LIMIT,
    PhotRow,
    convert_rows_to_ab,
    read_photometry_fits,
    write_photometry_fits,
)
from darkhunter_sed.wd_model import WDStarPriorBounds, _make_wdstar_loglike

_REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_registry_covers_every_path2_and_bergeron_band() -> None:
    bergeron_bands = {entry[1] for entry in _BAND_REGISTRY_DA}
    missing = (set(BAND_REGISTRY) | bergeron_bands) - set(MAG_SYSTEM_REGISTRY)
    assert not missing


def test_registry_entries_are_cited_and_consistent() -> None:
    for band, spec in MAG_SYSTEM_REGISTRY.items():
        assert spec.reference.strip(), band
        assert spec.system in ("AB", "Vega"), band
        if spec.system == "AB":
            assert spec.ab_minus_native == 0.0, band
        else:
            assert spec.ab_minus_native != 0.0, band


@pytest.mark.parametrize(
    ("band", "offset"),
    [
        ("GaiaDR3_G", 0.1136),
        ("GaiaDR3_BP", 0.0155),
        ("GaiaDR3_RP", 0.3561),
        ("2MASS_J", 0.91),
        ("2MASS_H", 1.39),
        ("2MASS_Ks", 1.85),
        ("WISE_W1", 2.699),
        ("WISE_W2", 3.339),
    ],
)
def test_vega_band_offsets(band: str, offset: float) -> None:
    assert MAG_SYSTEM_REGISTRY[band].system == "Vega"
    assert MAG_SYSTEM_REGISTRY[band].ab_minus_native == pytest.approx(offset, abs=1e-9)


@pytest.mark.parametrize("band", ["PS_g", "SDSS_u", "GALEX_NUV", "DECam_u"])
def test_ab_bands_have_zero_offset(band: str) -> None:
    assert MAG_SYSTEM_REGISTRY[band].system == "AB"


def test_ab_minus_native_unknown_band_raises() -> None:
    with pytest.raises(KeyError):
        ab_minus_native(["GaiaDR3_G", "NOT_A_BAND"])


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------
def _native_rows() -> list[PhotRow]:
    return [
        PhotRow("Gaia_parallax", 5.0, 0.1, FLAG_DETECTION),
        PhotRow("GaiaDR3_G", 12.0, 0.01, FLAG_DETECTION),
        PhotRow("2MASS_Ks", 10.0, 0.02, FLAG_DETECTION),
        PhotRow("WISE_W2", 15.0, 0.3, FLAG_UPPER_LIMIT),
        PhotRow("PS_g", 12.5, 0.02, FLAG_DETECTION),
        PhotRow("MYSTERY_BAND", 9.0, 0.1, FLAG_DETECTION),
    ]


def test_convert_rows_shifts_detections_and_upper_limits_once() -> None:
    rows = _native_rows()
    once = convert_rows_to_ab(rows)
    twice = convert_rows_to_ab(once)
    assert twice == once

    by_band = {r.band: r for r in once}
    assert by_band["GaiaDR3_G"].mag == pytest.approx(12.0 + 0.1136)
    assert by_band["2MASS_Ks"].mag == pytest.approx(10.0 + 1.85)
    assert by_band["WISE_W2"].mag == pytest.approx(15.0 + 3.339)
    assert by_band["WISE_W2"].flag == FLAG_UPPER_LIMIT
    assert by_band["PS_g"].mag == 12.5
    for band in ("GaiaDR3_G", "2MASS_Ks", "WISE_W2", "PS_g"):
        assert by_band[band].system == "AB"

    for orig, new in zip(rows, once):
        assert new.err == orig.err
        assert new.band == orig.band

    assert by_band["Gaia_parallax"] == rows[0]
    assert by_band["MYSTERY_BAND"] == rows[-1]


def test_fits_round_trip_stays_native_and_rejects_converted(tmp_path: Path) -> None:
    path = tmp_path / "123_phot.fits"
    write_photometry_fits(path, _native_rows())
    loaded = read_photometry_fits(path)
    assert loaded == _native_rows()

    converted = convert_rows_to_ab(loaded)
    assert read_photometry_fits(path) == _native_rows()
    with pytest.raises(ValueError, match="native"):
        write_photometry_fits(tmp_path / "bad_phot.fits", converted)


def test_path1_loader_reads_native_magnitudes(tmp_path: Path) -> None:
    write_photometry_fits(tmp_path / "123_phot.fits", _native_rows())
    phot, _ = stellar_data.load_photometry_fits("123", tmp_path)
    assert phot[stellar_data.standardize_band_name("GaiaDR3_G")][0] == 12.0
    assert phot[stellar_data.standardize_band_name("2MASS_Ks")][0] == 10.0


@pytest.mark.parametrize(
    "rel", ["darkhunter_sed/data.py", "darkhunter_sed/cli.py", "darkhunter_sed/batch.py",
            "darkhunter_sed/stellar_data.py"],
)
def test_path1_modules_do_not_convert(rel: str) -> None:
    text = (_REPO / rel).read_text()
    assert "convert_rows_to_ab" not in text
    assert "MAG_SYSTEM_REGISTRY" not in text


# ---------------------------------------------------------------------------
# Bergeron grid → AB
# ---------------------------------------------------------------------------
_NATIVE_WD_MAG = 12.0
_N_BERGERON_MAG_COLS = 38


def _write_flat_bergeron_dir(root: Path) -> Path:
    """Bergeron-format Table_DA/Table_DB with every magnitude = 12.0 (native)."""
    header = (
        "    5    3  Flat test grid\n"
        " Teff  log g  M/Mo  Mbol  BC  <38 mags>  Age\n"
    )
    lines = []
    for logg, mass in zip((7.0, 7.5, 8.0, 8.5, 9.0), (0.4, 0.55, 0.7, 0.9, 1.2)):
        for teff in (5000.0, 10000.0, 20000.0):
            vals = [teff, logg, mass, 12.0, -0.3] + [_NATIVE_WD_MAG] * _N_BERGERON_MAG_COLS
            lines.append(" ".join(f"{v:.4f}" for v in vals) + " 5.000E+08")
    body = header + "\n".join(lines) + "\n"
    (root / "Table_DA").write_text(body)
    (root / "Table_DB").write_text(body)
    return root


def test_bergeron_grid_returns_ab(tmp_path: Path) -> None:
    grid = BergeronGrid.from_dir(_write_flat_bergeron_dir(tmp_path), "DA")
    mags = grid.synth_phot(10000.0, 8.0, a_v=0.0, distance_pc=10.0)["mags"]
    for band, mag in mags.items():
        expected = _NATIVE_WD_MAG + MAG_SYSTEM_REGISTRY[band].ab_minus_native
        assert mag == pytest.approx(expected, abs=1e-9), band


def test_wdstar_likelihood_sums_wd_and_companion_in_ab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed native rows built from an AB WD+companion sum are fit exactly."""
    grid = BergeronGrid.from_dir(_write_flat_bergeron_dir(tmp_path), "DA")
    star_ab = 11.5
    bands = ["GaiaDR3_G", "2MASS_Ks", "WISE_W1", "PS_r"]

    monkeypatch.setattr(
        misty_iso,
        "evaluate_mist",
        lambda *a, **k: SimpleNamespace(teff_k=5800.0, logg=4.4, radius_cm=7e10, age_gyr=13.0),
    )
    monkeypatch.setattr(
        phoenix_grid,
        "phoenix_synth_phot",
        lambda grid_, teff, logg, feh, afe, av, bs, **k: {b: {"ab": star_ab} for b in bs},
    )

    offsets = ab_minus_native(bands)
    wd_ab = _NATIVE_WD_MAG + offsets
    total_ab = -2.5 * np.log10(10.0 ** (-0.4 * star_ab) + 10.0 ** (-0.4 * wd_ab))
    native = [PhotRow(b, float(m), 0.02, FLAG_DETECTION) for b, m in zip(bands, total_ab - offsets)]
    native.append(PhotRow("Gaia_parallax", 100.0, 1.0, FLAG_DETECTION))
    write_photometry_fits(tmp_path / "1_phot.fits", native)
    obs_ab = convert_rows_to_ab(read_photometry_fits(tmp_path / "1_phot.fits"))

    theta = np.array([350.0, 1.0, 0.0, 0.0, 100.0, 10000.0, 8.0, 0.0])
    err_eff = math.sqrt(0.02**2 + 0.01**2)
    ideal = len(bands) * -math.log(math.sqrt(2.0 * math.pi) * err_eff)

    def lnl(rows: list[PhotRow]) -> float:
        loglike, _ = _make_wdstar_loglike(
            grid, CummingsIFMR("MIST"), rows, WDStarPriorBounds(),
            mist_predictor=None, phoenix_grid=object(), bandpasses=None,
        )
        return float(loglike(theta))

    assert lnl(obs_ab) == pytest.approx(ideal, abs=1e-6)
    assert lnl(read_photometry_fits(tmp_path / "1_phot.fits")) < ideal - 100.0


# ---------------------------------------------------------------------------
# Registry offsets vs synphot synthetic AB(Vega)
# ---------------------------------------------------------------------------
def _local_vega_file() -> Path | None:
    calspec = nonhst_dir().parent.parent / "calspec"
    candidates = sorted(calspec.glob("alpha_lyr_stis_0*.fits"))
    return candidates[-1] if candidates else None


_VEGA_FILE = _local_vega_file()


@pytest.mark.skipif(_VEGA_FILE is None, reason="CALSPEC alpha_lyr_stis_*.fits not available")
@pytest.mark.parametrize(
    "band",
    ["GaiaDR3_G", "GaiaDR3_BP", "GaiaDR3_RP", "2MASS_J", "2MASS_H", "2MASS_Ks",
     "WISE_W1", "WISE_W2"],
)
def test_registry_offset_matches_synthetic_vega(band: str) -> None:
    """AB mag of Vega through each bandpass ≈ registry offset.

    Gaia's VEGAMAG zero point puts Vega at V=0.023 (not 0), so Gaia bands sit
    ~0.01–0.025 mag above the ZP difference; 0.03 mag covers that and CALSPEC
    version differences.
    """
    synphot = pytest.importorskip("synphot")
    units = pytest.importorskip("astropy.units")
    try:
        bp = load_bandpass(band)
    except FileNotFoundError:
        pytest.skip(f"{band} thruput not in local CDBS")
    vega = synphot.SourceSpectrum.from_file(str(_VEGA_FILE))
    obs = synphot.Observation(vega, bp, force="taper")
    ab_of_vega = float(obs.effstim(units.ABmag).value)
    assert ab_of_vega == pytest.approx(MAG_SYSTEM_REGISTRY[band].ab_minus_native, abs=0.03)
