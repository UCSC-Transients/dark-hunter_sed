"""Compact PHOENIX grid: rebinning, naming, FITS I/O, resume, loader auto-detect (no network)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from synphot import SpectralElement
from synphot.models import Empirical1D

from darkhunter_sed.phoenix_compact import (
    GRID_SUBDIR,
    HIRES_WAVE_NAME,
    bin_centers,
    build_compact_from_hires,
    compact_flux_is_valid,
    log_lambda_edges,
    make_jobs,
    parse_directory_listing,
    read_compact_wave,
    rebin_flux_conserving,
    split_done,
    write_compact_wave,
)
from darkhunter_sed.phoenix_grid import (
    PhoenixGrid,
    compact_flux_name,
    compact_wave_name,
    detect_phoenix_layout,
    parse_phoenix_filename,
    parse_subgrid_name,
    phoenix_model_path,
    phoenix_synth_phot,
)

REPO = Path(__file__).resolve().parent.parent
R_TEST = 400
TEFFS = (5000.0, 5200.0)
LOGGS = (4.0, 4.5)


def _line_spectrum(wave: np.ndarray, teff: float, seed: int) -> np.ndarray:
    """Smooth continuum with many narrow absorption lines (per-cm flux units)."""
    rng = np.random.default_rng(seed)
    cont = 1.0e7 * (wave / 5000.0) ** -2 * (teff / 5000.0) ** 4
    centers = rng.uniform(wave[0], wave[-1], 300)
    sig = rng.uniform(0.2, 2.0, 300)
    depth = rng.uniform(0.05, 0.9, 300)
    prof = depth[:, None] * np.exp(-0.5 * ((wave[None, :] - centers[:, None]) / sig[:, None]) ** 2)
    return cont * np.exp(-prof.sum(axis=0))


def _trapz_between(wave: np.ndarray, flux: np.ndarray, a: float, b: float) -> float:
    """Trapezoidal integral of linear-interpolated ``flux`` over ``[a, b]``."""
    x = np.union1d(wave[(wave > a) & (wave < b)], [a, b])
    return float(np.trapz(np.interp(x, wave, flux), x))


def _hires_name(teff: float, logg: float, sep: str = "-") -> str:
    return f"lte{int(teff):05d}{sep}{logg:.2f}-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"


@pytest.fixture()
def hires_tree(tmp_path: Path) -> Path:
    """Tiny HiRes-layout tree: non-uniform λ grid, 2×2 models in Z-0.0 with PHX cards."""
    root = tmp_path / "HiResFITS"
    sub = root / GRID_SUBDIR / "Z-0.0"
    sub.mkdir(parents=True)
    wave = np.unique(np.concatenate([np.arange(1000.0, 3000.0, 0.5), np.geomspace(3000.0, 20000.0, 6000)]))
    fits.PrimaryHDU(wave).writeto(root / HIRES_WAVE_NAME)
    for i, teff in enumerate(TEFFS):
        for j, logg in enumerate(LOGGS):
            hdu = fits.PrimaryHDU(_line_spectrum(wave, teff, 10 * i + j).astype(np.float32))
            hdu.header["BUNIT"] = "erg/s/cm^2/cm"
            hdu.header["PHXTEFF"] = (teff, "[K] effective temperature")
            hdu.header["PHXLOGG"] = (logg, "[cm/s^2] log (surface gravity)")
            hdu.header["PHXM_H"] = (0.0, "[M/H] metallicity")
            hdu.header["PHXALPHA"] = (0.0, "[a/M] alpha element enhancement")
            hdu.header["PHXREFF"] = (6.96e10, "[cm] effective stellar radius")
            hdu.writeto(sub / _hires_name(teff, logg))
    return root


def _build_compact(hires_root: Path, dest: Path, resolution: int = R_TEST) -> np.ndarray:
    wave = np.asarray(fits.getdata(hires_root / HIRES_WAVE_NAME), dtype=np.float64)
    edges = log_lambda_edges(wave[0], wave[-1], resolution)
    write_compact_wave(dest / compact_wave_name(resolution), edges, resolution=resolution,
                       source_wave=HIRES_WAVE_NAME)
    names = [p.name for p in sorted((hires_root / GRID_SUBDIR / "Z-0.0").iterdir())]
    for job in make_jobs(dest, "Z-0.0", names, resolution=resolution):
        build_compact_from_hires(hires_root / GRID_SUBDIR / "Z-0.0" / job.hires_name,
                                 job.compact_path, native_wave=wave, edges=edges,
                                 resolution=resolution)
    return edges


def test_log_lambda_edges_constant_resolution() -> None:
    edges = log_lambda_edges(500.0, 55000.0, 2000)
    assert edges[0] == 500.0
    assert edges[-1] <= 55000.0
    assert edges.size - 1 == int(np.floor(np.log(55000.0 / 500.0) * 2000))
    np.testing.assert_allclose(np.diff(np.log(edges)), 1.0 / 2000, rtol=1e-9)
    centers = bin_centers(edges)
    assert np.all((centers > edges[:-1]) & (centers < edges[1:]))
    np.testing.assert_allclose(centers / np.diff(edges), 2000.0, rtol=1e-3)
    with pytest.raises(ValueError):
        log_lambda_edges(5000.0, 5001.0, 10)
    with pytest.raises(ValueError):
        log_lambda_edges(-1.0, 10.0, 100)


def test_rebin_conserves_flux_full_and_partial_range() -> None:
    rng = np.random.default_rng(0)
    wave = np.sort(rng.uniform(1000.0, 20000.0, 50_000))
    flux = _line_spectrum(wave, 5000.0, 1)
    edges = log_lambda_edges(wave[0], wave[-1], 300)
    binned = rebin_flux_conserving(wave, flux, edges)
    total = float(np.sum(binned * np.diff(edges)))
    ref = _trapz_between(wave, flux, edges[0], edges[-1])
    assert total == pytest.approx(ref, rel=1e-10)
    stored = binned.astype(np.float32).astype(np.float64)
    assert float(np.sum(stored * np.diff(edges))) == pytest.approx(ref, rel=1e-6)

    inner = log_lambda_edges(1234.5, 17000.3, 250)
    ref_bins = np.array([_trapz_between(wave, flux, a, b) for a, b in zip(inner[:-1], inner[1:])])
    np.testing.assert_allclose(rebin_flux_conserving(wave, flux, inner) * np.diff(inner),
                               ref_bins, rtol=1e-9)


def test_rebin_linear_flux_exact_and_rejects_outside() -> None:
    wave = np.linspace(1000.0, 2000.0, 101)
    flux = 3.0 + 0.01 * wave
    edges = np.array([1000.0, 1003.3, 1500.0, 1999.0])
    mids = 0.5 * (edges[:-1] + edges[1:])
    np.testing.assert_allclose(rebin_flux_conserving(wave, flux, edges), 3.0 + 0.01 * mids, rtol=1e-12)
    with pytest.raises(ValueError, match="outside native coverage"):
        rebin_flux_conserving(wave, flux, np.array([999.0, 1500.0]))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits", (5000.0, 4.5, 0.0, 0.0, None)),
        ("lte04000+0.00-0.0.Alpha=+1.00.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
         (4000.0, 0.0, 0.0, 1.0, None)),
        ("lte12000-6.00+0.5.Alpha=-0.20.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
         (12000.0, 6.0, 0.5, -0.2, None)),
        ("lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-R2000.fits", (5000.0, 4.5, 0.0, 0.0, 2000)),
        ("lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-MedRes.fits", None),
        ("WAVE_PHOENIX-ACES-AGSS-COND-2011.fits", None),
    ],
)
def test_parse_phoenix_filename(name: str, expected: tuple | None) -> None:
    tags = parse_phoenix_filename(name)
    if expected is None:
        assert tags is None
        return
    assert tags is not None
    assert (tags.teff_k, tags.logg, tags.mh, tags.alpha) == pytest.approx(expected[:4])
    assert tags.resolution == expected[4]


def test_compact_naming_preserves_logg_token() -> None:
    hires = "lte04000+0.00-0.0.Alpha=+1.00.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    compact = compact_flux_name(hires, 2000)
    assert compact == "lte04000+0.00-0.0.Alpha=+1.00.PHOENIX-ACES-AGSS-COND-2011-R2000.fits"
    assert parse_phoenix_filename(compact).resolution == 2000
    assert compact_wave_name(2000) == "WAVE_PHOENIX-ACES-AGSS-COND-2011-R2000.fits"
    assert phoenix_model_path("/r", 5000, 4.5, 0.0, resolution=2000).name.endswith("-R2000.fits")
    with pytest.raises(ValueError):
        compact_flux_name(compact, 2000)
    assert parse_subgrid_name("Z-0.5.Alpha=+0.40") == (-0.5, 0.4)
    assert parse_subgrid_name("PHOENIX-ACES-AGSS-COND-2011") is None


def test_parse_directory_listing_decodes_url_escapes() -> None:
    page = """
    <a href="?C=N;O=D">Name</a> <a href="/pub/misc/phoenix/">Parent Directory</a>
    <a href="../">../</a>
    <a href="Z-0.0/">Z-0.0/</a>
    <a href="Z-0.0.Alpha%3D%2B0.40/">Z-0.0.Alpha=+0.40/</a>
    <a href="lte04000%2B0.00-0.0.Alpha%3D%2B1.00.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits">x</a>
    <a href="lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits">y</a>
    <a href="https://example.org/elsewhere">z</a>
    """
    entries = parse_directory_listing(page)
    assert [(e.name, e.is_dir) for e in entries] == [
        ("Z-0.0", True),
        ("Z-0.0.Alpha=+0.40", True),
        ("lte04000+0.00-0.0.Alpha=+1.00.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits", False),
        ("lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits", False),
    ]
    tags = parse_phoenix_filename(entries[2].name)
    assert tags is not None and tags.logg == 0.0 and tags.alpha == 1.0


def test_make_jobs_filters_and_ignores_non_hires(tmp_path: Path) -> None:
    names = [
        _hires_name(4000.0, 0.0, sep="+"),
        _hires_name(5000.0, 4.5),
        _hires_name(9000.0, 4.5),
        "lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-R2000.fits",
        "index.html",
    ]
    jobs = make_jobs(tmp_path, "Z-0.0", names, resolution=2000, teff_max=6000.0, logg_min=0.0)
    assert [j.hires_name for j in jobs] == names[:2]
    assert jobs[0].compact_path == (
        tmp_path / GRID_SUBDIR / "Z-0.0" / "lte04000+0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-R2000.fits"
    )
    assert make_jobs(tmp_path, "Z-0.0", names, resolution=2000, logg_min=1.0)[0].tags.teff_k == 5000.0


def test_compact_files_headers_and_wave(hires_tree: Path, tmp_path: Path) -> None:
    dest = tmp_path / "CompactFITS"
    edges = _build_compact(hires_tree, dest)
    centers, edges_rt, res = read_compact_wave(dest / compact_wave_name(R_TEST))
    assert res == R_TEST
    np.testing.assert_array_equal(edges_rt, edges)
    np.testing.assert_allclose(centers, bin_centers(edges))
    path = dest / GRID_SUBDIR / "Z-0.0" / compact_flux_name(_hires_name(5000.0, 4.5), R_TEST)
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        assert hdul[0].data.dtype == np.dtype(">f4")
        assert hdul[0].data.shape == (edges.size - 1,)
    assert hdr["PHXTEFF"] == 5000.0 and hdr["PHXLOGG"] == 4.5 and hdr["PHXREFF"] == 6.96e10
    assert hdr.comments["PHXTEFF"] == "[K] effective temperature"
    assert hdr["BUNIT"] == "erg/s/cm^2/cm"
    assert hdr["CMPRES"] == R_TEST
    assert hdr["CMPSRC"] == _hires_name(5000.0, 4.5)
    assert "flux-conserving" in hdr["CMPMETH"]
    native_wave = np.asarray(fits.getdata(hires_tree / HIRES_WAVE_NAME), dtype=np.float64)
    native = np.asarray(fits.getdata(hires_tree / GRID_SUBDIR / "Z-0.0" / _hires_name(5000.0, 4.5)),
                        dtype=np.float64)
    stored = np.asarray(fits.getdata(path), dtype=np.float64)
    assert float(np.sum(stored * np.diff(edges))) == pytest.approx(
        _trapz_between(native_wave, native, edges[0], edges[-1]), rel=1e-6
    )


def test_resume_validity_and_split_done(hires_tree: Path, tmp_path: Path) -> None:
    dest = tmp_path / "CompactFITS"
    edges = _build_compact(hires_tree, dest)
    n_bins = edges.size - 1
    names = [_hires_name(t, g) for t in TEFFS for g in LOGGS]
    jobs = make_jobs(dest, "Z-0.0", names, resolution=R_TEST)
    todo, done = split_done(jobs, n_bins=n_bins, resolution=R_TEST)
    assert not todo and len(done) == 4
    assert split_done(jobs, n_bins=None, resolution=R_TEST)[0] == jobs

    good, corrupt, truncated, missing = (j.compact_path for j in jobs)
    raw = bytearray(corrupt.read_bytes())
    raw[-100] ^= 0xFF
    corrupt.write_bytes(bytes(raw))
    truncated.write_bytes(truncated.read_bytes()[:5000])
    missing.unlink()
    assert compact_flux_is_valid(good, n_bins=n_bins, resolution=R_TEST)
    assert not compact_flux_is_valid(good, n_bins=n_bins, resolution=R_TEST + 1)
    assert not compact_flux_is_valid(good, n_bins=n_bins - 1, resolution=R_TEST)
    todo, done = split_done(jobs, n_bins=n_bins, resolution=R_TEST)
    assert [j.compact_path for j in done] == [good]
    assert [j.compact_path for j in todo] == [corrupt, truncated, missing]
    assert not list(dest.rglob("*.part"))


def test_loader_autodetects_compact_and_hires(hires_tree: Path, tmp_path: Path) -> None:
    hires_grid = PhoenixGrid(root=hires_tree)
    assert hires_grid.layout is not None and hires_grid.layout.kind == "hires"
    assert hires_grid.wavelength.size == fits.getdata(hires_tree / HIRES_WAVE_NAME).size

    dest = tmp_path / "CompactFITS"
    edges = _build_compact(hires_tree, dest)
    grid = PhoenixGrid(root=dest)
    assert grid.layout is not None and grid.layout.kind == "compact"
    assert grid.layout.resolution == R_TEST
    assert grid.wavelength.size == edges.size - 1
    assert grid.coverage == (edges[0], edges[-1])
    assert len(grid.points) == 4
    wave, flux = grid.spectrum(5100.0, 4.25, 0.0)
    assert wave.shape == flux.shape and np.all(np.isfinite(flux)) and np.all(flux > 0)

    # Same root holding both layouts: compact wins and HiRes spectra are not indexed.
    _build_compact(hires_tree, hires_tree)
    mixed = PhoenixGrid(root=hires_tree)
    assert mixed.layout.kind == "compact"
    assert all(p.path.name.endswith(f"-R{R_TEST}.fits") for p in mixed.points)

    write_compact_wave(hires_tree / compact_wave_name(R_TEST + 1), edges, resolution=R_TEST + 1,
                       source_wave="x")
    with pytest.raises(ValueError, match="Multiple compact"):
        detect_phoenix_layout(hires_tree)
    with pytest.raises(FileNotFoundError, match="build_phoenix_compact"):
        detect_phoenix_layout(tmp_path / "empty")


def test_compact_phot_matches_hires_and_zero_outside(hires_tree: Path, tmp_path: Path) -> None:
    dest = tmp_path / "CompactFITS"
    _build_compact(hires_tree, dest, resolution=2000)
    wbp = np.arange(4000.0, 9001.0, 5.0)
    thru = np.exp(-0.5 * ((wbp - 6500.0) / 900.0) ** 2)
    bps = {"mock": SpectralElement(Empirical1D, points=wbp * u.AA, lookup_table=thru)}
    mags = []
    for root in (hires_tree, dest):
        g = PhoenixGrid(root=root)
        mags.append(phoenix_synth_phot(g, 5100.0, 4.25, 0.0, 0.0, 0.3, ["mock"], bandpasses=bps)
                    ["mock"]["ab"])
    assert abs(mags[0] - mags[1]) < 2e-3

    grid = PhoenixGrid(root=dest)
    lo, hi = grid.coverage
    pw = np.array([lo - 50.0, lo + 10.0, 5000.0, hi - 10.0, hi + 50.0])
    grid.set_photometry_wavelengths(pw)
    _, flux = grid.extincted_spectrum(5000.0, 4.0, 0.0, 0.0, 0.0)
    assert flux[0] == 0.0 and flux[-1] == 0.0
    assert np.all(flux[1:-1] > 0.0)


def _load_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_phoenix_compact", REPO / "scripts" / "build_phoenix_compact.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_cli_local_build_is_resumable(hires_tree: Path, tmp_path: Path,
                                      capsys: pytest.CaptureFixture[str],
                                      monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load_script(monkeypatch)
    dest = tmp_path / "out"
    base = ["--src-local", str(hires_tree), "--dest", str(dest), "--resolution", str(R_TEST)]
    assert script.main(base + ["--dry-run"]) == 0
    assert "0 already valid, 4 to build" in capsys.readouterr().out
    assert not dest.exists()

    assert script.main(base + ["--workers", "2", "--teff-max", "5000"]) == 0
    out = capsys.readouterr().out
    assert "built: 2" in out
    assert script.main(base) == 0
    out = capsys.readouterr().out
    assert "2 already valid, 2 to build" in out and "built: 2" in out
    assert script.main(base) == 0
    assert "built: 0  skipped(valid): 4" in capsys.readouterr().out
    assert len(PhoenixGrid(root=dest).points) == 4
    assert not list(dest.glob(".tmp-build-*"))
    assert sorted((hires_tree / GRID_SUBDIR / "Z-0.0").iterdir())  # local source untouched

    with pytest.raises(SystemExit):
        script.main(["--src-local", str(hires_tree), "--dest", str(dest), "--resolution", "999"])
